"""Search and evidence collection utilities using Tavily API."""
import os
import time
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import concurrent.futures
import bs4
import requests
import spacy
import torch
from spacy.cli import download
from sentence_transformers import CrossEncoder
from urllib3.exceptions import InsecureRequestWarning

warnings.filterwarnings("ignore", category=InsecureRequestWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# spaCy setup
# ---------------------------------------------------------------------------
try:
    TOKENIZER = spacy.load("en_core_web_sm")
except OSError:
    logger.info("Downloading spaCy model...")
    download("en_core_web_sm")
    TOKENIZER = spacy.load("en_core_web_sm")

if "sentencizer" not in TOKENIZER.pipe_names:
    TOKENIZER.add_pipe("sentencizer")

# ---------------------------------------------------------------------------
# Cross-encoder for passage ranking
# ---------------------------------------------------------------------------
PASSAGE_RANKER = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    max_length=512,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# ---------------------------------------------------------------------------
# Tavily configuration
# ---------------------------------------------------------------------------
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_SEARCH_URL = "https://api.tavily.com/search"

# ---------------------------------------------------------------------------
# Serper (Google) configuration  -- primary retriever
# ---------------------------------------------------------------------------
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
# Use "search" for general Google, or "scholar" for Google Scholar.
SERPER_ENDPOINT = "search"   # change to "scholar" for academic-only results
SERPER_URL = f"https://google.serper.dev/{SERPER_ENDPOINT}"


def search_serper(query: str, max_results: int = 5):
    """Query Serper (Google) and return dicts with 'url', 'content', 'title' --
    the same shape run_search expects, so it is a drop-in for search_tavily."""
    if not SERPER_API_KEY:
        logger.error("SERPER_API_KEY not set. Cannot perform search.")
        return []
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "num": max_results}
    try:
        resp = requests.post(SERPER_URL, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        organic = data.get("organic", []) or []
        results = []
        for r in organic[:max_results]:
            results.append({
                "url": r.get("link", ""),
                "content": (r.get("snippet", "") or "").strip(),
                "title": r.get("title", ""),
            })
        logger.info(f"Serper ({SERPER_ENDPOINT}) returned {len(results)} results for: {query}")
        return results
    except Exception as e:
        logger.error(f"Serper search failed: {e}")
        return []


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = ' '.join(text.split())
    text = text.replace("..", ".").replace("!.", "!").replace("?.", "?")
    return text


def is_tag_visible(element: bs4.element) -> bool:
    if element.parent.name in ["style", "script", "head", "title", "meta", "[document]"]:
        return False
    if isinstance(element, bs4.element.Comment):
        return False
    return True


def create_session() -> requests.Session:
    session = requests.Session()
    session.verify = False
    session.headers.update(HEADERS)
    return session


# ---------------------------------------------------------------------------
# Tavily search  (replaces Bing)
# ---------------------------------------------------------------------------
def search_tavily(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Query the Tavily Search API and return a list of result dicts, each with
    keys: 'url', 'content' (snippet), 'title'.

    Falls back to an empty list if the key is missing or the call fails.
    """
    if not TAVILY_API_KEY:
        logger.error("TAVILY_API_KEY not set. Cannot perform search.")
        return []

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "basic",
        "include_answer": False,
        "include_raw_content": False,
        "max_results": max_results,
    }

    try:
        response = requests.post(TAVILY_SEARCH_URL, json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        logger.info(f"Tavily returned {len(results)} results for: {query}")
        return results
    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return []


# ---------------------------------------------------------------------------
# URL scraping (kept for when we want full page text beyond snippets)
# ---------------------------------------------------------------------------
def scrape_url(url: str, timeout: float = 10, max_retries: int = 3) -> Tuple[Optional[str], str]:
    session = create_session()
    for attempt in range(max_retries):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            soup = bs4.BeautifulSoup(response.text, "html.parser")
            for element in soup(["script", "style", "footer", "header"]):
                element.decompose()
            texts = soup.findAll(text=True)
            visible_texts = filter(is_tag_visible, texts)
            text = ' '.join(t.strip() for t in visible_texts if t.strip())
            return clean_text(text), url
        except Exception as e:
            logger.warning(f"Scrape attempt {attempt + 1} failed for {url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None, url


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_text(
    text: str,
    sentences_per_passage: int,
    sliding_distance: Optional[int] = None,
    filter_sentence_len: int = 250,
) -> List[str]:
    if not sliding_distance or sliding_distance > sentences_per_passage:
        sliding_distance = sentences_per_passage
    if not text:
        return []
    try:
        text = clean_text(text)
        doc = TOKENIZER(text[:500000])
        sentences = [
            sent.text.strip()
            for sent in doc.sents
            if sent.text.strip() and len(sent.text.strip()) <= filter_sentence_len
        ]
        passages = []
        for i in range(0, len(sentences), sliding_distance):
            passage = ' '.join(sentences[i: i + sentences_per_passage])
            if passage:
                passages.append(passage)
        return passages
    except Exception as e:
        logger.error(f"Error in chunk_text: {e}")
        simple_sentences = [s.strip() for s in text.split('.') if s.strip()]
        return [
            ' '.join(simple_sentences[i: i + sentences_per_passage])
            for i in range(0, len(simple_sentences), sliding_distance or sentences_per_passage)
        ]


# ---------------------------------------------------------------------------
# Query enhancement
# ---------------------------------------------------------------------------
def generate_search_query(
    question: str,
    context: Optional[str] = None,
    atomic_statement: Optional[str] = None,
) -> str:
    return question.strip()


# ---------------------------------------------------------------------------
# Main search entry point
# ---------------------------------------------------------------------------
def run_search(
    query: str,
    cached_search_results: Optional[List[str]] = None,
    context: Optional[str] = None,
    atomic_statement: Optional[str] = None,
    max_search_results_per_query: int = 3,
    max_sentences_per_passage: int = 5,
    sliding_distance: int = 1,
    max_passages_per_search_result_to_return: int = 1,
    timeout: float = 10,
    filter_sentence_len: int = 250,
    max_passages_per_search_result_to_score: int = 30,
    **kwargs,  # absorb any extra kwargs gracefully
) -> List[Dict[str, Any]]:
    """
    1. Build enhanced query.
    2. Call Tavily (or use cached URLs).
    3. Use Tavily snippets directly (fast) and optionally scrape for more text.
    4. Rank passages with CrossEncoder.
    5. Return top passages.
    """
    enhanced_query = generate_search_query(query, context, atomic_statement)
    logger.info(f"Enhanced query: {enhanced_query}")

    retrieved_passages: List[Dict[str, Any]] = []
    seen_passages: set = set()

    # ---- Use Tavily results ------------------------------------------------
    if cached_search_results is not None:
        # cached_search_results is a list of URL strings (legacy support)
        tavily_results = [{"url": u, "content": "", "title": ""} for u in cached_search_results]
    else:
        tavily_results = search_azure(enhanced_query, max_results=max_search_results_per_query)

    if not tavily_results:
        logger.warning("No search results found.")
        return []

    for result in tavily_results[:max_search_results_per_query]:
        url = result.get("url", "")
        snippet = result.get("content", "").strip()

        # Prefer Tavily snippet; fall back to scraping if snippet is empty
        if snippet:
            passages = chunk_text(
                snippet,
                sentences_per_passage=max_sentences_per_passage,
                sliding_distance=sliding_distance,
                filter_sentence_len=filter_sentence_len,
            )
            # If snippet is short, just use it as one passage
            if not passages:
                passages = [snippet]
        else:
            # Scrape the URL for full text
            webtext, _ = scrape_url(url, timeout=timeout)
            if not webtext:
                continue
            passages = chunk_text(
                webtext,
                sentences_per_passage=max_sentences_per_passage,
                sliding_distance=sliding_distance,
                filter_sentence_len=filter_sentence_len,
            )

        if not passages:
            continue

        passages = passages[:max_passages_per_search_result_to_score]

        with torch.no_grad():
            pairs = [(enhanced_query, p) for p in passages]
            scores = PASSAGE_RANKER.predict(pairs).tolist()

        passage_scores = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)

        for passage, score in passage_scores[:max_passages_per_search_result_to_return]:
            if passage not in seen_passages:
                seen_passages.add(passage)
                retrieved_passages.append({
                    "text": passage,
                    "url": url,
                    "query": query,
                    "enhanced_query": enhanced_query,
                    "context_used": bool(context),
                    "atomic_statement": atomic_statement,
                    "retrieval_score": score,
                })

    # Softmax over raw retrieval scores for interpretability
    if retrieved_passages:
        scores_tensor = torch.tensor([p["retrieval_score"] for p in retrieved_passages])
        probs = torch.nn.functional.softmax(scores_tensor, dim=0).tolist()
        for passage, prob in zip(retrieved_passages, probs):
            passage["score"] = prob

    return retrieved_passages


def search_for_atomic_statement(
    statement: str,
    questions: List[str],
    context: Optional[str] = None,
    **search_kwargs,
) -> List[Dict[str, Any]]:
    all_passages: List[Dict[str, Any]] = []
    seen_texts: set = set()
    for i, question in enumerate(questions, start=1):
        logger.info(f"Processing question {i}/{len(questions)}: {question}")
        passages = run_search(
            query=question,
            context=context,
            atomic_statement=statement,
            **search_kwargs,
        )
        for passage in passages:
            if passage["text"] not in seen_texts:
                seen_texts.add(passage["text"])
                all_passages.append(passage)
    all_passages.sort(key=lambda x: x.get("score", 0), reverse=True)
    return all_passages


if __name__ == "__main__":
    test_statement = "Lena Headey portrayed Cersei Lannister in Game of Thrones."
    test_questions = [
        "When did Lena Headey start playing Cersei Lannister?",
        "What awards did she receive for this role?",
    ]
    results = search_for_atomic_statement(
        statement=test_statement,
        questions=test_questions,
        max_search_results_per_query=2,
    )
    for r in results:
        print(f"\nQuery: {r['query']}")
        print(f"Score: {r.get('score', 0):.3f}")
        print(f"Text: {r['text'][:200]}...")



# """Search and evidence collection utilities with improved text processing."""
# import itertools
# import os
# import time
# import random
# import logging
# import warnings
# from typing import Any, Dict, List, Tuple, Optional
 
# import concurrent.futures
# import bs4
# import requests
# import spacy
# import torch
# from spacy.cli import download
# from sentence_transformers import CrossEncoder
# from urllib3.exceptions import InsecureRequestWarning
 
# warnings.filterwarnings("ignore", category=InsecureRequestWarning)
 
# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)
 
# # Initialize spaCy
# try:
#     TOKENIZER = spacy.load("en_core_web_sm")
# except OSError:
#     logger.info("Downloading spaCy model...")
#     download("en_core_web_sm")
#     TOKENIZER = spacy.load("en_core_web_sm")
 
# if "sentencizer" not in TOKENIZER.pipe_names:
#     TOKENIZER.add_pipe("sentencizer")
 
# # Initialize cross-encoder for passage ranking
# PASSAGE_RANKER = CrossEncoder(
#     "cross-encoder/ms-marco-MiniLM-L-6-v2",
#     max_length=512,
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# )
 
# # Bing Search configuration
# SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search/"
# SUBSCRIPTION_KEY = os.getenv("AZURE_SEARCH_KEY")
# HEADERS = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
# }
 
# def clean_text(text: str) -> str:
#     """
#     Clean and normalize text by:
#       - Collapsing extra whitespace
#       - Removing certain punctuation anomalies
#     """
#     if not text:
#         return ""
#     text = ' '.join(text.split())
#     text = text.replace("..", ".").replace("!.", "!").replace("?.", "?")
#     return text
 
# def is_tag_visible(element: bs4.element) -> bool:
#     """
#     Check if an HTML element is typically visible (not a script, style, comment, etc.).
#     Used when scraping web content.
#     """
#     if element.parent.name in ["style", "script", "head", "title", "meta", "[document]"]:
#         return False
#     if isinstance(element, bs4.element.Comment):
#         return False
#     return True
 
# def create_session() -> requests.Session:
#     """
#     Create a requests session with default headers and SSL verification disabled.
#     """
#     session = requests.Session()
#     session.verify = False
#     session.headers.update(HEADERS)
#     return session
 
# def scrape_url(url: str, timeout: float = 10, max_retries: int = 3) -> Tuple[Optional[str], str]:
#     """
#     Scrape the plain text from a given URL with retry logic.
#     Returns a tuple of (scraped_text, original_url).
#     """
#     session = create_session()
    
#     for attempt in range(max_retries):
#         try:
#             response = session.get(url, timeout=timeout)
#             response.raise_for_status()
            
#             soup = bs4.BeautifulSoup(response.text, "html.parser")
#             # Remove irrelevant elements
#             for element in soup(["script", "style", "footer", "header"]):
#                 element.decompose()
            
#             texts = soup.findAll(text=True)
#             visible_texts = filter(is_tag_visible, texts)
#             text = ' '.join(t.strip() for t in visible_texts if t.strip())
            
#             return clean_text(text), url
                
#         except Exception as e:
#             wait_time = 2 ** attempt
#             logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
#             if attempt < max_retries - 1:
#                 time.sleep(wait_time)
    
#     # If all retries fail
#     return None, url
 
# def fallback_search(query: str) -> List[str]:
#     """
#     Attempt a DuckDuckGo search if Bing is unavailable or fails.
#     Returns up to 10 result URLs.
#     """
#     try:
#         from duckduckgo_search import ddg
#         results = ddg(query, max_results=10)
#         return [r['link'] for r in results] if results else []
#     except Exception as e:
#         logger.error(f"Fallback search failed: {e}")
#         return []
 
# def search_bing(query: str, timeout: float = 5, max_retries: int = 3) -> List[str]:
#     """
#     Performs a Bing web search using the AZURE_SEARCH_KEY environment variable.
#     Falls back to DuckDuckGo if no results or Bing fails.
#     """
#     urls = []
    
#     if SUBSCRIPTION_KEY:
#         headers = {"Ocp-Apim-Subscription-Key": SUBSCRIPTION_KEY}
#         params = {
#             "q": query,
#             "textDecorations": True,
#             "textFormat": "HTML",
#             "count": 10
#         }
        
#         session = create_session()
#         for attempt in range(max_retries):
#             try:
#                 response = session.get(
#                     SEARCH_URL,
#                     headers=headers,
#                     params=params,
#                     timeout=timeout
#                 )
#                 response.raise_for_status()
#                 results = response.json()
#                 urls = [r["url"] for r in results.get("webPages", {}).get("value", [])]
#                 break
#             except Exception as e:
#                 logger.warning(f"Bing search failed (attempt {attempt+1}): {e}")
#                 time.sleep(2 ** attempt)
    
#     if not urls:
#         # If Bing fails or returns empty, fallback
#         urls = fallback_search(query)
        
#     # Exclude certain URLs (pdf, major social media, etc.)
#     filtered_urls = [
#         url for url in urls
#         if not any(
#             domain in url.lower()
#             for domain in [".pdf", "facebook.com", "twitter.com", "instagram.com"]
#         )
#     ]
#     return filtered_urls
 
# def chunk_text(
#     text: str,
#     sentences_per_passage: int,
#     sliding_distance: Optional[int] = None,
#     filter_sentence_len: int = 250
# ) -> List[str]:
#     """
#     Breaks text into passages using spaCy's sentence detection.
#     - `sentences_per_passage`: how many sentences to group in one chunk
#     - `sliding_distance`: overlap between chunks (defaults to no overlap if None)
#     - `filter_sentence_len`: skip sentences over this length
#     """
#     if not sliding_distance or sliding_distance > sentences_per_passage:
#         sliding_distance = sentences_per_passage
    
#     if not text:
#         return []
    
#     try:
#         text = clean_text(text)
#         doc = TOKENIZER(text[:500000])  # limit to first 500k chars
 
#         # Collect viable sentences
#         sentences = []
#         for sent in doc.sents:
#             sent_text = sent.text.strip()
#             if sent_text and len(sent_text) <= filter_sentence_len:
#                 sentences.append(sent_text)
        
#         # Create passages with sliding window
#         passages = []
#         for i in range(0, len(sentences), sliding_distance):
#             passage = ' '.join(sentences[i:i + sentences_per_passage])
#             if passage:
#                 passages.append(passage)
        
#         return passages
    
#     except Exception as e:
#         logger.error(f"Error in chunk_text: {e}")
#         # Fallback: naive split on '.'
#         simple_sentences = [s.strip() for s in text.split('.') if s.strip()]
#         if simple_sentences:
#             return [
#                 ' '.join(simple_sentences[i:i + sentences_per_passage])
#                 for i in range(0, len(simple_sentences), sliding_distance)
#             ]
#         return []
 
# def generate_search_query(
#     question: str,
#     context: Optional[str] = None,
#     atomic_statement: Optional[str] = None
# ) -> str:
#     """
#     Builds a query string by incorporating terms from:
#       - the question itself
#       - the atomic statement
#       - the prior context
#     Only includes a small subset of keywords to avoid bloat.
#     """
#     query_parts = [question.strip()]
    
#     if atomic_statement:
#         doc = TOKENIZER(atomic_statement)
#         statement_terms = [
#             token.text.lower() for token in doc
#             if not token.is_stop and len(token.text) > 3
#         ]
#         if statement_terms:
#             query_parts.extend(statement_terms[:2])
    
#     if context:
#         doc = TOKENIZER(context)
#         context_terms = [
#             token.text.lower() for token in doc
#             if not token.is_stop and len(token.text) > 3
#         ]
#         if context_terms:
#             query_parts.extend(context_terms[:2])
    
#     # Remove duplicates while preserving order
#     seen = set()
#     unique_parts = []
#     for part in query_parts:
#         if part not in seen:
#             seen.add(part)
#             unique_parts.append(part)
    
#     # Join into a single string
#     return ' '.join(unique_parts)
 
# def run_search(
#     query: str,
#     cached_search_results: Optional[List[str]] = None,
#     context: Optional[str] = None,
#     atomic_statement: Optional[str] = None,
#     max_search_results_per_query: int = 3,
#     max_sentences_per_passage: int = 5,
#     sliding_distance: int = 1,
#     max_passages_per_search_result_to_return: int = 1,
#     timeout: float = 10,
#     randomize_num_sentences: bool = False,
#     filter_sentence_len: int = 250,
#     max_passages_per_search_result_to_score: int = 30,
# ) -> List[Dict[str, Any]]:
#     """
#     Primary method to:
#       1. Generate an enhanced query from question/context/statement
#       2. Perform web search (bing or fallback)
#       3. Scrape and chunk text from top URLs
#       4. Rank passages via cross-encoder
#       5. Return top N passages per result
#     """
#     enhanced_query = generate_search_query(query, context, atomic_statement)
#     logger.info(f"Enhanced query: {enhanced_query}")
    
#     # Use cached search results if provided
#     search_results = cached_search_results if cached_search_results is not None else search_bing(enhanced_query, timeout)
    
#     if not search_results:
#         logger.warning("No search results found.")
#         return []
    
#     # Scrape URLs concurrently
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         scraped_results = list(executor.map(
#             lambda url: scrape_url(url, timeout),
#             search_results
#         ))
    
#     valid_results = [(text, url) for text, url in scraped_results if text]
    
#     if not valid_results:
#         logger.warning("No valid content found in search results.")
#         return []
    
#     retrieved_passages = []
#     seen_passages = set()
    
#     # Process each URL's text to generate passages
#     for webtext, url in valid_results[:max_search_results_per_query]:
#         # Randomly adjust sentences_per_passage if desired
#         if randomize_num_sentences:
#             sents_per_passage = random.randint(1, max_sentences_per_passage)
#         else:
#             sents_per_passage = max_sentences_per_passage
        
#         passages = chunk_text(
#             text=webtext,
#             sentences_per_passage=sents_per_passage,
#             sliding_distance=sliding_distance,
#             filter_sentence_len=filter_sentence_len
#         )
        
#         if not passages:
#             continue
        
#         # Limit how many passages we rank
#         passages = passages[:max_passages_per_search_result_to_score]
        
#         with torch.no_grad():
#             pairs = [(enhanced_query, p) for p in passages]
#             scores = PASSAGE_RANKER.predict(pairs).tolist()
        
#         # Sort passages by descending score
#         passage_scores = sorted(
#             zip(passages, scores),
#             key=lambda x: x[1],
#             reverse=True
#         )
        
#         # Take only top N (max_passages_per_search_result_to_return) from each URL
#         for passage, score in passage_scores[:max_passages_per_search_result_to_return]:
#             if passage not in seen_passages:
#                 seen_passages.add(passage)
#                 retrieved_passages.append({
#                     "text": passage,
#                     "url": url,
#                     "query": query,
#                     "enhanced_query": enhanced_query,
#                     "context_used": bool(context),
#                     "atomic_statement": atomic_statement,
#                     "retrieval_score": score
#                 })
    
#     # Convert raw scores to softmax probabilities for interpretability
#     if retrieved_passages:
#         scores_tensor = torch.tensor([p["retrieval_score"] for p in retrieved_passages])
#         probs = torch.nn.functional.softmax(scores_tensor, dim=0).tolist()
#         for passage, prob in zip(retrieved_passages, probs):
#             passage["score"] = prob
    
#     return retrieved_passages
 
# def search_for_atomic_statement(
#     statement: str,
#     questions: List[str],
#     context: Optional[str] = None,
#     **search_kwargs
# ) -> List[Dict[str, Any]]:
#     """
#     Aggregates evidence for an atomic statement by iterating over each question,
#     running `run_search`, and merging the results.
#     Sorts final passages by highest aggregated score.
#     """
#     all_passages = []
#     seen_texts = set()
    
#     for i, question in enumerate(questions, start=1):
#         logger.info(f"Processing question {i}/{len(questions)}: {question}")
        
#         # Run search for each question
#         passages = run_search(
#             query=question,
#             context=context,
#             atomic_statement=statement,
#             **search_kwargs
#         )
        
#         # Deduplicate passages by text content
#         for passage in passages:
#             if passage["text"] not in seen_texts:
#                 seen_texts.add(passage["text"])
#                 all_passages.append(passage)
    
#     # Sort all retrieved passages by descending score
#     all_passages.sort(key=lambda x: x.get("score", 0), reverse=True)
#     return all_passages
 
# if __name__ == "__main__":
#     # Example usage
#     test_statement = "Lena Headey portrayed Cersei Lannister in Game of Thrones."
#     test_questions = [
#         "When did Lena Headey start playing Cersei Lannister?",
#         "What awards did she receive for this role?",
#         "How long did she play the character?"
#     ]
    
#     results = search_for_atomic_statement(
#         statement=test_statement,
#         questions=test_questions,
#         max_search_results_per_query=2
#     )
    
#     print("\nSearch Results:")
#     for r in results:
#         print(f"\nQuery: {r['query']}")
#         print(f"Enhanced query: {r['enhanced_query']}")
#         print(f"Score: {r['score']:.3f}")
#         print(f"Text: {r['text'][:200]}...")
 






# """Utils for searching a query and returning top passages from search results."""
# import concurrent.futures
# import itertools
# import os
# import random
# import logging
# from typing import Any, Dict, List, Tuple

# import bs4
# import requests
# import spacy
# import torch
# from sentence_transformers import CrossEncoder

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Load the passage ranker model
# try:
#     PASSAGE_RANKER = CrossEncoder(
#         "cross-encoder/ms-marco-MiniLM-L-6-v2",
#         max_length=512,
#         device="cpu",  # Default to CPU
#     )
# except Exception as e:
#     logging.error(f"Failed to load CrossEncoder model: {e}")
#     PASSAGE_RANKER = None

# # Bing search API setup
# SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search/"
# SUBSCRIPTION_KEY = os.getenv("AZURE_SEARCH_KEY")
# if not SUBSCRIPTION_KEY:
#     logging.error("AZURE_SEARCH_KEY environment variable is not set.")

# # Load spaCy tokenizer
# try:
#     TOKENIZER = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
# except Exception as e:
#     logging.error(f"Failed to load spaCy model: {e}")
#     TOKENIZER = None

# def chunk_text(
#     text: str,
#     sentences_per_passage: int,
#     filter_sentence_len: int,
#     sliding_distance: int = None,
# ) -> List[str]:
#     """Chunks text into passages using a sliding window approach."""
#     if not sliding_distance or sliding_distance > sentences_per_passage:
#         sliding_distance = sentences_per_passage

#     passages = []
#     try:
#         # Process up to 500,000 characters to avoid overwhelming the tokenizer.
#         doc = TOKENIZER(text[:500000])
#         sentences = [s.text for s in doc.sents if len(s.text) <= filter_sentence_len]

#         for idx in range(0, len(sentences), sliding_distance):
#             passage = " ".join(sentences[idx: idx + sentences_per_passage])
#             passages.append(passage)
#     except Exception as e:
#         logging.error(f"Error in chunk_text: {e}")

#     return passages

# def is_tag_visible(element: bs4.element) -> bool:
#     """Checks if an HTML element is visible on the page."""
#     if element.parent.name in ["style", "script", "head", "title", "meta", "[document]"] or isinstance(element, bs4.element.Comment):
#         return False
#     return True

# def scrape_url(url: str, timeout: float = 3) -> Tuple[str, str]:
#     """Scrapes a URL for text content."""
#     try:
#         response = requests.get(url, timeout=timeout)
#         response.raise_for_status()

#         soup = bs4.BeautifulSoup(response.text, "html.parser")
#         texts = filter(is_tag_visible, soup.findAll(text=True))

#         web_text = " ".join(t.strip() for t in texts).strip()
#         web_text = " ".join(web_text.split())
#         return web_text, url
#     except requests.RequestException as e:
#         logging.error(f"Error scraping URL {url}: {e}")
#         return None, url

# def search_bing(query: str, timeout: float = 3) -> List[str]:
#     """Uses Bing's Web Search API to retrieve URLs for a query."""
#     if not SUBSCRIPTION_KEY:
#         logging.error("AZURE_SEARCH_KEY is not set. Cannot perform Bing search.")
#         return []

#     headers = {"Ocp-Apim-Subscription-Key": SUBSCRIPTION_KEY}
#     params = {"q": query, "textDecorations": True, "textFormat": "HTML"}
#     try:
#         response = requests.get(SEARCH_URL, headers=headers, params=params, timeout=timeout)
#         response.raise_for_status()
#         search_results = [result["url"] for result in response.json().get("webPages", {}).get("value", [])]
#         return search_results
#     except requests.RequestException as e:
#         logging.error(f"Error in Bing search: {e}")
#         return []

# def run_search(
#     query: str,
#     cached_search_results: List[str] = None,
#     max_search_results_per_query: int = 7,  
#     max_sentences_per_passage: int = 6,
#     sliding_distance: int = 1,
#     max_passages_per_search_result_to_return: int = 5, 
#     timeout: float = 2,
#     randomize_num_sentences: bool = False,
#     filter_sentence_len: int = 250,
#     max_passages_per_search_result_to_score: int = 30,
# ) -> List[Dict[str, Any]]:
#     """Performs search and retrieves relevant passages."""
#     if not PASSAGE_RANKER or not TOKENIZER:
#         logging.error("PASSAGE_RANKER or TOKENIZER not initialized. Cannot perform search.")
#         return []

#     search_results = cached_search_results or search_bing(query, timeout=timeout)

#     if not search_results:
#         logging.warning("No search results found.")
#         return []

#     # Scrape URLs concurrently
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         scraped_results = list(executor.map(scrape_url, search_results[:max_search_results_per_query], itertools.repeat(timeout)))
#     scraped_results = [result for result in scraped_results if result[0] and ".pdf" not in result[1]]

#     retrieved_passages = []
#     for web_text, url in scraped_results:
#         sents_per_passage = random.randint(1, max_sentences_per_passage) if randomize_num_sentences else max_sentences_per_passage
#         passages = chunk_text(web_text, sents_per_passage, filter_sentence_len, sliding_distance)[:max_passages_per_search_result_to_score]

#         if not passages:
#             continue

#         try:
#             # Rank passages based on relevance to the query
#             scores = PASSAGE_RANKER.predict([(query, p) for p in passages]).tolist()
#             passage_scores = list(zip(passages, scores))

#             # Sort and select the top passages
#             passage_scores.sort(key=lambda x: x[1], reverse=True)
#             for passage, score in passage_scores[:max_passages_per_search_result_to_return]:
#                 retrieved_passages.append({
#                     "text": passage,
#                     "url": url,
#                     "query": query,
#                     "sents_per_passage": sents_per_passage,
#                     "retrieval_score": score,
#                 })
#         except Exception as e:
#             logging.error(f"Error scoring passages: {e}")

#     if retrieved_passages:
#         # Normalize scores with softmax
#         scores = [r["retrieval_score"] for r in retrieved_passages]
#         probs = torch.nn.functional.softmax(torch.Tensor(scores), dim=-1).tolist()
#         for prob, passage in zip(probs, retrieved_passages):
#             passage["score"] = prob  # Assign normalized score as "score"

#     logging.info(f"Retrieved {len(retrieved_passages)} passages for query: {query}")
#     return retrieved_passages

# if __name__ == "__main__":
#     query = "What is the capital of France?"
#     results = run_search(query)
#     for i, result in enumerate(results, 1):
#         print(f"Result {i}:")
#         print(f"Text: {result['text'][:100]}...")
#         print(f"URL: {result['url']}")
#         print(f"Score: {result['score']}")
#         print()




# """Utils for searching a query and returning top passages from search results."""
# import concurrent.futures
# import itertools
# import os
# import random
# import logging
# from typing import Any, Dict, List, Tuple

# import bs4
# import requests
# import spacy
# import torch
# from sentence_transformers import CrossEncoder

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# try:
#     PASSAGE_RANKER = CrossEncoder(
#         "cross-encoder/ms-marco-MiniLM-L-6-v2",
#         max_length=512,
#         device="cpu",
#     )
# except Exception as e:
#     logging.error(f"Failed to load CrossEncoder: {e}")
#     PASSAGE_RANKER = None

# SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search/"
# SUBSCRIPTION_KEY = os.getenv("AZURE_SEARCH_KEY")
# if not SUBSCRIPTION_KEY:
#     logging.error("AZURE_SEARCH_KEY environment variable is not set")

# try:
#     TOKENIZER = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
# except Exception as e:
#     logging.error(f"Failed to load spaCy model: {e}")
#     TOKENIZER = None

# def chunk_text(
#     text: str,
#     sentences_per_passage: int,
#     filter_sentence_len: int,
#     sliding_distance: int = None,
# ) -> List[str]:
#     """Chunks text into passages using a sliding window."""
#     if not sliding_distance or sliding_distance > sentences_per_passage:
#         sliding_distance = sentences_per_passage
#     assert sentences_per_passage > 0 and sliding_distance > 0

#     passages = []
#     try:
#         doc = TOKENIZER(text[:500000])  # Take 500k chars to not break tokenization.
#         sents = [
#             s.text
#             for s in doc.sents
#             if len(s.text) <= filter_sentence_len  # Long sents are usually metadata.
#         ]
#         for idx in range(0, len(sents), sliding_distance):
#             passages.append(" ".join(sents[idx : idx + sentences_per_passage]))
#     except UnicodeEncodeError as e:
#         logging.error(f"Unicode error when using Spacy: {e}")
#     except Exception as e:
#         logging.error(f"Error in chunk_text: {e}")

#     return passages

# def is_tag_visible(element: bs4.element) -> bool:
#     """Determines if an HTML element is visible."""
#     if element.parent.name in [
#         "style", "script", "head", "title", "meta", "[document]"
#     ] or isinstance(element, bs4.element.Comment):
#         return False
#     return True

# def scrape_url(url: str, timeout: float = 3) -> Tuple[str, str]:
#     """Scrapes a URL for all text information."""
#     try:
#         response = requests.get(url, timeout=timeout)
#         response.raise_for_status()

#         soup = bs4.BeautifulSoup(response.text, "html.parser")
#         texts = soup.findAll(text=True)
#         visible_text = filter(is_tag_visible, texts)

#         web_text = " ".join(t.strip() for t in visible_text).strip()
#         web_text = " ".join(web_text.split())
#         return web_text, url
#     except requests.exceptions.RequestException as e:
#         logging.error(f"Error scraping URL {url}: {e}")
#         return None, url
#     except Exception as e:
#         logging.error(f"Unexpected error scraping URL {url}: {e}")
#         return None, url

# def search_bing(query: str, timeout: float = 3) -> List[str]:
#     """Searches the query using Bing."""
#     if not SUBSCRIPTION_KEY:
#         logging.error("AZURE_SEARCH_KEY is not set. Cannot perform Bing search.")
#         return []

#     headers = {"Ocp-Apim-Subscription-Key": SUBSCRIPTION_KEY}
#     params = {"q": query, "textDecorations": True, "textFormat": "HTML"}
#     try:
#         response = requests.get(SEARCH_URL, headers=headers, params=params, timeout=timeout)
#         response.raise_for_status()
#         response_json = response.json()
#         search_results = [r["url"] for r in response_json["webPages"]["value"]]
#         return search_results
#     except requests.exceptions.RequestException as e:
#         logging.error(f"Error in Bing search: {e}")
#         return []
#     except KeyError as e:
#         logging.error(f"Unexpected response format from Bing API: {e}")
#         return []

# def run_search(
#     query: str,
#     cached_search_results: List[str] = None,
#     max_search_results_per_query: int = 3,
#     max_sentences_per_passage: int = 5,
#     sliding_distance: int = 1,
#     max_passages_per_search_result_to_return: int = 1,
#     timeout: float = 3,
#     randomize_num_sentences: bool = False,
#     filter_sentence_len: int = 250,
#     max_passages_per_search_result_to_score: int = 30,
# ) -> List[Dict[str, Any]]:
#     """Searches the query on a search engine and returns the most relevant information."""
#     if not PASSAGE_RANKER or not TOKENIZER:
#         logging.error("PASSAGE_RANKER or TOKENIZER not initialized. Cannot perform search.")
#         return []

#     if cached_search_results is not None:
#         search_results = cached_search_results
#     else:
#         search_results = search_bing(query, timeout=timeout)

#     if not search_results:
#         logging.warning("No search results found.")
#         return []

#     with concurrent.futures.ThreadPoolExecutor() as e:
#         scraped_results = list(e.map(scrape_url, search_results, itertools.repeat(timeout)))
#     scraped_results = [r for r in scraped_results if r[0] and ".pdf" not in r[1]]

#     retrieved_passages = []
#     for webtext, url in scraped_results[:max_search_results_per_query]:
#         sents_per_passage = random.randint(1, max_sentences_per_passage) if randomize_num_sentences else max_sentences_per_passage

#         passages = chunk_text(
#             text=webtext,
#             sentences_per_passage=sents_per_passage,
#             filter_sentence_len=filter_sentence_len,
#             sliding_distance=sliding_distance,
#         )
#         passages = passages[:max_passages_per_search_result_to_score]
#         if not passages:
#             continue

#         try:
#             scores = PASSAGE_RANKER.predict([(query, p) for p in passages]).tolist()
#             passage_scores = list(zip(passages, scores))

#             passage_scores.sort(key=lambda x: x[1], reverse=True)
#             for passage, score in passage_scores[:max_passages_per_search_result_to_return]:
#                 retrieved_passages.append({
#                     "text": passage,
#                     "url": url,
#                     "query": query,
#                     "sents_per_passage": sents_per_passage,
#                     "retrieval_score": score,
#                 })
#         except Exception as e:
#             logging.error(f"Error scoring passages: {e}")

#     if retrieved_passages:
#         retrieved_passages.sort(key=lambda d: d["retrieval_score"], reverse=True)

#         scores = [r["retrieval_score"] for r in retrieved_passages]
#         probs = torch.nn.functional.softmax(torch.Tensor(scores), dim=-1).tolist()
#         for prob, passage in zip(probs, retrieved_passages):
#             passage["score"] = prob

#     logging.info(f"Retrieved {len(retrieved_passages)} passages for query: {query}")
#     return retrieved_passages

# if __name__ == "__main__":
#     # Example usage
#     query = "What is the capital of France?"
#     results = run_search(query)
#     for i, result in enumerate(results, 1):
#         print(f"Result {i}:")
#         print(f"Text: {result['text'][:100]}...")
#         print(f"URL: {result['url']}")
#         print(f"Score: {result['score']}")
#         print()





# """Utils for searching a query and returning top passages from search results."""
# import concurrent.futures
# import itertools
# import os
# import random
# from typing import Any, Dict, List, Tuple

# import bs4
# import requests
# import spacy
# import torch
# from sentence_transformers import CrossEncoder

# PASSAGE_RANKER = CrossEncoder(
#     "cross-encoder/ms-marco-MiniLM-L-6-v2",
#     max_length=512,
#     device="cpu",
# )
# SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search/"
# SUBSCRIPTION_KEY = os.getenv("AZURE_SEARCH_KEY")
# TOKENIZER = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])


# def chunk_text(
#     text: str,
#     sentences_per_passage: int,
#     filter_sentence_len: int,
#     sliding_distance: int = None,
# ) -> List[str]:
#     """Chunks text into passages using a sliding window.

#     Args:
#         text: Text to chunk into passages.
#         sentences_per_passage: Number of sentences for each passage.
#         filter_sentence_len: Maximum number of chars of each sentence before being filtered.
#         sliding_distance: Sliding distance over the text. Allows the passages to have
#             overlap. The sliding distance cannot be greater than the window size.
#     Returns:
#         passages: Chunked passages from the text.
#     """
#     if not sliding_distance or sliding_distance > sentences_per_passage:
#         sliding_distance = sentences_per_passage
#     assert sentences_per_passage > 0 and sliding_distance > 0

#     passages = []
#     try:
#         doc = TOKENIZER(text[:500000])  # Take 500k chars to not break tokenization.
#         sents = [
#             s.text
#             for s in doc.sents
#             if len(s.text) <= filter_sentence_len  # Long sents are usually metadata.
#         ]
#         for idx in range(0, len(sents), sliding_distance):
#             passages.append(" ".join(sents[idx : idx + sentences_per_passage]))
#     except UnicodeEncodeError as _:  # Sometimes run into Unicode error when tokenizing.
#         print("Unicode error when using Spacy. Skipping text.")

#     return passages


# def is_tag_visible(element: bs4.element) -> bool:
#     """Determines if an HTML element is visible.

#     Args:
#         element: A BeautifulSoup element to check the visiblity of.
#     returns:
#         Whether the element is visible.
#     """
#     if element.parent.name in [
#         "style",
#         "script",
#         "head",
#         "title",
#         "meta",
#         "[document]",
#     ] or isinstance(element, bs4.element.Comment):
#         return False
#     return True


# def scrape_url(url: str, timeout: float = 3) -> Tuple[str, str]:
#     """Scrapes a URL for all text information.

#     Args:
#         url: URL of webpage to scrape.
#         timeout: Timeout of the requests call.
#     Returns:
#         web_text: The visible text of the scraped URL.
#         url: URL input.
#     """
#     # Scrape the URL
#     try:
#         response = requests.get(url, timeout=timeout)
#         response.raise_for_status()
#     except requests.exceptions.RequestException as _:
#         return None, url

#     # Extract out all text from the tags
#     try:
#         soup = bs4.BeautifulSoup(response.text, "html.parser")
#         texts = soup.findAll(text=True)
#         # Filter out invisible text from the page.
#         visible_text = filter(is_tag_visible, texts)
#     except Exception as _:
#         return None, url

#     # Returns all the text concatenated as a string.
#     web_text = " ".join(t.strip() for t in visible_text).strip()
#     # Clean up spacing.
#     web_text = " ".join(web_text.split())
#     return web_text, url


# def search_bing(query: str, timeout: float = 3) -> List[str]:
#     """Searches the query using Bing.
#     Args:
#         query: Search query.
#         timeout: Timeout of the requests call.
#     Returns:
#         search_results: A list of the top URLs relevant to the query.
#     """
#     headers = {"Ocp-Apim-Subscription-Key": SUBSCRIPTION_KEY}
#     params = {"q": query, "textDecorations": True, "textFormat": "HTML"}
#     response = requests.get(SEARCH_URL, headers=headers, params=params, timeout=timeout)
#     response.raise_for_status()

#     response = response.json()
#     search_results = [r["url"] for r in response["webPages"]["value"]]
#     return search_results


# def run_search(
#     query: str,
#     cached_search_results: List[str] = None,
#     max_search_results_per_query: int = 3,
#     max_sentences_per_passage: int = 5,
#     sliding_distance: int = 1,
#     max_passages_per_search_result_to_return: int = 1,
#     timeout: float = 3,
#     randomize_num_sentences: bool = False,
#     filter_sentence_len: int = 250,
#     max_passages_per_search_result_to_score: int = 30,
# ) -> List[Dict[str, Any]]:
#     """Searches the query on a search engine and returns the most relevant information.

#     Args:
#         query: Search query.
#         max_search_results_per_query: Maximum number of search results to get return.
#         max_sentences_per_passage: Maximum number of sentences for each passage.
#         filter_sentence_len: Maximum length of a sentence before being filtered.
#         sliding_distance: Sliding distance over the sentences of each search result.
#             Used to extract passages.
#         max_passages_per_search_result_to_score: Maxinum number of passages to score for
#             each search result.
#         max_passages_per_search_result_to_return: Maximum number of passages to return
#             for each search result.
#     Returns:
#         retrieved_passages: Top retrieved passages for the search query.
#     """
#     if cached_search_results is not None:
#         search_results = cached_search_results
#     else:
#         search_results = search_bing(query, timeout=timeout)

#     # Scrape search results in parallel
#     with concurrent.futures.ThreadPoolExecutor() as e:
#         scraped_results = e.map(scrape_url, search_results, itertools.repeat(timeout))
#     # Remove URLs if we weren't able to scrape anything or if they are a PDF.
#     scraped_results = [r for r in scraped_results if r[0] and ".pdf" not in r[1]]

#     # Iterate through the scraped results and extract out the most useful passages.
#     retrieved_passages = []
#     for webtext, url in scraped_results[:max_search_results_per_query]:
#         if randomize_num_sentences:
#             sents_per_passage = random.randint(1, max_sentences_per_passage)
#         else:
#             sents_per_passage = max_sentences_per_passage

#         # Chunk the extracted text into passages.
#         passages = chunk_text(
#             text=webtext,
#             sentences_per_passage=sents_per_passage,
#             filter_sentence_len=filter_sentence_len,
#             sliding_distance=sliding_distance,
#         )
#         passages = passages[:max_passages_per_search_result_to_score]
#         if not passages:
#             continue

#         # Score the passages by relevance to the query using a cross-encoder.
#         scores = PASSAGE_RANKER.predict([(query, p) for p in passages]).tolist()
#         passage_scores = list(zip(passages, scores))

#         # Take the top passages_per_search passages for the current search result.
#         passage_scores.sort(key=lambda x: x[1], reverse=True)
#         for passage, score in passage_scores[:max_passages_per_search_result_to_return]:
#             retrieved_passages.append(
#                 {
#                     "text": passage,
#                     "url": url,
#                     "query": query,
#                     "sents_per_passage": sents_per_passage,
#                     "retrieval_score": score,  # Cross-encoder score as retr score
#                 }
#             )

#     if retrieved_passages:
#         # Sort all retrieved passages by the retrieval score.
#         retrieved_passages = sorted(
#             retrieved_passages, key=lambda d: d["retrieval_score"], reverse=True
#         )

#         # Normalize the retreival scores into probabilities
#         scores = [r["retrieval_score"] for r in retrieved_passages]
#         probs = torch.nn.functional.softmax(torch.Tensor(scores), dim=-1).tolist()
#         for prob, passage in zip(probs, retrieved_passages):
#             passage["score"] = prob

#     return retrieved_passages

def search_azure(query, max_results=5):
    from azure_search_retrieval import search as _azsearch
    rows = _azsearch(query, k=max_results)
    out = [{"url": r.get("url",""), "content": r.get("content",""), "title": r.get("title","")} for r in rows]
    logger.info(f"Azure Search returned {len(out)} results for: {query}")
    return out
