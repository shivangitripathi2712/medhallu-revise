import os, json, hashlib, sys
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, SearchFieldDataType,
)

ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
KEY = os.environ["AZURE_SEARCH_ADMIN_KEY"]
INDEX_NAME = "medhallu-knowledge"
cred = AzureKeyCredential(KEY)


def create_index():
    c = SearchIndexClient(ENDPOINT, cred)
    c.create_or_update_index(SearchIndex(name=INDEX_NAME, fields=[
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="url", type=SearchFieldDataType.String),
    ]))
    print(f"Index '{INDEX_NAME}' ready.")


def upload_corpus_from_jsonl(path):
    seen, docs = set(), []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            kn = r.get("knowledge", "")
            content = (" ".join(map(str, kn)) if isinstance(kn, list) else str(kn)).strip()
            if not content or content in seen:
                continue
            seen.add(content)
            docs.append({"id": hashlib.md5(content.encode()).hexdigest(),
                         "title": str(r.get("question_text", "")), "content": content, "url": ""})
    if not docs:
        print(f"WARNING: no 'knowledge' found in {path}"); return
    sc = SearchClient(ENDPOINT, INDEX_NAME, cred)
    for i in range(0, len(docs), 1000):
        sc.upload_documents(docs[i:i + 1000])
    print(f"Uploaded {len(docs)} passages from {path}.")


def search(query, k=4):
    sc = SearchClient(ENDPOINT, INDEX_NAME, cred)
    return [{"id": r["id"], "title": r["title"], "url": r.get("url", ""),
             "content": r["content"], "score": r["@search.score"]}
            for r in sc.search(search_text=query, top=k)]


if __name__ == "__main__":
    create_index()
    srcs = sys.argv[1:] or [f for f in ["medhallu_3.jsonl", "medhallu_30.jsonl",
                                        "medhallu_statements.jsonl"] if os.path.exists(f)]
    for s in srcs:
        upload_corpus_from_jsonl(s)
    print("\n--- test query ---")
    for ev in search("visual acuity Landolt C Snellen E strabismus amblyopia", k=3):
        print(f"[{ev['score']:.2f}] {ev['title'][:70]}")
        print(f"  {ev['content'][:160]}...\n")
