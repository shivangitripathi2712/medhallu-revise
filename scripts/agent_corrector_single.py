#!/usr/bin/env python3
"""Black-box agentic medical hallucination corrector (Serper backend).
Autonomous tool-calling agent: the LLM decides when/what to search, reads results,
may search again, emits a corrected answer. Needs OPENAI_API_KEY, SERPER_API_KEY."""
import argparse, json, os, sys
import requests
from openai import OpenAI

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
client = OpenAI()

SYSTEM_PROMPT = (
    "You are a careful medical fact-checker. You will be given a medical statement that "
    "may contain hallucinated or factually incorrect claims, and optionally the question "
    "it is meant to answer.\n\n"
    "Your job: use the web_search tool to find authoritative sources (PubMed, NIH, peer-reviewed "
    "journals, medical references) that confirm or correct the statement. Do NOT rely on unverified "
    "prior knowledge -- verify with search first. You may search multiple times to refine.\n\n"
    "When you have enough evidence, output a FINAL corrected version of the statement that is "
    "factually consistent with what you found. Keep it concise and on-topic. If the statement is "
    "already correct, say so. If the web evidence is insufficient, state that explicitly rather "
    "than guessing.\n\n"
    "Begin your final answer with 'CORRECTED:' on its own line."
)
TOOLS = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web (Google via Serper) for authoritative medical sources. Returns titles, snippets, links.",
        "parameters": {"type": "object",
            "properties": {"query": {"type": "string", "description": "The search query."}},
            "required": ["query"]},
    },
}]

def web_search(query, k=5):
    if not SERPER_API_KEY:
        return "ERROR: SERPER_API_KEY not set."
    try:
        resp = requests.post("https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": k}, timeout=20)
        resp.raise_for_status()
        rows = resp.json().get("organic", []) or []
        if not rows:
            return "No results."
        return "\n".join(f"- {r.get('title','')}: {r.get('snippet','')} ({r.get('link','')})" for r in rows[:k])
    except Exception as e:
        return f"Search error: {e}"

def run_agent(paragraph, question="", model="gpt-3.5-turbo", max_steps=6):
    user = f"Statement to check:\n{paragraph}\n"
    if question:
        user += f"\nQuestion it should answer:\n{question}\n"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    for step in range(1, max_steps + 1):
        resp = client.chat.completions.create(model=model, messages=messages,
            tools=TOOLS, tool_choice="auto", temperature=0.0)
        msg = resp.choices[0].message
        entry = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            entry["tool_calls"] = [{"id": tc.id, "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in msg.tool_calls]
        messages.append(entry)
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                q = args.get("query", "")
                print(f"  [step {step}] SEARCH: {q}", file=sys.stderr)
                result = web_search(q)
                print(f"            -> {result.splitlines()[0][:100] if result else ''}...", file=sys.stderr)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            continue
        return msg.content or "(no answer)"
    return "(stopped: reached max_steps without a final answer)"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paragraph", nargs="?")
    ap.add_argument("--question", default="")
    ap.add_argument("--model", default="gpt-3.5-turbo")
    ap.add_argument("--max_steps", type=int, default=6)
    args = ap.parse_args()
    paragraph = args.paragraph or ("Syncope during bathing in infants is a manifestation of water-induced "
        "vasodilation and hypovolemia, which can lead to transient circulatory collapse.")
    question = args.question or "Syncope during bathing in infants, a pediatric form of water-induced urticaria?"
    print("="*70, file=sys.stderr); print("INPUT (hallucinated):", paragraph, file=sys.stderr); print("="*70, file=sys.stderr)
    answer = run_agent(paragraph, question, model=args.model, max_steps=args.max_steps)
    print("\n" + "="*70); print(answer); print("="*70)

if __name__ == "__main__":
    main()
