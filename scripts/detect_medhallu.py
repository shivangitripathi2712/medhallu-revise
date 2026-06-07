"""
Evaluate gpt-4o-mini (Azure OpenAI) on MedHallu hallucination detection.

Task: binary classification. Given a medical question (+ optional knowledge)
and an answer, the model judges whether the answer is hallucinated.
Each MedHallu sample yields 2 eval rows: ground-truth answer (label 0)
and hallucinated answer (label 1) -> balanced eval set.

Usage:
  export AZURE_OPENAI_API_KEY="your-key"
  python eval_medhallu.py --n 100                  # without knowledge, 200 API calls
  python eval_medhallu.py --n 100 --knowledge      # with-knowledge setting
  python eval_medhallu.py --n 0                    # full 1k set (2000 calls)

Outputs: medhallu_results_<setting>.csv + printed metrics
"""

import os
import re
import sys
import time
import json
import random
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from datasets import load_dataset
from openai import AzureOpenAI, RateLimitError, APIError
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ---------------- config ----------------
ENDPOINT = "https://tjk86-7265-resource.openai.azure.com/"
DEPLOYMENT = "gpt-4o-mini"          # your deployment name in Foundry
API_VERSION = "2024-10-21"
DATASET = "UTAustin-AIHealth/MedHallu"
CONFIG = "pqa_labeled"              # 1k high-quality split
SEED = 42
MAX_RETRIES = 6

client = AzureOpenAI(
    azure_endpoint=ENDPOINT,
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=API_VERSION,
)

SYSTEM_PROMPT = (
    "You are an expert medical fact-checker. You judge whether an answer to a "
    "medical question contains hallucinated content: factual contradictions, "
    "fabricated findings, or claims unsupported by medical evidence. "
    "Follow the output format exactly."
)


def build_prompt(question: str, answer: str, knowledge: str | None) -> str:
    parts = []
    if knowledge:
        parts.append(f"#World Knowledge#:\n{knowledge}")
    parts.append(f"#Question#:\n{question}")
    parts.append(f"#Answer#:\n{answer}")
    parts.append(
        "Is the answer hallucinated (factually incorrect, fabricated, or "
        "contradicting medical knowledge)? Reply with exactly one line:\n"
        '#Judgement#: Yes\nor\n#Judgement#: No'
    )
    return "\n\n".join(parts)


def call_model(prompt: str) -> str:
    delay = 2.0
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=DEPLOYMENT,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=16,
            )
            return resp.choices[0].message.content or ""
        except (RateLimitError, APIError) as e:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 60)
    return ""


def parse_judgement(text: str) -> int | None:
    """1 = hallucinated, 0 = not, None = unparseable."""
    m = re.search(r"judgement\s*#?\s*:\s*(yes|no)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"\b(yes|no)\b", text, re.IGNORECASE)
    if not m:
        return None
    return 1 if m.group(1).lower() == "yes" else 0


def get_col(cols, target: str) -> str:
    """Case/format-insensitive column lookup."""
    norm = lambda s: s.lower().replace("_", " ").strip()
    for c in cols:
        if norm(c) == norm(target):
            return c
    raise KeyError(f"Column '{target}' not found. Available: {list(cols)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100,
                    help="number of MedHallu samples (0 = all). Each sample = 2 API calls.")
    ap.add_argument("--knowledge", action="store_true",
                    help="include World Knowledge in the prompt (easier setting)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    print(f"Loading {DATASET} ({CONFIG}) ...")
    ds = load_dataset(DATASET, CONFIG, split="train")
    df = ds.to_pandas()

    q_col = get_col(df.columns, "Question")
    k_col = get_col(df.columns, "Knowledge")
    gt_col = get_col(df.columns, "Ground Truth")
    ha_col = get_col(df.columns, "Hallucinated Answer")
    diff_col = get_col(df.columns, "Difficulty Level")
    cat_col = get_col(df.columns, "Category of Hallucination")

    random.seed(SEED)
    if args.n and args.n < len(df):
        df = df.sample(n=args.n, random_state=SEED).reset_index(drop=True)
    print(f"{len(df)} samples -> {2 * len(df)} judgements "
          f"({'with' if args.knowledge else 'without'} knowledge)")

    # build balanced eval rows: each sample -> GT (label 0) + hallucinated (label 1)
    rows = []
    for i, r in df.iterrows():
        know = str(r[k_col]) if args.knowledge else None
        for answer, label in [(r[gt_col], 0), (r[ha_col], 1)]:
            rows.append({
                "sample_id": i,
                "question": str(r[q_col]),
                "answer": str(answer),
                "knowledge": know,
                "label": label,
                "difficulty": str(r[diff_col]),
                "category": str(r[cat_col]) if label == 1 else "n/a (ground truth)",
            })

    def worker(row):
        raw = call_model(build_prompt(row["question"], row["answer"], row["knowledge"]))
        row["raw_output"] = raw.strip()
        row["pred"] = parse_judgement(raw)
        return row

    results, done = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(worker, r) for r in rows]
        for f in as_completed(futures):
            results.append(f.result())
            done += 1
            if done % 25 == 0 or done == len(rows):
                print(f"  {done}/{len(rows)}", flush=True)

    res = pd.DataFrame(results)
    unparsed = res["pred"].isna().sum()
    res["pred_filled"] = res["pred"].fillna(1 - res["label"])  # unparseable counted wrong

    y_true, y_pred = res["label"], res["pred_filled"].astype(int)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0)

    setting = "with_knowledge" if args.knowledge else "no_knowledge"
    print("\n========== RESULTS ==========")
    print(f"Setting:    {setting}   |  n judgements: {len(res)}  |  unparseable: {unparsed}")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {p:.4f}   (positive class = hallucinated)")
    print(f"Recall:     {r:.4f}")
    print(f"F1:         {f1:.4f}")

    print("\n--- F1 by difficulty (hallucinated rows detection) ---")
    for d, g in res.groupby("difficulty"):
        _, _, f1d, _ = precision_recall_fscore_support(
            g["label"], g["pred_filled"].astype(int),
            average="binary", pos_label=1, zero_division=0)
        print(f"  {d:<10} n={len(g):<5} F1={f1d:.4f}")

    print("\n--- Recall by hallucination category ---")
    hall = res[res["label"] == 1]
    for c, g in hall.groupby("category"):
        rec = (g["pred_filled"] == 1).mean()
        print(f"  {c:<35} n={len(g):<5} recall={rec:.4f}")

    out_csv = f"medhallu_results_{setting}.csv"
    res.drop(columns=["knowledge"]).to_csv(out_csv, index=False)
    with open(f"medhallu_metrics_{setting}.json", "w") as fp:
        json.dump({"setting": setting, "n": len(res), "accuracy": acc,
                   "precision": p, "recall": r, "f1": f1,
                   "unparseable": int(unparsed)}, fp, indent=2)
    print(f"\nSaved: {out_csv}, medhallu_metrics_{setting}.json")


if __name__ == "__main__":
    main()