#!/usr/bin/env python3
"""
make_medhallu_input.py
----------------------
Build an input JSONL file for run_editor_sequential.py from the MedHallu
benchmark (UTAustin-AIHealth/MedHallu) on Hugging Face.

Idea of the experiment
    Each MedHallu sample contains a `Hallucinated Answer`: a plausible but
    factually wrong medical statement. We treat that hallucinated answer as the
    misinformation passage (the claim to be fact-checked and revised). The
    REVISE/RARR pipeline then retrieves evidence from the open web (Tavily) and
    tries to correct the claim. We can compare the revised claim against the
    MedHallu `Ground Truth` to see whether the hallucination was fixed.

How it maps to the pipeline
    run_editor_sequential.py reads the claim from a TOP-LEVEL key (in JSONL mode
    the priority is: claim > text > passage > long_answer > misinformation_passage).
    So each output line sets:
        claim          = Hallucinated Answer   <- what the pipeline fact-checks
        question_text  = Question              <- shows in the Excel "Question" column
        long_answer    = Ground Truth          <- shows in the Excel "Ground Truth" column
    Because `claim` has priority over `long_answer` in JSONL mode, setting
    long_answer for the Excel does NOT hijack the claim.

    Evidence retrieval is done by the pipeline via Tavily (open web). The
    MedHallu `Knowledge` field is carried along only as reference metadata; it
    is NOT used for retrieval.

This script does not modify any existing project file. It only creates a new
input JSONL (default: medhallu_statements.jsonl).

Usage
    python make_medhallu_input.py                          # first 5 from pqa_labeled
    python make_medhallu_input.py -n 10                     # first 10
    python make_medhallu_input.py --config pqa_artificial   # use the 9k auto-generated split
    python make_medhallu_input.py --out my_input.jsonl      # custom output path

Requires: pip install datasets
"""

import argparse
import json
import sys

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("Missing dependency. Run:  pip install datasets")


def main():
    ap = argparse.ArgumentParser(
        description="Convert MedHallu hallucinated answers into pipeline input JSONL."
    )
    ap.add_argument("-n", "--num", type=int, default=5,
                    help="Number of samples to take from the start of the split (default: 5).")
    ap.add_argument("--config", default="pqa_labeled",
                    choices=["pqa_labeled", "pqa_artificial"],
                    help="MedHallu config (default: pqa_labeled, the 1k human-annotated set).")
    ap.add_argument("--split", default="train",
                    help="Dataset split (default: train).")
    ap.add_argument("--out", default="medhallu_statements.jsonl",
                    help="Output JSONL path (default: medhallu_statements.jsonl).")
    args = ap.parse_args()

    print(f"Loading MedHallu '{args.config}' (split='{args.split}') ...", file=sys.stderr)
    ds = load_dataset("UTAustin-AIHealth/MedHallu", args.config, split=args.split)
    print(f"Loaded {len(ds)} total samples.", file=sys.stderr)

    n = min(args.num, len(ds))
    written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(n):
            row = ds[i]
            hallucinated = (row.get("Hallucinated Answer") or "").strip()
            if not hallucinated:
                print(f"  [skip] row {i}: empty Hallucinated Answer", file=sys.stderr)
                continue

            # Knowledge is a list[str] (PubMed contexts); flatten for reference only.
            knowledge = row.get("Knowledge")
            if isinstance(knowledge, list):
                knowledge_text = " ".join(k for k in knowledge if k)
            else:
                knowledge_text = knowledge or ""

            record = {
                # --- field the pipeline actually fact-checks ---
                "claim": hallucinated,
                # --- fields the Excel exporter reads (populate its columns) ---
                "question_text": (row.get("Question") or "").strip(),
                "long_answer": (row.get("Ground Truth") or "").strip(),
                # --- reference metadata (carried through to output JSONL) ---
                "misinformation_passage": hallucinated,
                "medhallu_index": i,
                "difficulty": (row.get("Difficulty Level") or "").strip(),
                "category": (row.get("Category of Hallucination") or "").strip(),
                "knowledge": knowledge_text,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"\nWrote {written} record(s) to {args.out}", file=sys.stderr)

    # Preview so you can eyeball the claims before spending any API calls.
    print("\n--- preview (claim = hallucinated answer) ---", file=sys.stderr)
    with open(args.out, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            q = obj["question_text"][:70]
            c = obj["claim"][:90]
            print(f"  [{obj['medhallu_index']}] Q: {q}...", file=sys.stderr)
            print(f"        claim: {c}...", file=sys.stderr)


if __name__ == "__main__":
    main()