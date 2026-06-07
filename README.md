#  Hallucination Correction with REVISE

A retrieve-and-edit pipeline that corrects hallucinated content in medical text.
Given a statement that may contain factual errors, the system retrieves
supporting evidence and rewrites the statement to be consistent with that
evidence. It is built on the RARR/REVISE approach and evaluated on the
[MedHallu](https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu) benchmark.

## Overview

For each input paragraph (a hallucinated medical answer from MedHallu), the
pipeline runs four stages:

1. **Question generation** — an LLM produces fact-checking questions for the claim.
2. **Retrieval** — each question is used to fetch supporting evidence.
3. **Agreement gate** — an NLI check decides whether the evidence contradicts the
   claim; irrelevant evidence is discarded.
4. **Editing** — when the gate fires, an LLM rewrites the claim to match the evidence.

The output is a corrected paragraph plus the evidence and gate decision behind
each edit.

## Retrieval settings

The pipeline supports three evidence sources, used as separate experimental
conditions:

| Setting | Evidence source | Purpose |
|---------|----------------|---------|
| **Azure AI Search** | A corpus indexed in Azure (keyword search) | Tests correction under corpus-based retrieval. |
| **Web (Serper)** | Live Google search | Tests correction under open-web retrieval. |


## Repository structure

```
.
├── run_editor_sequential.py     # main correction pipeline
├── azure_search_retrieval.py    # Azure AI Search retriever
├── requirements.txt
│
├── scripts/
│   ├── build_medhallu_input.py      # build MedHallu input files
│   ├── detect_medhallu.py           # hallucination detection benchmark
│   ├── agent_corrector_single.py    # black-box agent (single example)
│   └── agent_corrector_batch.py     # black-box agent (batch)
│
├── inputs/                      # MedHallu dataset inputs
├── outputs/
│   ├── azure/                       # results using Azure AI Search
│   ├── tavily_serper/               # results using web search + agent
│   ├── gold_context/                # results using the gold passage
│   └── detection/                   # detection benchmark results
│
├── utils/                       # pipeline modules
│   ├── search.py                    # retrieval (Azure / web)
│   ├── agreement_gate.py            # evidence agreement gate
│   ├── editor.py                    # claim editing
│   └── question_generation.py       # question generation
│
└── prompts/                     # prompt templates
```

## Setup

The pipeline runs in a Python 3.10 environment:

```bash
conda create -n revise python=3.10 -y
conda activate revise
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt')"
```

Set the required keys (the LLM is always used; the search keys depend on the
retrieval setting):

```bash
export OPENAI_API_KEY="..."                 # always
export AZURE_SEARCH_ENDPOINT="..."          # for Azure AI Search
export AZURE_SEARCH_ADMIN_KEY="..."         # for Azure AI Search
export SERPER_API_KEY="..."                 # for web search
```

## Usage

**Build the dataset input:**
```bash
python scripts/build_medhallu_input.py
```

**(Azure setting) Build the search index:**
```bash
python azure_search_retrieval.py inputs/medhallu_statements.jsonl
```

**Run the correction pipeline:**
```bash
python run_editor_sequential.py \
  --input_file  inputs/medhallu_3.jsonl \
  --output_file outputs/azure/medhallu_azure_3.jsonl \
  --model gpt-3.5-turbo \
  --max_search_results_per_query 3
```

Add `--use_knowledge` to run the gold-context setting (uses the dataset's
reference passage instead of retrieval).

**Inspect results:**
```bash
python -c "
import json
for line in open('outputs/azure/medhallu_azure_3.jsonl'):
    o = json.loads(line)
    print('Hallucinated:', o.get('claim','')[:90])
    print('Ground truth:', o.get('long_answer','')[:110])
    print('Revised     :', o.get('revised_claim','')[:130])
    print('-'*60)
"
```

**Run the black-box agent baseline:**
```bash
python scripts/agent_corrector_batch.py \
  --input inputs/medhallu_3.jsonl \
  --output outputs/tavily_serper/agent_3.jsonl \
  --model gpt-4o-mini
```

**Run the detection benchmark** (a separate classification task — is an answer
hallucinated?):
```bash
python scripts/detect_medhallu.py --n 100              # without reference
python scripts/detect_medhallu.py --n 100 --knowledge  # with reference
```

## Output

Each run produces:
- a `.jsonl` file with the full records (original claim, revised claim, ground
  truth, and the evidence and gate decision per statement), and
- an `.xlsx` sheet comparing the hallucinated claim, ground truth, and revised
  claim side by side.

## Acknowledgement

This work builds on RARR (Gao et al., 2022), *Researching and Revising What
Language Models Say, Using Language Models*.
