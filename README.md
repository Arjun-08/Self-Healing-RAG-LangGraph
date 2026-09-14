# Self-Healing RAG

A reliability-focused Retrieval-Augmented Generation system built with LangGraph.

Instead of stopping after one retrieval + generation step, the system evaluates its own answer. If the answer is weakly grounded, poorly aligned, or missing valid evidence citations, it reformulates the search query, retrieves again, and regenerates the answer. It stops after a configurable retry limit and avoids inventing unsupported information.

## Architecture

```text
                         User Query
                             |
                             v
                    +------------------+
                    | Hybrid Retrieval |
                    | Dense + BM25     |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |  LLM Generator   |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    | Critic / Scorer  |
                    | Grounding        |
                    | Relevance        |
                    | Citation         |
                    +--------+---------+
                             |
                  +----------+----------+
                  |                     |
                 PASS                  FAIL
                  |                     |
                  v                     v
              Final Answer       Query Rewriter
                                        |
                                        v
                                Re-retrieve evidence
                                        |
                                        +----> Generate
```

## Features

- LangGraph stateful cyclic workflow
- Hybrid dense + BM25 retrieval
- Reciprocal Rank Fusion
- Evidence citations
- Self-critique and safe retry
- Query reformulation
- Maximum retry limit
- Standard RAG vs Self-Healing RAG comparison
- Retrieval, grounding, relevance, citation, confidence and latency metrics
- PDF, DOCX and TXT ingestion
- Streamlit interface
- Fully local open models; no paid LLM API is required

## Models

Embedding model:

`sentence-transformers/all-MiniLM-L6-v2`

Generation model:

`Qwen/Qwen2.5-0.5B-Instruct`

The Qwen model is Apache-2.0 licensed. The MiniLM embedding model is Apache-2.0 licensed.

The small generation model is selected to make the project runnable without an API key. For a stronger local deployment, replace it with a larger instruction model when hardware permits.

## Project structure

```text
self-healing-rag/
├── app.py
├── rag_engine.py
├── evaluate.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
└── data/
    ├── sample_company_handbook.txt
    ├── evaluation.json
    └── README.md
```

## Run locally

Use Python 3.11.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

The first run downloads the embedding and generation models from Hugging Face. Subsequent runs use the local Hugging Face cache.

## Evaluation

Run:

```bash
python evaluate.py
```

This compares:

- Standard RAG
- Self-Healing RAG

The included evaluation set is intentionally small and demonstrates the pipeline. For a serious portfolio benchmark, create 100–500 manually verified questions and record ground-truth relevant documents and expected answers.

Metrics currently exposed include:

- Retrieval keyword recall
- Answer keyword accuracy
- Groundedness
- Citation validity
- Confidence
- Latency
- Retry count

Do not claim benchmark improvements until you run the evaluation and record the actual numbers.

## Streamlit deployment

Push the repository to GitHub.

Then open Streamlit Community Cloud and create an app from:

```text
app.py
```

The repository should contain `requirements.txt` in the root.

No API secret is needed because the demo uses local Hugging Face models.

Important: the generation model is roughly 1 GB before runtime memory, so Community Cloud may be slower or may hit resource limits depending on the current environment. If that happens, use a hosted inference endpoint or deploy the same code on a machine with more RAM.

## Recommended portfolio experiment

The strongest result is not simply showing that the application works.

Create a benchmark:

```text
                    Standard RAG    Self-Healing RAG
Faithfulness            X%                Y%
Answer accuracy         X%                Y%
Citation accuracy       X%                Y%
Retrieval recall        X%                Y%
Avg latency             Xs                Ys
Avg retries             0                 Z
```

Then analyze failure cases where the initial answer was rejected and the retry recovered the correct evidence.

## Future work

- Cross-encoder reranking
- LLM-as-a-judge evaluation
- RAGAS/DeepEval integration
- Persistent vector database
- User feedback collection
- Token/cost tracking
- Distributed inference
- Better query decomposition
- Multi-hop retrieval
- Evaluation dataset with human annotations
