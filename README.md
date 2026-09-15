# Self-Healing RAG with LangGraph

A reliability-focused Retrieval-Augmented Generation system that does more than retrieve and generate. It evaluates its own answer at the claim level, checks evidence citations, reformulates the retrieval query when evidence is weak, retries within a bounded LangGraph cycle, and stops safely when sufficient evidence cannot be found.

## Why this project is different

A conventional RAG pipeline is usually:

```text
Question → Retrieve → Generate → Answer
```

This project implements:

```text
Question
   ↓
Hybrid Retrieval
(Dense + BM25 → RRF)
   ↓
Generate answer + citations
   ↓
Evidence Critic
   ├── PASS → Final answer
   └── FAIL
         ↓
    Query Rewriter
         ↓
    Re-retrieve
         ↓
    Generate
         ↓
    Critic
         ↓
    PASS or bounded retry
```

The important contribution is not simply adding another LLM call. The system measures whether a generated claim is actually supported by retrieved evidence and uses the result to decide whether the workflow should continue.

## Features

- Stateful cyclic workflow using LangGraph
- Hybrid dense + BM25 retrieval
- Reciprocal Rank Fusion (RRF)
- Claim-level evidence support scoring
- Citation validity and citation-support checks
- Automatic query reformulation
- Bounded self-healing retries
- Safe stopping after the retry limit
- Standard RAG vs Self-Healing RAG benchmark
- Retrieval, answer, grounding, citation, confidence and latency metrics
- Recovery-rate measurement for initially failed queries
- TXT, PDF and DOCX ingestion
- Streamlit interface with execution trace
- Fully local open models; no paid API key required

## Models

Embedding:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Generation and query rewriting:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

The small generator is chosen for a no-API-key demo. A larger instruction model can be substituted when more compute is available.

## Project structure

```text
self-healing-rag/
├── app.py
├── rag_engine.py
├── evaluate.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
└── data/
    ├── sample_company_handbook.txt
    ├── evaluation.json
    └── README.md
```

## Installation

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

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the application

```bash
streamlit run app.py
```

The first run downloads the Hugging Face embedding and generation models. Later runs use the local Hugging Face cache.

## Run the benchmark

```bash
python evaluate.py
```

The script writes:

```text
evaluation_results.csv
healing_summary.csv
```

The benchmark compares:

- Standard RAG: one retrieval → generation → critique pass
- Self-Healing RAG: retrieval → generation → critique → reformulation → retry, up to the configured limit

## Metrics

### Retrieval

- Retrieval keyword recall

For a serious benchmark, expand this to Recall@K, MRR and nDCG with manually annotated relevant chunks.

### Generation

- Answer keyword accuracy
- Expected-answer overlap

### Reliability

- Grounded rate
- Groundedness score
- Citation validity
- Citation support
- Relevance score
- Confidence

### System behavior

- Average latency
- Retry count
- Standard-RAG failure count
- Recovery count
- Self-Healing recovery rate

## The key experiment

The most important result is not raw answer accuracy. Measure whether the healing loop actually recovers failures.

```text
Standard RAG
    ↓
initial failure
    ↓
Self-Healing retry
    ↓
correct + grounded + cited
```

Recovery rate:

```text
recovered standard-RAG failures
--------------------------------
all standard-RAG failures
```


## Important interpretation

Self-Healing RAG should not be expected to improve every metric. It may improve reliability while increasing latency and LLM calls. That trade-off is part of the engineering result.

A strong final analysis should answer:

1. How often does the first RAG answer fail?
2. How often does the healing loop recover it?
3. Which failure types are recoverable?
4. How much additional latency does healing introduce?
5. When should the system stop retrying and say that evidence is insufficient?

## Streamlit deployment

Push the repository to GitHub and deploy `app.py` using Streamlit Community Cloud.

The repository root must contain:

```text
app.py
requirements.txt
```

No API secret is required by this implementation.

Because the generator runs locally, hosted CPU environments can be considerably slower than a GPU machine. If deployment memory or latency becomes a problem, replace only the `LocalLLM` implementation with a hosted inference backend; the LangGraph, retrieval, critic and evaluation architecture can remain unchanged.

## Future improvements

- Cross-encoder reranker
- RAGAS/DeepEval or a separate LLM-as-a-judge evaluation layer
- Persistent vector database
- Human feedback collection
- Token/cost tracking
- Query decomposition and multi-hop retrieval
- Better document metadata filtering
- GPU/quantized inference
- Larger human-annotated evaluation set
- Production tracing and monitoring
