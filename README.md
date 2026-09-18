# Self-Healing RAG with LangGraph

A reliability-oriented Retrieval-Augmented Generation system that uses a cyclic LangGraph workflow to detect weak or insufficient answers, reformulate the query, retrieve additional evidence, regenerate the answer, and evaluate whether the repaired response is better supported by the source documents.

## Overview

Standard RAG follows a mostly linear pipeline:

```text
Query
  ↓
Retrieve
  ↓
Generate
  ↓
Answer
```

Self-Healing RAG adds an evidence-driven recovery loop:

```text
Query
  ↓
Hybrid Retrieval
(Dense + BM25 + RRF)
  ↓
Generate
  ↓
Claim / Evidence Verification
  ↓
Critic
  ├── Accept ─────────────→ Final Answer
  │
  └── Repair
       ↓
   Query Rewrite
       ↓
   Re-retrieve
       ↓
   Regenerate
       ↓
   Critic
       ↓
   Quality Gate
       ├── Better → Accept
       └── Not better → Preserve / Abstain
```

The goal is not to retry every response. The system attempts recovery when the evidence indicates that the current answer needs repair, while avoiding unnecessary degradation of answers that are already satisfactory.

## Key Features

- LangGraph-based cyclic RAG workflow
- Dense semantic retrieval with Sentence Transformers
- BM25 lexical retrieval
- Reciprocal Rank Fusion (RRF)
- Evidence-aware answer generation
- Claim-level grounding analysis
- Deterministic citation attribution
- Evidence-based critic
- Query reformulation for recovery
- Bounded healing iterations
- Quality-gated answer selection
- Abstention when evidence is insufficient
- Standard RAG versus Self-Healing RAG evaluation
- Retrieval, answer-quality, grounding, citation, latency, and healing metrics
- Per-question and category-level evaluation
- Recovery and degradation analysis
- CSV reports for reproducible experiments
- Local open-source models with no paid API dependency

## Architecture

### Document Processing

```text
Documents
   ↓
Parsing
   ↓
Chunking
   ↓
Metadata
   ↓
Embeddings
```

Documents are divided into overlapping chunks while retaining source and chunk metadata.

### Hybrid Retrieval

```text
                    Query
                      │
             ┌────────┴────────┐
             ↓                 ↓
      Dense Retrieval        BM25
             │                 │
             └────────┬────────┘
                      ↓
             Reciprocal Rank
                 Fusion
                      ↓
                    Top-K
```

Dense retrieval captures semantic similarity. BM25 complements it with lexical matching for exact terms, names, numbers, and terminology.

### Reciprocal Rank Fusion

For a document or chunk `d`:

\[
RRF(d)=\sum_{m\in M}rac{1}{k+rank_m(d)}
\]

where `M` is the set of retrieval methods, `rank_m(d)` is the rank assigned by method `m`, and `k` is the smoothing constant.

### Generation

The generator receives the question and retrieved evidence and produces an answer constrained by the available source material.

### Claim-Level Evidence Verification

The generated response is decomposed into claims. Each claim is matched against retrieved evidence.

```text
Generated Answer
       ↓
     Claims
       ↓
C1    C2    C3
 │     │     │
 └─────┴─────┘
       ↓
Evidence Matching
       ↓
Support Scores
```

For claim `c` and evidence `e`:

\[
Support(c,e)=0.65	imes SemanticSimilarity(c,e)
+0.35	imes LexicalOverlap(c,e)
\]

The strongest evidence score for each claim is used for grounding analysis.

### Deterministic Citations

Citation attribution is performed by the evidence layer rather than relying solely on the generator to produce citation identifiers.

Example:

```text
The company was founded in 2018. [1]

The annual learning budget is $2,000. [2]
```

The citation numbers map to retrieved evidence chunks, allowing validity and support to be evaluated independently.

### Evidence Critic

The critic considers:

- Groundedness
- Relevance
- Completeness
- Citation support
- Confidence

It can accept an answer, request targeted healing, or stop when the available evidence is insufficient.

### Query Reformulation

When repair is required:

```text
Initial Query
     ↓
Critic identifies evidence deficiency
     ↓
Query Rewriter
     ↓
Reformulated Query
     ↓
Hybrid Retrieval
```

### Quality-Gated Healing

A repaired answer is not automatically accepted.

```text
Initial Answer
      ↓
    Critic
      ↓
    Repair
      ↓
New Answer
      ↓
Quality Gate
   ├── Better → Accept
   └── Not better → Preserve previous answer
```

This prevents additional generation from unnecessarily replacing a satisfactory answer with a weaker response.

### Bounded Recovery

Healing is limited to a configured maximum number of attempts. If sufficient evidence cannot be obtained, the system can abstain rather than repeatedly generate unsupported content.

## Models

### Embedding Model

`sentence-transformers/all-MiniLM-L6-v2`

Used for semantic embeddings and similarity calculations.

### Generation Model

`Qwen/Qwen2.5-0.5B-Instruct`

Used for answer generation and query reformulation.

The generation component can be replaced with another compatible open model without changing the overall workflow.

## Evaluation

The evaluation framework measures both conventional RAG quality and the specific contribution of the self-healing loop.

### Retrieval Metrics

#### Recall@K

\[
Recall@K=
rac{Relevant\ items\ retrieved\ in\ top\ K}
{Total\ relevant\ items}
\]

#### Mean Reciprocal Rank (MRR)

\[
MRR=rac{1}{N}\sum_{i=1}^{N}rac{1}{rank_i}
\]

where `rank_i` is the position of the first relevant result for query `i`.

#### Mean Average Precision (MAP)

\[
MAP=rac{1}{N}\sum_{i=1}^{N}AP_i
\]

where `AP_i` is average precision for query `i`.

#### nDCG@K

\[
DCG@K=\sum_{i=1}^{K}rac{rel_i}{\log_2(i+1)}
\]

\[
nDCG@K=rac{DCG@K}{IDCG@K}
\]

## Answer Metrics

### Keyword Recall

\[
KeywordRecall=
rac{|ExpectedKeywords\cap RetrievedKeywords|}
{|ExpectedKeywords|}
\]

### Answer Keyword Accuracy

\[
AnswerKeywordAccuracy=
rac{|ExpectedKeywords\cap AnswerKeywords|}
{|ExpectedKeywords|}
\]

### Expected Answer Overlap

\[
Overlap=
rac{|ReferenceTokens\cap AnswerTokens|}
{|ReferenceTokens|}
\]

### Token F1

\[
Precision=rac{Overlap}{AnswerTokens}
\]

\[
Recall=rac{Overlap}{ReferenceTokens}
\]

\[
F1=rac{2PR}{P+R}
\]

### Exact Match

A normalized generated answer is compared with the normalized reference answer:

\[
ExactMatch=
egin{cases}
1 & 	ext{if normalized answers match}\
0 & 	ext{otherwise}
\end{cases}
\]

## Grounding Metrics

### Grounded Rate

The proportion of responses whose factual claims satisfy the configured evidence-support criterion.

### Grounded Score

For each claim, the strongest supporting evidence score is used:

\[
GroundedScore=
rac{1}{C}\sum_{j=1}^{C}Support(c_j)
\]

where `C` is the number of evaluated claims.

### Supported Claim Rate

\[
SupportedClaimRate=
rac{SupportedClaims}{TotalClaims}
\]

## Citation Metrics

### Citation Validity Rate

\[
CitationValidity=
rac{ValidCitations}{TotalCitations}
\]

A citation is valid when it refers to an actual retrieved evidence item.

### Citation Supported Rate

\[
CitationSupportedRate=
rac{SupportedCitations}{EvaluatedCitations}
\]

A citation is supported when the referenced evidence actually supports the associated claim.

## Relevance

Answer relevance is calculated using semantic similarity between question and answer embeddings:

\[
Relevance=CosineSimilarity(E_q,E_a)
\]

where `E_q` is the question embedding and `E_a` is the answer embedding.

## Confidence

The reported confidence is a heuristic composite score, not a calibrated probability:

\[
Confidence=
0.45G+0.30R+0.15C+0.10K
\]

where:

- `G` = grounded score
- `R` = relevance score
- `C` = supported claim rate
- `K` = completeness score

## Self-Healing Metrics

### First-Attempt Failure

A query is a first-attempt failure when its initial response does not satisfy the configured benchmark success criteria.

### Recovery Rate

\[
RecoveryRate=
rac{RecoveredAfterHealing}{InitialFailures}
\]

This measures how often the healing loop successfully repairs an initially failed response.

### Healing Attempts

The number of additional repair cycles executed after the initial generation.

### Degradation Rate

\[
DegradationRate=
rac{HealedAnswersThatBecameWorse}{HealedAnswers}
\]

This measures cases where healing changes a previously satisfactory answer into an inferior result.

### F1 Delta

\[
F1Delta=F1_{healed}-F1_{initial}
\]

Positive values indicate improvement for a query; negative values indicate degradation.

### Latency

End-to-end execution time for the evaluated request.

### Latency Overhead

\[
LatencyOverhead=
Latency_{SelfHealing}-Latency_{Initial}
\]

This quantifies the computational cost of the recovery mechanism.

## Evaluation Modes

### Standard RAG

```text
Query
 ↓
Hybrid Retrieval
 ↓
Generation
 ↓
Evaluation
```

### Self-Healing RAG

```text
Query
 ↓
Hybrid Retrieval
 ↓
Generation
 ↓
Evidence Verification
 ↓
Critic
 ├── Accept → Final
 └── Repair
       ↓
   Query Reformulation
       ↓
   Re-retrieval
       ↓
   Regeneration
       ↓
   Quality Gate
```

The baseline provides a reference point for determining whether the healing loop produces measurable recovery.

## Evaluation Outputs

Running:

```bash
python evaluate.py
```

generates:

```text
evaluation_results.csv
metrics_summary.csv
category_metrics.csv
healing_analysis.csv
healing_summary.csv
```

### `evaluation_results.csv`

Per-question results including answer quality, grounding, citations, confidence, latency, retries, and benchmark outcome.

### `metrics_summary.csv`

Aggregate metrics for Standard RAG and Self-Healing RAG.

### `category_metrics.csv`

Metrics grouped by question category.

### `healing_analysis.csv`

Per-question healing outcomes including initial failures, recovery, degradation, F1 changes, and latency changes.

### `healing_summary.csv`

Aggregate recovery and degradation statistics.

## Evaluation Categories

The benchmark includes:

- `direct_factual`
- `paraphrase`
- `numerical`
- `policy`
- `security`
- `negative`
- `unanswerable`

Category-level evaluation helps identify where the healing mechanism is useful rather than relying only on a single aggregate score.

## Installation

```bash
git clone https://github.com/Arjun-08/Self-Healing-RAG-LangGraph.git
cd Self-Healing-RAG-LangGraph
```

Create an environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scriptsctivate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run an interactive query:

```bash
python run.py
```

Run the complete evaluation:

```bash
python evaluate.py
```

## Repository Structure

```text
Self-Healing-RAG-LangGraph/
│
├── rag_engine.py
│   ├── Document processing
│   ├── Dense retrieval
│   ├── BM25 retrieval
│   ├── RRF fusion
│   ├── Generation
│   ├── Claim analysis
│   ├── Citation attribution
│   ├── Evidence critic
│   ├── Query rewriting
│   └── LangGraph workflow
│
├── evaluate.py
│   ├── Benchmark execution
│   ├── Retrieval metrics
│   ├── Answer metrics
│   ├── Grounding metrics
│   ├── Citation metrics
│   ├── Healing metrics
│   └── CSV report generation
│
├── run.py
├── requirements.txt
│
├── data/
│   ├── sample_company_handbook.txt
│   ├── evaluation.json
│   └── README.md
│
└── README.md
```

## Reproducibility

For a meaningful comparison:

1. Use the same source documents for both modes.
2. Use the same evaluation questions.
3. Keep retrieval configuration fixed.
4. Keep the generation model fixed.
5. Report latency separately from quality metrics.
6. Report recovery and degradation, not only aggregate accuracy.
7. Expand the benchmark before making strong general claims.

The supplied benchmark is a development benchmark. A larger manually verified evaluation set should be used for stronger empirical conclusions.

## Interpreting the Experiment

The intended behavior is:

```text
Sufficient initial evidence
        ↓
Accept and preserve answer
```

and:

```text
Insufficient or weak evidence
        ↓
Detect failure
        ↓
Reformulate query
        ↓
Retrieve additional evidence
        ↓
Regenerate
        ↓
Accept only if quality improves
```

The central measurements are therefore:

- Initial failure rate
- Recovery rate
- Degradation rate
- Grounding improvement
- Citation-support improvement
- Answer-quality change
- Latency overhead

The project evaluates self-healing as a reliability mechanism rather than treating repeated LLM calls as an improvement by default.

## Limitations

- The default generation model is small and can produce weaker responses than larger instruction-tuned models.
- Local CPU inference can result in substantial latency.
- Semantic and lexical evidence scores are heuristic.
- Confidence is not a calibrated probability.
- The supplied benchmark is small.
- Claim extraction and evidence matching are approximate.
- Self-healing cannot recover information absent from the source corpus; in such cases, abstention is the appropriate behavior.

## Future Extensions

- Larger manually verified evaluation datasets
- Multi-hop evaluation
- Cross-encoder reranking
- Stronger open-source generation models
- LLM-as-a-judge evaluation
- More sophisticated claim decomposition
- Learned healing policies
- Retrieval confidence calibration
- Evaluation on external document collections
- Experiment tracking and latency profiling

