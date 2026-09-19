# Self-Healing RAG with LangGraph

A reliability-oriented Retrieval-Augmented Generation system that uses a cyclic LangGraph workflow to detect weak or insufficient answers, reformulate the query, retrieve additional evidence, regenerate the answer, and evaluate whether the repaired response is better supported by the source documents.

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

For a document or chunk $d$:

$$
RRF(d) =
\sum_{m \in M}
\frac{1}{k + \{rank}_m(d)}
$$

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

$$
Support(c,e) = 0.65 \cdot SemanticSimilarity(c,e) + 0.35 \cdot LexicalOverlap(c,e)
$$

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
### Recall@K

$$
Recall@K =
\frac{\text{Relevant items retrieved in top } K}
{\text{Total relevant items}}
$$

#### Mean Reciprocal Rank (MRR)

$$
MRR =
\frac{1}{N}
\sum_{i=1}^{N}
\frac{1}{rank_i}
$$

where `rank_i` is the position of the first relevant result for query `i`.

#### Mean Average Precision (MAP)

$$
MAP =
\frac{1}{N}
\sum_{i=1}^{N}
AP_i
$$

where `AP_i` is average precision for query `i`.

#### nDCG@K

$$
DCG@K =
\sum_{i=1}^{K}
\frac{rel_i}{\log_2(i+1)}
$$

$$
nDCG@K =
\frac{DCG@K}{IDCG@K}
$$


## Answer Metrics

### Keyword Recall

$$
\text{KeywordRecall} =
\frac{|\text{ExpectedKeywords} \cap \text{RetrievedKeywords}|}
{|\text{ExpectedKeywords}|}
$$

### Answer Keyword Accuracy

$$
\text{AnswerKeywordAccuracy} =
\frac{|\text{ExpectedKeywords} \cap \text{AnswerKeywords}|}
{|\text{ExpectedKeywords}|}
$$

### Expected Answer Overlap

$$
\text{Overlap} =
\frac{|\text{ReferenceTokens} \cap \text{AnswerTokens}|}
{|\text{ReferenceTokens}|}
$$

### Token F1

$$
\text{Precision} =
\frac{\text{Overlap}}
{|\text{AnswerTokens}|}
$$

$$
\text{Recall} =
\frac{\text{Overlap}}
{|\text{ReferenceTokens}|}
$$

$$
F1 =
\frac{2PR}{P+R}
$$

### Exact Match

A normalized generated answer is compared with the normalized reference answer:

$$
\text{ExactMatch} =
\begin{cases}
1 & \text{if normalized answers match} \\
0 & \text{otherwise}
\end{cases}
$$


## Grounding Metrics

### Grounded Rate

The proportion of responses whose factual claims satisfy the configured evidence-support criterion.

### Grounded Score

For each claim, the strongest supporting evidence score is used:

$$
\text{GroundedScore} =
\frac{1}{C}
\sum_{j=1}^{C}
\text{Support}(c_j)
$$

where `C` is the number of evaluated claims.

### Supported Claim Rate

$$
\text{SupportedClaimRate} =
\frac{\text{SupportedClaims}}
{\text{TotalClaims}}
$$

## Citation Metrics

### Citation Validity Rate

$$
\text{CitationValidity} =
\frac{\text{ValidCitations}}
{\text{TotalCitations}}
$$

A citation is valid when it refers to an actual retrieved evidence item.

### Citation Supported Rate

$$
\text{CitationSupportedRate} =
\frac{\text{SupportedCitations}}
{\text{EvaluatedCitations}}
$$

A citation is supported when the referenced evidence actually supports the associated claim.

## Relevance

Answer relevance is calculated using semantic similarity between question and answer embeddings:

$$
\text{Relevance} =
\text{CosineSimilarity}(E_q, E_a)
$$

where `E_q` is the question embedding and `E_a` is the answer embedding.

## Confidence

The reported confidence is a heuristic composite score, not a calibrated probability:

$$
\text{Confidence} =
0.45G + 0.30R + 0.15C + 0.10K
$$

where:

* `G` = grounded score
* `R` = relevance score
* `C` = supported claim rate
* `K` = completeness score

## Self-Healing Metrics

### First-Attempt Failure

A query is a first-attempt failure when its initial response does not satisfy the configured benchmark success criteria.

### Recovery Rate

$$
\text{RecoveryRate} =
\frac{\text{RecoveredAfterHealing}}
{\text{InitialFailures}}
$$

This measures how often the healing loop successfully repairs an initially failed response.

### Healing Attempts

The number of additional repair cycles executed after the initial generation.

### Degradation Rate

$$
\text{DegradationRate} =
\frac{\text{HealedAnswersThatBecameWorse}}
{\text{HealedAnswers}}
$$

This measures cases where healing changes a previously satisfactory answer into an inferior result.

### F1 Delta

$$
\text{F1Delta} =
F1_{\text{healed}} - F1_{\text{initial}}
$$

Positive values indicate improvement for a query; negative values indicate degradation.

### Latency

End-to-end execution time for the evaluated request.

### Latency Overhead

$$
LatencyOverhead = Latency_{SelfHealing} - Latency_{Initial}
$$

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


## Results and Discussion

| Metric | Self-Healing RAG |
|---|---:|
| Recall@K | 1.000 |
| MRR | 0.962 |
| MAP | 0.962 |
| nDCG@K | 0.972 |
| Keyword Recall | 0.812 |
| Answer Keyword Accuracy | 0.706 |
| Expected Answer Overlap | 0.605 |
| Answer Token F1 | 0.516 |
| Relevance Score | 0.864 |
| Confidence | 0.514 |
| Average Latency | 40.68 s |

The Self-Healing RAG system demonstrates a **strong retrieval foundation**, achieving **Recall@K of 1.00, MRR of 0.962, MAP of 0.962, and nDCG@K of 0.972**. These results indicate that the system consistently retrieves relevant evidence and ranks useful information near the top of the retrieved context.

At the answer level, the system achieves a **keyword recall of 0.812**, **answer keyword accuracy of 0.706**, and **expected answer overlap of 0.605**, showing meaningful alignment between generated responses and the expected information. The **relevance score of 0.864** further indicates that the generated responses remain strongly focused on the user's query.

The Self-Healing RAG architecture adds verification and bounded recovery stages after generation. These stages provide the foundation for detecting weaknesses in generated responses and attempting targeted correction through query reformulation, re-retrieval, and regeneration. The measured latency of **40.68 seconds** reflects the additional computational cost of this verification and recovery pipeline and provides a baseline for future optimization.

---

## Future Development

Future development will focus on making the Self-Healing RAG pipeline more intelligent, reliable, and efficient. The critic can be improved to identify why an answer is incorrect and choose the right recovery method.

Query rewriting can be made more targeted based on the detected error. Better claim-level verification can help retrieve relevant evidence and check the regenerated answer before accepting it.

A quality gate can compare the original and healed answers and keep the better one. Deterministic citations and an abstention mechanism can further improve reliability by ensuring that claims are supported by evidence.

> **Note:** The `app.py` file corresponds to an earlier version of the project that includes the Streamlit interface. The latest Self-Healing RAG implementation has not yet been integrated into `app.py`, as further development is still ongoing.
