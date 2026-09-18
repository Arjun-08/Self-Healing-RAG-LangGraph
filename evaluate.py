from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from rag_engine import EMBED_MODEL, GEN_MODEL, HybridRetriever, LocalLLM, build_graph, load_document_bytes, run_rag

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
EVAL_FILE = DATA_DIR / "evaluation.json"
DOC_FILE = DATA_DIR / "sample_company_handbook.txt"


STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "to", "of", "and", "in", "on", "for", "with", "from", "by", "or", "as", "what", "when", "where", "who", "which", "how", "does", "do", "did", "can", "may", "must", "should", "their", "they", "them", "this", "that"
}


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"\b[a-zA-Z0-9][a-zA-Z0-9_-]*\b", text.lower()) if t not in STOPWORDS]


def keyword_recall(expected_keywords: Iterable[str], text: str) -> float:
    keys = [k.lower() for k in expected_keywords]
    if not keys:
        return 0.0
    text = text.lower()
    return sum(k in text for k in keys) / len(keys)


def token_f1(expected: str, actual: str) -> float:
    a = tokens(expected)
    b = tokens(actual)
    if not a or not b:
        return 0.0
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(b)
    recall = overlap / len(a)
    return 2 * precision * recall / (precision + recall)


def exact_match(expected: str, actual: str) -> float:
    clean = lambda x: re.sub(r"\W+", " ", x.lower()).strip()
    return float(clean(expected) == clean(actual))


def answer_contains_keywords(item: dict, answer: str) -> float:
    return keyword_recall(item.get("keywords", []), answer)


def expected_overlap(expected: str, answer: str) -> float:
    e = set(tokens(expected))
    a = set(tokens(answer))
    return len(e & a) / max(1, len(e))


def reciprocal_rank(relevant_ids: list[str], retrieved_ids: list[str]) -> float:
    rel = set(relevant_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in rel:
            return 1.0 / rank
    return 0.0


def recall_at_k(relevant_ids: list[str], retrieved_ids: list[str]) -> float:
    rel = set(relevant_ids)
    if not rel:
        return 0.0
    return len(rel & set(retrieved_ids)) / len(rel)


def average_precision(relevant_ids: list[str], retrieved_ids: list[str]) -> float:
    rel = set(relevant_ids)
    if not rel:
        return 0.0
    hits = 0
    total = 0.0
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in rel:
            hits += 1
            total += hits / rank
    return total / len(rel)


def ndcg_at_k(relevant_ids: list[str], retrieved_ids: list[str]) -> float:
    rel = set(relevant_ids)
    gains = [1 if cid in rel else 0 for cid in retrieved_ids]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(gains, reverse=True)
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    return float(dcg / idcg) if idcg else 0.0


def question_retrieval_metrics(item: dict, result: dict) -> dict:
    retrieved = [x.chunk.chunk_id for x in result.get("contexts", [])]
    relevant = item.get("relevant_chunk_ids", [])
    if not relevant:
        return {"recall_at_k": np.nan, "mrr": np.nan, "map": np.nan, "ndcg_at_k": np.nan}
    return {
        "recall_at_k": recall_at_k(relevant, retrieved),
        "mrr": reciprocal_rank(relevant, retrieved),
        "map": average_precision(relevant, retrieved),
        "ndcg_at_k": ndcg_at_k(relevant, retrieved),
    }


def benchmark_success(item: dict, row: dict) -> bool:
    """Task-level success gate used only for recovery/degradation analysis."""
    if not item.get("answerable", True):
        answer = row.get("answer", "").lower()
        abstention_markers = [
            "i don't have enough information",
            "not enough information",
            "cannot answer from the provided documents",
            "do not contain enough information",
        ]
        return any(marker in answer for marker in abstention_markers)
    return bool(
        row.get("answer_token_f1", 0.0) >= 0.70
        and row.get("grounded_score", 0.0) >= 0.43
        and row.get("citation_supported", False)
    )


def first_attempt_trace(result: dict) -> list[dict]:
    trace = result.get("trace", [])
    # First CRITIC closes the first attempt. Everything before it is attempt 0.
    out = []
    for event in trace:
        out.append(event)
        if event.get("node") == "CRITIC":
            break
    return out


def evaluate_answer(result: dict) -> dict:
    critic = result.get("critic", {})
    return {
        "grounded": bool(critic.get("grounded", False)),
        "relevant": bool(critic.get("relevant", False)),
        "complete": bool(critic.get("complete", False)),
        "citation_valid": bool(critic.get("citation_valid", False)),
        "citation_supported": bool(critic.get("citation_supported", False)),
        "grounded_score": float(critic.get("grounded_score", 0.0)),
        "relevance_score": float(critic.get("relevance_score", 0.0)),
        "confidence": float(critic.get("confidence", 0.0)),
        "supported_claim_rate": float(critic.get("supported_claim_rate", 0.0)),
    }


def run_benchmark() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    chunks = load_document_bytes(DOC_FILE.name, DOC_FILE.read_bytes())
    embedder = SentenceTransformer(EMBED_MODEL)
    llm = LocalLLM(GEN_MODEL)
    retriever = HybridRetriever(chunks, embedder)
    graph = build_graph(retriever, llm, embedder, top_k=5)
    items = json.loads(EVAL_FILE.read_text(encoding="utf-8"))

    rows = []
    for idx, item in enumerate(items, start=1):
        print(f"\n[{idx}/{len(items)}] {item['question']}")
        for mode in ["Standard RAG", "Self-Healing RAG"]:
            started = time.perf_counter()
            result = run_rag(graph, item["question"], max_retries=2, mode=mode)
            latency = time.perf_counter() - started

            answer = result.get("answer", "")
            retrieved_text = " ".join(x.chunk.text for x in result.get("contexts", []))
            retrieved_ids = [x.chunk.chunk_id for x in result.get("contexts", [])]
            final_metrics = evaluate_answer(result)
            first_trace = first_attempt_trace(result)
            first_critic = next((x for x in first_trace if x.get("node") == "CRITIC"), {})
            first_action = first_critic.get("action", "")
            retries = sum(e.get("node") == "QUERY_REWRITER" for e in result.get("trace", []))

            qmetrics = question_retrieval_metrics(item, result)
            row = {
                "mode": mode,
                "question_id": item.get("id", idx),
                "category": item.get("category", "unspecified"),
                "answerable": bool(item.get("answerable", True)),
                "difficulty": item.get("difficulty", "unspecified"),
                "question": item["question"],
                "expected_answer": item.get("answer", ""),
                "answer": answer,
                "retrieved_chunk_ids": "|".join(retrieved_ids),
                **qmetrics,
                "keyword_recall": keyword_recall(item.get("keywords", []), retrieved_text),
                "answer_keyword_accuracy": answer_contains_keywords(item, answer),
                "expected_answer_overlap": expected_overlap(item.get("answer", ""), answer),
                "answer_token_f1": token_f1(item.get("answer", ""), answer),
                "answer_exact_match": exact_match(item.get("answer", ""), answer),
                **final_metrics,
                "first_attempt_action": first_action,
                "first_attempt_failed": first_action != "PASS",
                "retries": retries,
                "latency_s": latency,
                "final_action": result.get("critic", {}).get("action", ""),
                "failure_type": result.get("critic", {}).get("failure_type"),
                "trace_length": len(result.get("trace", [])),
            }
            row["benchmark_success"] = benchmark_success(item, row)
            rows.append(row)
            print(f"  {mode}: accuracy={row['answer_token_f1']:.3f} grounded={row['grounded_score']:.3f} citations={row['citation_supported']} retries={retries} latency={latency:.2f}s")

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "evaluation_results.csv", index=False)

    metric_cols = [
        "recall_at_k", "mrr", "map", "ndcg_at_k", "keyword_recall",
        "answer_keyword_accuracy", "expected_answer_overlap", "answer_token_f1",
        "answer_exact_match", "grounded", "relevant", "complete",
        "citation_valid", "citation_supported", "grounded_score",
        "relevance_score", "confidence", "supported_claim_rate",
        "latency_s", "retries", "first_attempt_failed", "benchmark_success",
    ]
    summary = df.groupby("mode")[metric_cols].mean()
    summary.to_csv(ROOT / "metrics_summary.csv")

    # Category-level results show where healing helps rather than hiding everything in one mean.
    category_summary = (
        df.groupby(["category", "mode"])[
            ["answer_token_f1", "grounded_score", "supported_claim_rate", "confidence", "latency_s", "retries"]
        ].mean().reset_index()
    )
    category_summary.to_csv(ROOT / "category_metrics.csv", index=False)

    # Pairwise healing analysis.
    std = df[df["mode"] == "Standard RAG"].set_index("question_id")
    sh = df[df["mode"] == "Self-Healing RAG"].set_index("question_id")
    pairs = []
    for qid in std.index.intersection(sh.index):
        s = std.loc[qid]
        h = sh.loc[qid]
        standard_quality = float(s["answer_token_f1"])
        healing_quality = float(h["answer_token_f1"])
        std_success = bool(s["benchmark_success"])
        healing_success = bool(h["benchmark_success"])
        pairs.append({
            "question_id": qid,
            "question": s["question"],
            "standard_f1": standard_quality,
            "healing_f1": healing_quality,
            "f1_delta": healing_quality - standard_quality,
            "standard_grounded": bool(s["grounded"]),
            "healing_grounded": bool(h["grounded"]),
            "standard_citation_supported": bool(s["citation_supported"]),
            "healing_citation_supported": bool(h["citation_supported"]),
            "standard_success": std_success,
            "healing_success": healing_success,
            "healing_retried": int(h["retries"]) > 0,
            "healing_recovered": (not std_success) and healing_success and int(h["retries"]) > 0,
            "healing_degraded": std_success and (not healing_success) and int(h["retries"]) > 0,
            "latency_delta_s": float(h["latency_s"] - s["latency_s"]),
        })
    pairs_df = pd.DataFrame(pairs)
    pairs_df.to_csv(ROOT / "healing_analysis.csv", index=False)

    initial_failures = int(((~pairs_df["standard_success"]) if len(pairs_df) else pd.Series(dtype=bool)).sum())
    recovered = int(pairs_df["healing_recovered"].sum()) if len(pairs_df) else 0
    attempted_heal = int(pairs_df["healing_retried"].sum()) if len(pairs_df) else 0
    degraded = int(pairs_df["healing_degraded"].sum()) if len(pairs_df) else 0

    healing_summary = pd.DataFrame([{
        "questions": len(pairs_df),
        "standard_failures": initial_failures,
        "recovered_after_healing": recovered,
        "recovery_rate": recovered / initial_failures if initial_failures else 0.0,
        "healing_attempts": attempted_heal,
        "healing_degradations": degraded,
        "degradation_rate_among_healed": degraded / attempted_heal if attempted_heal else 0.0,
        "mean_f1_delta": float(pairs_df["f1_delta"].mean()) if len(pairs_df) else 0.0,
        "mean_latency_delta_s": float(pairs_df["latency_delta_s"].mean()) if len(pairs_df) else 0.0,
    }])
    healing_summary.to_csv(ROOT / "healing_summary.csv", index=False)

    print("\n========== OVERALL METRICS ==========")
    print(summary.round(3).to_string())
    print("\n========== HEALING ANALYSIS ==========")
    print(healing_summary.round(3).to_string(index=False))
    print("\n========== CATEGORY METRICS ==========")
    print(category_summary.round(3).to_string(index=False))
    print("\nSaved:")
    print("  evaluation_results.csv")
    print("  metrics_summary.csv")
    print("  category_metrics.csv")
    print("  healing_analysis.csv")
    print("  healing_summary.csv")
    return df, summary, healing_summary


if __name__ == "__main__":
    run_benchmark()
