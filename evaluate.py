from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from sentence_transformers import SentenceTransformer
from rag_engine import EMBED_MODEL, GEN_MODEL, HybridRetriever, LocalLLM, build_graph, load_document_bytes, run_rag

DATA_DIR = Path(__file__).parent / "data"
EVAL_FILE = DATA_DIR / "evaluation.json"
DOC_FILE = DATA_DIR / "sample_company_handbook.txt"


def keyword_score(keywords, text):
    text = text.lower()
    if not keywords:
        return 0.0
    return sum(k.lower() in text for k in keywords) / len(keywords)


def expected_answer_score(expected, answer):
    """Soft semantic score for the benchmark; keyword score remains transparent."""
    return keyword_score([w for w in expected.lower().split() if len(w) >= 4], answer)


def run_benchmark():
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
            critic = result.get("critic", {})
            contexts = result.get("contexts", [])
            retrieved_text = " ".join(x.chunk.text for x in contexts)
            retries = sum(e.get("node") == "QUERY_REWRITER" for e in result.get("trace", []))

            row = {
                "mode": mode,
                "question": item["question"],
                "expected_answer": item.get("answer", ""),
                "answer": result.get("answer", ""),
                "retrieval_keyword_recall": round(keyword_score(item["keywords"], retrieved_text), 3),
                "answer_keyword_accuracy": round(keyword_score(item["keywords"], result.get("answer", "")), 3),
                "expected_answer_overlap": round(expected_answer_score(item.get("answer", ""), result.get("answer", "")), 3),
                "grounded": bool(critic.get("grounded", False)),
                "citation_valid": bool(critic.get("citation_valid", False)),
                "citation_supported": bool(critic.get("citation_supported", False)),
                "grounded_score": round(float(critic.get("grounded_score", 0.0)), 3),
                "relevance_score": round(float(critic.get("relevance_score", 0.0)), 3),
                "confidence": round(float(critic.get("confidence", 0.0)), 3),
                "latency_s": round(latency, 3),
                "retries": retries,
                "critic_action": critic.get("action", ""),
                "critic_reason": critic.get("reason", ""),
            }
            rows.append(row)
            print(f"  {mode}: grounded={row['grounded']} citation={row['citation_supported']} retries={retries} latency={row['latency_s']}s")

    df = pd.DataFrame(rows)
    df.to_csv("evaluation_results.csv", index=False)

    summary = (
        df.groupby("mode")
        .agg(
            retrieval_recall=("retrieval_keyword_recall", "mean"),
            answer_accuracy=("answer_keyword_accuracy", "mean"),
            expected_answer_overlap=("expected_answer_overlap", "mean"),
            grounded_rate=("grounded", "mean"),
            citation_valid_rate=("citation_valid", "mean"),
            citation_supported_rate=("citation_supported", "mean"),
            grounded_score=("grounded_score", "mean"),
            relevance_score=("relevance_score", "mean"),
            confidence=("confidence", "mean"),
            latency_s=("latency_s", "mean"),
            avg_retries=("retries", "mean"),
        )
    )

    print("\n========== SUMMARY ==========")
    print(summary.round(3).to_string())

    # Recovery is computed only on questions that failed the first-pass self-healing attempt.
    sh = df[df.mode == "Self-Healing RAG"].set_index("question")
    std = df[df.mode == "Standard RAG"].set_index("question")
    initial_fail = sh[sh["critic_action"] != "PASS"]
    # A query is considered recovered when its final answer is accepted and it actually retried.
    recovered = initial_fail[(initial_fail["retries"] > 0) & (initial_fail["critic_action"] == "PASS")]
    # The previous line cannot capture final PASS because critic_action is final state; use trace-based signal below.
    recovered_count = 0
    initial_fail_count = 0
    for q in sh.index:
        row = sh.loc[q]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        retries = int(row["retries"])
        final_pass = bool(row["grounded"] and row["citation_supported"])
        std_row = std.loc[q]
        if isinstance(std_row, pd.DataFrame):
            std_row = std_row.iloc[0]
        standard_failed = not bool(std_row["grounded"] and std_row["citation_supported"])
        if standard_failed:
            initial_fail_count += 1
            if retries > 0 and final_pass:
                recovered_count += 1

    recovery_rate = recovered_count / initial_fail_count if initial_fail_count else 0.0
    print(f"\nStandard-RAG failures: {initial_fail_count}")
    print(f"Recovered by Self-Healing: {recovered_count}")
    print(f"Self-Healing recovery rate: {recovery_rate:.1%}")

    pd.DataFrame([{
        "standard_failures": initial_fail_count,
        "recovered_after_retry": recovered_count,
        "recovery_rate": round(recovery_rate, 3),
    }]).to_csv("healing_summary.csv", index=False)

    print("\nSaved: evaluation_results.csv")
    print("Saved: healing_summary.csv")
    return df, summary


if __name__ == "__main__":
    run_benchmark()
