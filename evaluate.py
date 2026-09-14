import json
import time
from pathlib import Path

import pandas as pd

from rag_engine import (
    HybridRetriever,
    LocalLLM,
    build_graph,
    load_document_bytes,
    run_rag,
)
from sentence_transformers import SentenceTransformer


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    chunks = []
    for path in Path("data").glob("*.txt"):
        chunks.extend(load_document_bytes(path.name, path.read_bytes()))

    embedder = SentenceTransformer(EMBED_MODEL)
    retriever = HybridRetriever(chunks, embedder)
    llm = LocalLLM(GEN_MODEL)
    graph = build_graph(retriever, llm, embedder)

    items = json.loads(Path("data/evaluation.json").read_text())
    rows = []

    for item in items:
        for mode in ["Standard RAG", "Self-Healing RAG"]:
            start = time.perf_counter()
            result = run_rag(graph, item["question"], 2, mode)
            elapsed = time.perf_counter() - start

            context = " ".join(
                x.chunk.text.lower() for x in result["contexts"]
            )
            answer = result["answer"].lower()
            keys = item["keywords"]

            retrieval = sum(k.lower() in context for k in keys) / len(keys)
            answer_acc = sum(k.lower() in answer for k in keys) / len(keys)

            rows.append(
                {
                    "mode": mode,
                    "question": item["question"],
                    "retrieval_keyword_recall": retrieval,
                    "answer_keyword_accuracy": answer_acc,
                    "grounded": result["critic"]["grounded"],
                    "citation_valid": result["critic"]["citation_valid"],
                    "confidence": result["critic"]["confidence"],
                    "latency_s": elapsed,
                    "retries": sum(
                        e["node"] == "QUERY_REWRITER"
                        for e in result["trace"]
                    ),
                }
            )

    df = pd.DataFrame(rows)
    summary = df.groupby("mode").mean(numeric_only=True)
    print("\n========== SELF-HEALING RAG EVALUATION ==========")
    print(df.to_string(index=False))
    print("\n========== SUMMARY ==========")
    print(summary.to_string())
    df.to_csv("evaluation_results.csv", index=False)
    print("\nSaved evaluation_results.csv")


if __name__ == "__main__":
    main()
