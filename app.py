from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from rag_engine import EMBED_MODEL, GEN_MODEL, HybridRetriever, LocalLLM, SentenceTransformer, build_graph, load_document_bytes, run_rag

st.set_page_config(page_title="Self-Healing RAG", page_icon="↻", layout="wide")

st.title("Self-Healing RAG")
st.caption("LangGraph + hybrid retrieval + evidence critique + query reformulation")


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource(show_spinner="Loading local generation model...")
def get_llm():
    return LocalLLM(GEN_MODEL)


def build_system(chunks):
    embedder = get_embedder()
    llm = get_llm()
    retriever = HybridRetriever(chunks, embedder)
    return build_graph(retriever, llm, embedder, top_k=5)


def index_documents(chunks, names):
    st.session_state.chunks = chunks
    st.session_state.documents = names
    st.session_state.graph = build_system(chunks)


for key, default in {
    "chunks": [],
    "documents": [],
    "graph": None,
    "result": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.header("Knowledge Base")

    if st.button("Load demo knowledge base", use_container_width=True):
        p = Path("data/sample_company_handbook.txt")
        chunks = load_document_bytes(p.name, p.read_bytes())
        with st.spinner("Indexing demo documents..."):
            index_documents(chunks, [p.name])
        st.success(f"Indexed {len(chunks)} chunks")

    uploads = st.file_uploader("Upload TXT / PDF / DOCX", type=["txt", "pdf", "docx"], accept_multiple_files=True)
    if st.button("Build knowledge base", use_container_width=True):
        if not uploads:
            st.warning("Upload at least one document.")
        else:
            chunks = []
            names = []
            for upload in uploads:
                chunks.extend(load_document_bytes(upload.name, upload.getvalue()))
                names.append(upload.name)
            with st.spinner("Building hybrid index..."):
                index_documents(chunks, names)
            st.success(f"Indexed {len(chunks)} chunks from {len(names)} documents")

    st.divider()
    mode = st.radio("RAG mode", ["Self-Healing RAG", "Standard RAG"], index=0)
    max_retries = st.slider("Maximum healing retries", 0, 3, 2)
    st.caption(f"Embedding: `{EMBED_MODEL}`")
    st.caption(f"Generator: `{GEN_MODEL}`")

    if st.session_state.documents:
        st.caption("Indexed documents")
        for name in st.session_state.documents:
            st.write(f"• {name}")

if not st.session_state.graph:
    st.info("Load the demo knowledge base or upload your own documents to begin.")
    st.stop()

chat, trace_tab, eval_tab, architecture = st.tabs(["Chat", "Execution Trace", "Evaluation", "Architecture"])

with chat:
    question = st.text_input("Ask a question", placeholder="Example: What must happen before a critical service is deployed?")

    if st.button("Run RAG", type="primary", disabled=not question):
        started = time.perf_counter()
        with st.spinner("Running LangGraph workflow..."):
            result = run_rag(st.session_state.graph, question, max_retries=max_retries, mode=mode)
        result["total_latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        st.session_state.result = result

    result = st.session_state.result
    if result:
        critic = result.get("critic", {})
        st.subheader("Answer")
        st.write(result.get("answer", ""))

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Grounded", "PASS" if critic.get("grounded") else "FAIL")
        c2.metric("Citations", "PASS" if critic.get("citation_supported") else "FAIL")
        c3.metric("Confidence", f"{critic.get('confidence', 0):.2f}")
        c4.metric("Retries", sum(e.get("node") == "QUERY_REWRITER" for e in result.get("trace", [])))
        c5.metric("Latency", f"{result.get('total_latency_ms', 0):.0f} ms")

        st.divider()
        st.subheader("Evidence")
        for i, item in enumerate(result.get("contexts", []), start=1):
            chunk = item.chunk
            title = f"[{i}] {chunk.source}" + (f" — page {chunk.page}" if chunk.page else "")
            with st.expander(f"{title} | dense={item.dense_score:.3f} | BM25={item.bm25_score:.2f} | RRF={item.rrf_score:.4f}"):
                st.write(chunk.text)

        st.info(critic.get("reason", "No critic result."))

with trace_tab:
    st.subheader("LangGraph execution trace")
    result = st.session_state.result
    if not result:
        st.info("Run a question first.")
    else:
        for i, event in enumerate(result.get("trace", []), 1):
            with st.container(border=True):
                st.markdown(f"**{i}. {event.get('node')}**")
                st.write(event.get("message", ""))
                extras = {k: v for k, v in event.items() if k not in {"time", "node", "message"}}
                if extras:
                    st.json(extras)

with eval_tab:
    st.subheader("Standard RAG vs Self-Healing RAG")
    st.write("The benchmark compares first-pass RAG with the cyclic system. Results should be reported only after running the benchmark on a sufficiently large, manually verified evaluation set.")
    eval_path = Path("data/evaluation.json")
    if eval_path.exists() and st.button("Run evaluation"):
        items = json.loads(eval_path.read_text(encoding="utf-8"))
        rows = []
        progress = st.progress(0)
        for i, item in enumerate(items):
            for eval_mode in ["Standard RAG", "Self-Healing RAG"]:
                started = time.perf_counter()
                result = run_rag(st.session_state.graph, item["question"], max_retries=2, mode=eval_mode)
                elapsed = time.perf_counter() - started
                critic = result.get("critic", {})
                retrieved_text = " ".join(x.chunk.text for x in result.get("contexts", []))
                answer = result.get("answer", "")
                rows.append({
                    "mode": eval_mode,
                    "question": item["question"],
                    "retrieval_keyword_recall": round(sum(k.lower() in retrieved_text.lower() for k in item["keywords"]) / max(1, len(item["keywords"])), 3),
                    "answer_keyword_accuracy": round(sum(k.lower() in answer.lower() for k in item["keywords"]) / max(1, len(item["keywords"])), 3),
                    "grounded": bool(critic.get("grounded")),
                    "citation_valid": bool(critic.get("citation_valid")),
                    "citation_supported": bool(critic.get("citation_supported")),
                    "confidence": round(float(critic.get("confidence", 0)), 3),
                    "latency_s": round(elapsed, 3),
                    "retries": sum(e.get("node") == "QUERY_REWRITER" for e in result.get("trace", [])),
                })
            progress.progress((i + 1) / len(items))

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        summary = df.groupby("mode").agg(
            retrieval_recall=("retrieval_keyword_recall", "mean"),
            answer_accuracy=("answer_keyword_accuracy", "mean"),
            grounded_rate=("grounded", "mean"),
            citation_valid_rate=("citation_valid", "mean"),
            citation_supported_rate=("citation_supported", "mean"),
            confidence=("confidence", "mean"),
            latency_s=("latency_s", "mean"),
            avg_retries=("retries", "mean"),
        ).reset_index()
        st.subheader("Summary")
        st.dataframe(summary.round(3), use_container_width=True)
        st.download_button("Download evaluation_results.csv", df.to_csv(index=False), "evaluation_results.csv", "text/csv")

with architecture:
    st.subheader("Workflow")
    st.code("""START\n  ↓\nHybrid Retrieval\n(Dense + BM25 → RRF)\n  ↓\nGenerate Answer + Citations\n  ↓\nEvidence Critic\n  ├── PASS → FINAL\n  └── FAIL\n        ↓\n   Query Rewriter\n        ↓\n   Re-retrieve\n        ↓\n   Generate → Critic\n        ↓\n   PASS or max retries\n""", language="text")
    st.markdown("### What makes it self-healing")
    st.write("The system does not blindly accept the first generation. It evaluates claim-level evidence support and citation support. A failed answer triggers a new retrieval query, while a retry limit provides a safe stopping condition.")
