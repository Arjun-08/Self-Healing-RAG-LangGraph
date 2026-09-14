import io
import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from rag_engine import (
    EMBED_MODEL,
    GEN_MODEL,
    HybridRetriever,
    LocalLLM,
    build_graph,
    load_document_bytes,
    run_rag,
)


st.set_page_config(
    page_title="Self-Healing RAG",
    page_icon="↻",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource(show_spinner="Loading local generation model (~1 GB)...")
def get_llm():
    return LocalLLM(GEN_MODEL)


def init_state():
    defaults = {
        "chunks": [],
        "retriever": None,
        "graph": None,
        "results": None,
        "documents": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def build_index_from_chunks(chunks):
    embedder = get_embedder()
    st.session_state.chunks = chunks
    st.session_state.retriever = HybridRetriever(chunks, embedder)
    st.session_state.graph = build_graph(
        st.session_state.retriever,
        get_llm(),
        embedder,
        top_k=5,
    )


def process_sample_data():
    data_dir = Path("data")
    chunks = []
    for path in data_dir.glob("*.txt"):
        chunks.extend(load_document_bytes(path.name, path.read_bytes()))
    if chunks:
        build_index_from_chunks(chunks)
        st.session_state.documents = [p.name for p in data_dir.glob("*.txt")]


def process_uploads(uploaded_files):
    chunks = []
    names = []
    for uploaded in uploaded_files:
        chunks.extend(load_document_bytes(uploaded.name, uploaded.getvalue()))
        names.append(uploaded.name)
    if chunks:
        build_index_from_chunks(chunks)
        st.session_state.documents = names


init_state()

st.title("Self-Healing RAG")
st.caption(
    "A stateful LangGraph RAG system that retrieves, generates, critiques, "
    "reformulates, and retries when evidence is insufficient."
)

with st.sidebar:
    st.header("Knowledge Base")

    if st.button("Load demo knowledge base", use_container_width=True):
        with st.spinner("Indexing demo documents..."):
            process_sample_data()
        st.success(f"Indexed {len(st.session_state.chunks)} chunks.")

    uploaded = st.file_uploader(
        "Upload TXT / PDF / DOCX",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
    )

    if st.button("Build knowledge base", use_container_width=True):
        if not uploaded:
            st.warning("Upload at least one document.")
        else:
            with st.spinner("Parsing and indexing documents..."):
                process_uploads(uploaded)
            st.success(f"Indexed {len(st.session_state.chunks)} chunks.")

    st.divider()

    mode = st.radio(
        "RAG mode",
        ["Self-Healing RAG", "Standard RAG"],
        help="Standard RAG generates once. Self-Healing RAG can reformulate and retry.",
    )
    max_retries = st.slider("Maximum healing retries", 0, 3, 2)
    st.caption("Models")
    st.code(f"Embedding: {EMBED_MODEL}\nLLM: {GEN_MODEL}", language="text")

    if st.session_state.documents:
        st.caption("Indexed documents")
        for name in st.session_state.documents:
            st.write(f"• {name}")

if not st.session_state.graph:
    st.info(
        "Start by loading the demo knowledge base or upload your own PDF/DOCX/TXT files."
    )
    st.stop()

tab_chat, tab_trace, tab_eval, tab_about = st.tabs(
    ["Chat", "Execution Trace", "Evaluation", "Architecture"]
)

with tab_chat:
    question = st.text_input(
        "Ask a question",
        placeholder="Example: What happens after a production incident?",
    )

    if st.button("Run RAG", type="primary", disabled=not question):
        start = time.perf_counter()
        with st.spinner("Running LangGraph workflow..."):
            result = run_rag(
                st.session_state.graph,
                question,
                max_retries=max_retries,
                mode=mode,
            )
        result["total_latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
        st.session_state.results = result

    result = st.session_state.results
    if result:
        st.subheader("Answer")
        st.write(result.get("answer", ""))

        critic = result.get("critic", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Grounded", "PASS" if critic.get("grounded") else "FAIL")
        c2.metric("Relevance", "PASS" if critic.get("relevant") else "FAIL")
        c3.metric("Confidence", f"{critic.get('confidence', 0):.2f}")
        c4.metric("Latency", f"{result.get('total_latency_ms', 0):.0f} ms")

        st.divider()
        st.subheader("Retrieved evidence")

        for i, item in enumerate(result.get("contexts", []), start=1):
            chunk = item.chunk
            title = f"[{i}] {chunk.source}"
            if chunk.page:
                title += f" — page {chunk.page}"
            with st.expander(
                f"{title} | dense={item.dense_score:.3f} | BM25={item.bm25_score:.2f} | RRF={item.rrf_score:.4f}"
            ):
                st.write(chunk.text)

        st.info(
            f"Critic: {critic.get('reason', 'No critic result.')}"
            + (f" | Action: {critic.get('action')}" if critic else "")
        )

with tab_trace:
    st.subheader("LangGraph execution trace")
    result = st.session_state.results
    if not result:
        st.info("Run a question first.")
    else:
        for i, event in enumerate(result.get("trace", []), start=1):
            node = event.get("node", "NODE")
            msg = event.get("message", "")
            with st.container(border=True):
                st.markdown(f"**{i}. {node}**")
                st.write(msg)
                extras = {
                    k: v
                    for k, v in event.items()
                    if k not in {"time", "node", "message"}
                }
                if extras:
                    st.json(extras)

with tab_eval:
    st.subheader("Evaluation")
    st.write(
        "This evaluation uses the included small demonstration set. "
        "For a serious portfolio result, expand this to 100–500 manually verified questions."
    )

    eval_path = Path("data/evaluation.json")
    if not eval_path.exists():
        st.warning("data/evaluation.json not found.")
    else:
        eval_items = json.loads(eval_path.read_text(encoding="utf-8"))

        if st.button("Run evaluation", type="primary"):
            rows = []
            progress = st.progress(0)
            for idx, item in enumerate(eval_items):
                for eval_mode in ["Standard RAG", "Self-Healing RAG"]:
                    started = time.perf_counter()
                    result = run_rag(
                        st.session_state.graph,
                        item["question"],
                        max_retries=2,
                        mode=eval_mode,
                    )
                    elapsed = time.perf_counter() - started

                    retrieved_text = " ".join(
                        x.chunk.text.lower() for x in result.get("contexts", [])
                    )
                    answer = result.get("answer", "").lower()
                    keywords = item["keywords"]
                    retrieval_hit = sum(
                        1 for k in keywords if k.lower() in retrieved_text
                    ) / max(1, len(keywords))
                    answer_hit = sum(
                        1 for k in keywords if k.lower() in answer
                    ) / max(1, len(keywords))

                    rows.append(
                        {
                            "mode": eval_mode,
                            "question": item["question"],
                            "retrieval_keyword_recall": round(retrieval_hit, 3),
                            "answer_keyword_accuracy": round(answer_hit, 3),
                            "grounded": bool(result.get("critic", {}).get("grounded")),
                            "citation_valid": bool(
                                result.get("critic", {}).get("citation_valid")
                            ),
                            "confidence": round(
                                result.get("critic", {}).get("confidence", 0), 3
                            ),
                            "latency_s": round(elapsed, 2),
                            "attempts": sum(
                                1
                                for e in result.get("trace", [])
                                if e.get("node") == "QUERY_REWRITER"
                            ),
                        }
                    )
                progress.progress((idx + 1) / len(eval_items))

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

            summary = (
                df.groupby("mode")
                .agg(
                    retrieval_recall=("retrieval_keyword_recall", "mean"),
                    answer_accuracy=("answer_keyword_accuracy", "mean"),
                    grounded_rate=("grounded", "mean"),
                    citation_rate=("citation_valid", "mean"),
                    confidence=("confidence", "mean"),
                    latency_s=("latency_s", "mean"),
                    avg_retries=("attempts", "mean"),
                )
                .reset_index()
            )
            st.subheader("Summary")
            st.dataframe(summary, use_container_width=True)

            st.download_button(
                "Download evaluation CSV",
                df.to_csv(index=False),
                file_name="rag_evaluation.csv",
                mime="text/csv",
            )

with tab_about:
    st.subheader("Architecture")
    st.code(
        """User Query
    |
    v
Hybrid Retrieval
(Dense embeddings + BM25)
    |
    v
LLM Generator
    |
    v
Critic
    |
    +---- PASS -----------------> Final Answer
    |
    +---- FAIL
            |
            v
       Query Rewriter
            |
            v
       Re-retrieve
            |
            +----> Generate -> Critic
                         |
                         +----> PASS
                         |
                         +----> max retries -> safe fallback
""",
        language="text",
    )

    st.markdown(
        """
### Why this is different from basic RAG

- Stateful cyclic workflow with LangGraph
- Hybrid dense + lexical retrieval
- Evidence citations
- Automatic critique
- Query reformulation and retry
- Standard-vs-self-healing evaluation
- Latency, confidence, retrieval, grounding and citation metrics
- Local open models; no paid API is required
"""
    )
