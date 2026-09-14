from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, TypedDict

import numpy as np
import torch
from langgraph.graph import END, START, StateGraph
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("self_healing_rag")


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    page: int | None = None


@dataclass
class RetrievedChunk:
    chunk: Chunk
    dense_score: float
    bm25_score: float
    rrf_score: float


@dataclass
class CriticResult:
    grounded: bool
    relevant: bool
    complete: bool
    citation_valid: bool
    confidence: float
    reason: str
    action: str


class GraphState(TypedDict, total=False):
    question: str
    search_query: str
    answer: str
    contexts: List[RetrievedChunk]
    attempt: int
    max_retries: int
    critic: Dict[str, Any]
    mode: str
    trace: List[Dict[str, Any]]
    final: bool


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z0-9][a-zA-Z0-9_-]*\b", text.lower())


def split_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(start + 1, end - overlap)
    return chunks


def load_document_bytes(name: str, data: bytes) -> List[Chunk]:
    """Parse TXT, PDF, or DOCX bytes into chunks with source metadata."""
    suffix = name.lower().split(".")[-1]
    raw_chunks: List[Chunk] = []

    if suffix == "txt":
        text = data.decode("utf-8", errors="ignore")
        for i, part in enumerate(split_text(text)):
            raw_chunks.append(Chunk(f"{name}-c{i}", part, name, None))

    elif suffix == "pdf":
        from io import BytesIO
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(data))
        for page_no, page in enumerate(reader.pages, start=1):
            text = normalize_text(page.extract_text() or "")
            for i, part in enumerate(split_text(text)):
                raw_chunks.append(
                    Chunk(f"{name}-p{page_no}-c{i}", part, name, page_no)
                )

    elif suffix == "docx":
        from io import BytesIO
        from docx import Document

        doc = Document(BytesIO(data))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        for i, part in enumerate(split_text(text)):
            raw_chunks.append(Chunk(f"{name}-c{i}", part, name, None))
    else:
        raise ValueError(f"Unsupported file type: .{suffix}")

    logger.info("Parsed %s -> %d chunks", name, len(raw_chunks))
    return raw_chunks


class HybridRetriever:
    """Dense + BM25 retrieval with Reciprocal Rank Fusion."""

    def __init__(self, chunks: List[Chunk], embedding_model: SentenceTransformer):
        if not chunks:
            raise ValueError("No document chunks available.")
        self.chunks = chunks
        self.embedder = embedding_model
        self.texts = [c.text for c in chunks]
        self.tokens = [tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(self.tokens)
        self.embeddings = self.embedder.encode(
            self.texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        ).astype("float32")
        logger.info("Built retriever over %d chunks", len(chunks))

    def search(self, query: str, top_k: int = 5, dense_k: int = 12, bm25_k: int = 12):
        qvec = self.embedder.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0].astype("float32")

        dense_scores = self.embeddings @ qvec
        dense_order = np.argsort(-dense_scores)[: min(dense_k, len(self.chunks))]

        bm25_scores = np.asarray(self.bm25.get_scores(tokenize(query)))
        bm25_order = np.argsort(-bm25_scores)[: min(bm25_k, len(self.chunks))]

        # Reciprocal Rank Fusion.
        rrf: Dict[int, float] = {}
        for rank, idx in enumerate(dense_order, start=1):
            rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (60 + rank)
        for rank, idx in enumerate(bm25_order, start=1):
            rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (60 + rank)

        ordered = sorted(rrf, key=rrf.get, reverse=True)[:top_k]

        results = [
            RetrievedChunk(
                chunk=self.chunks[i],
                dense_score=float(dense_scores[i]),
                bm25_score=float(bm25_scores[i]),
                rrf_score=float(rrf[i]),
            )
            for i in ordered
        ]
        logger.info(
            "Retrieved query=%r top_k=%d best_rrf=%.4f",
            query,
            top_k,
            results[0].rrf_score if results else 0.0,
        )
        return results


class LocalLLM:
    """Small fully local open model; no API key required."""

    def __init__(self, model_name: str = GEN_MODEL):
        logger.info("Loading generation model: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        logger.info("Generation model ready on %s", self.device)

    def generate(self, system: str, user: str, max_new_tokens: int = 220) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return text


def answer_prompt(question: str, contexts: List[RetrievedChunk]) -> str:
    context_text = "\n\n".join(
        f"[{i+1}] {r.chunk.text}" for i, r in enumerate(contexts)
    )
    return f"""Answer the user's question using ONLY the evidence below.

Rules:
- If the evidence is insufficient, say exactly: "I don't have enough information in the provided documents."
- Do not invent facts.
- Cite supporting evidence using [1], [2], etc.
- Keep the answer concise and factual.

Question:
{question}

Evidence:
{context_text}
"""


def rewrite_prompt(question: str, previous_answer: str, critic_reason: str) -> str:
    return f"""Rewrite the search query to find stronger evidence for the question.

Original question:
{question}

Previous answer:
{previous_answer}

Critic feedback:
{critic_reason}

Return only one concise search query. Do not answer the question.
"""


def parse_json_object(text: str) -> Dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def lexical_support(sentence: str, context: str) -> float:
    s = set(tokenize(sentence))
    c = set(tokenize(context))
    if not s:
        return 0.0
    return len(s & c) / len(s)


def deterministic_critic(
    question: str,
    answer: str,
    contexts: List[RetrievedChunk],
    embedder: SentenceTransformer,
) -> CriticResult:
    if not answer.strip() or "I don't have enough information" in answer:
        return CriticResult(
            grounded=False,
            relevant=False,
            complete=False,
            citation_valid=False,
            confidence=0.0,
            reason="The model did not produce a supported answer.",
            action="RETRIEVE_MORE",
        )

    context_text = " ".join(r.chunk.text for r in contexts)
    answer_sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()
    ]
    if not answer_sentences:
        answer_sentences = [answer]

    sent_vecs = embedder.encode(
        answer_sentences + [context_text],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    context_vec = sent_vecs[-1]
    similarities = sent_vecs[:-1] @ context_vec
    lexical = [lexical_support(s, context_text) for s in answer_sentences]

    grounded_score = float(
        0.65 * np.mean(similarities) + 0.35 * np.mean(lexical)
    )
    grounded = grounded_score >= 0.48

    qvec = embedder.encode(
        [question], normalize_embeddings=True, show_progress_bar=False
    )[0]
    avec = embedder.encode(
        [answer], normalize_embeddings=True, show_progress_bar=False
    )[0]
    relevance_score = float(qvec @ avec)
    relevant = relevance_score >= 0.35

    citation_numbers = [int(x) for x in re.findall(r"\[(\d+)\]", answer)]
    citation_valid = bool(citation_numbers) and all(
        1 <= x <= len(contexts) for x in citation_numbers
    )

    complete = len(answer.split()) >= 4
    confidence = float(
        np.clip(
            0.45 * grounded_score
            + 0.35 * max(0.0, relevance_score)
            + 0.20 * (1.0 if citation_valid else 0.0),
            0,
            1,
        )
    )

    if not grounded:
        action, reason = "RETRIEVE_MORE", "Answer contains claims weakly supported by retrieved evidence."
    elif not relevant:
        action, reason = "RETRIEVE_MORE", "Answer appears weakly aligned with the question."
    elif not citation_valid:
        action, reason = "REGENERATE", "Answer is missing valid evidence citations."
    else:
        action, reason = "PASS", "Answer is sufficiently grounded, relevant, and cited."

    return CriticResult(
        grounded=grounded,
        relevant=relevant,
        complete=complete,
        citation_valid=citation_valid,
        confidence=confidence,
        reason=reason,
        action=action,
    )


def build_graph(retriever: HybridRetriever, llm: LocalLLM, embedder: SentenceTransformer, top_k: int = 5):
    def add_trace(state: GraphState, node: str, message: str, **extra):
        trace = list(state.get("trace", []))
        trace.append(
            {
                "time": time.strftime("%H:%M:%S"),
                "node": node,
                "message": message,
                **extra,
            }
        )
        return trace

    def retrieve_node(state: GraphState):
        q = state.get("search_query") or state["question"]
        results = retriever.search(q, top_k=top_k)
        return {
            "contexts": results,
            "trace": add_trace(
                state,
                "RETRIEVER",
                f"Retrieved {len(results)} chunks using hybrid dense + BM25 search.",
                query=q,
            ),
        }

    def generate_node(state: GraphState):
        start = time.perf_counter()
        answer = llm.generate(
            "You are a careful retrieval-grounded assistant.",
            answer_prompt(state["question"], state["contexts"]),
        )
        latency = time.perf_counter() - start
        return {
            "answer": answer,
            "trace": add_trace(
                state,
                "GENERATOR",
                "Generated an evidence-grounded candidate answer.",
                latency_ms=round(latency * 1000, 1),
            ),
        }

    def critic_node(state: GraphState):
        start = time.perf_counter()
        result = deterministic_critic(
            state["question"],
            state["answer"],
            state["contexts"],
            embedder,
        )
        latency = time.perf_counter() - start
        return {
            "critic": result.__dict__,
            "trace": add_trace(
                state,
                "CRITIC",
                result.reason,
                action=result.action,
                confidence=round(result.confidence, 3),
                latency_ms=round(latency * 1000, 1),
            ),
        }

    def rewrite_node(state: GraphState):
        start = time.perf_counter()
        rewritten = llm.generate(
            "You rewrite retrieval queries. Return only the improved search query.",
            rewrite_prompt(
                state["question"],
                state.get("answer", ""),
                state["critic"]["reason"],
            ),
            max_new_tokens=80,
        )
        rewritten = normalize_text(rewritten).strip('"')
        if not rewritten:
            rewritten = state["question"]
        return {
            "search_query": rewritten,
            "attempt": state.get("attempt", 0) + 1,
            "trace": add_trace(
                state,
                "QUERY_REWRITER",
                f"Reformulated query for retry: {rewritten}",
                latency_ms=round((time.perf_counter() - start) * 1000, 1),
            ),
        }

    def standard_route(state: GraphState):
        return "final"

    def self_healing_route(state: GraphState):
        critic = state.get("critic", {})
        if critic.get("action") == "PASS":
            return "final"
        if state.get("attempt", 0) < state.get("max_retries", 2):
            return "retry"
        return "final"

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("rewrite", rewrite_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "critic")

    workflow.add_conditional_edges(
        "critic",
        self_healing_route,
        {"retry": "rewrite", "final": END},
    )
    workflow.add_edge("rewrite", "retrieve")

    return workflow.compile()


def run_rag(
    graph,
    question: str,
    max_retries: int = 2,
    mode: str = "Self-Healing RAG",
):
    state: GraphState = {
        "question": question,
        "search_query": question,
        "attempt": 0,
        "max_retries": max_retries if mode == "Self-Healing RAG" else 0,
        "mode": mode,
        "trace": [],
    }
    result = graph.invoke(state)

    # For standard RAG, we still have a critic measurement, but it never causes retry.
    return result
