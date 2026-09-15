from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, TypedDict

import numpy as np
import torch
from langgraph.graph import END, START, StateGraph
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("self_healing_rag")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_CONTEXT_CHARS = 7000


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
    citation_supported: bool
    grounded_score: float
    relevance_score: float
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


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z0-9][a-zA-Z0-9_-]*\b", text.lower())


def split_text(text: str, chunk_size: int = 180, overlap: int = 35) -> List[str]:
    """Word-based chunking. Small chunks improve citation-level attribution."""
    text = normalize_text(text)
    if not text:
        return []
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(start + 1, end - overlap)
    return chunks


def load_document_bytes(name: str, data: bytes) -> List[Chunk]:
    suffix = name.lower().split(".")[-1]
    chunks: List[Chunk] = []

    if suffix == "txt":
        text = data.decode("utf-8", errors="ignore")
        for i, part in enumerate(split_text(text)):
            chunks.append(Chunk(f"{name}-c{i}", part, name, None))

    elif suffix == "pdf":
        from io import BytesIO
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(data))
        for page_no, page in enumerate(reader.pages, start=1):
            text = normalize_text(page.extract_text() or "")
            for i, part in enumerate(split_text(text)):
                chunks.append(Chunk(f"{name}-p{page_no}-c{i}", part, name, page_no))

    elif suffix == "docx":
        from io import BytesIO
        from docx import Document

        doc = Document(BytesIO(data))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        for i, part in enumerate(split_text(text)):
            chunks.append(Chunk(f"{name}-c{i}", part, name, None))
    else:
        raise ValueError(f"Unsupported file type: .{suffix}")

    logger.info("Parsed %s -> %d chunks", name, len(chunks))
    return chunks


class HybridRetriever:
    """Dense + BM25 retrieval followed by Reciprocal Rank Fusion."""

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

    def search(self, query: str, top_k: int = 5, dense_k: int = 12, bm25_k: int = 12) -> List[RetrievedChunk]:
        qvec = self.embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0].astype("float32")
        dense_scores = self.embeddings @ qvec
        dense_order = np.argsort(-dense_scores)[: min(dense_k, len(self.chunks))]

        bm25_scores = np.asarray(self.bm25.get_scores(tokenize(query)))
        bm25_order = np.argsort(-bm25_scores)[: min(bm25_k, len(self.chunks))]

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
        logger.info("Retrieved query=%r top_k=%d", query, top_k)
        return results


class LocalLLM:
    """Small fully local instruction model. No API key is required."""

    def __init__(self, model_name: str = GEN_MODEL):
        logger.info("Loading generation model: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        logger.info("Generation model ready on %s", self.device)

    def generate(self, system: str, user: str, max_new_tokens: int = 180) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def format_contexts(contexts: List[RetrievedChunk]) -> str:
    blocks = []
    total = 0
    for i, item in enumerate(contexts, start=1):
        block = f"[{i}] {item.chunk.text}"
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)


def answer_prompt(question: str, contexts: List[RetrievedChunk]) -> str:
    return f"""Answer the question using ONLY the evidence.

Strict rules:
- Do not use outside knowledge.
- If the evidence does not contain enough information, say: I don't have enough information in the provided documents.
- Give a concise answer, usually 1-3 sentences.
- Every factual sentence MUST end with one or more citations such as [1] or [2].
- Citation numbers MUST refer only to the evidence blocks below.
- Never invent a citation number.

Question:
{question}

Evidence:
{format_contexts(contexts)}

Answer:"""


def rewrite_prompt(question: str, previous_answer: str, critic_reason: str, old_query: str) -> str:
    return f"""Create a better retrieval query. Return ONLY the query, no explanation.

Original question: {question}
Previous search query: {old_query}
Previous answer: {previous_answer}
Critic feedback: {critic_reason}

Use important entities, exact terms, numbers, and concepts from the question. Try a different wording that can retrieve missing evidence.
"""


def sentence_parts(answer: str) -> List[str]:
    answer = re.sub(r"\s+", " ", answer).strip()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]


def strip_citations(text: str) -> str:
    return re.sub(r"\s*\[(\d+)\]", "", text).strip()


def lexical_support(sentence: str, context: str) -> float:
    s = set(tokenize(strip_citations(sentence)))
    c = set(tokenize(context))
    if not s:
        return 0.0
    # Ignore common function words so factual overlap matters more.
    stop = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "in", "on", "for", "with", "what", "how", "when", "where", "does", "do", "did"}
    s -= stop
    return len(s & c) / max(1, len(s))


def support_matrix(answer_sentences: List[str], contexts: List[RetrievedChunk], embedder: SentenceTransformer) -> np.ndarray:
    if not answer_sentences or not contexts:
        return np.zeros((len(answer_sentences), len(contexts)))
    sentences = [strip_citations(s) for s in answer_sentences]
    ctx = [r.chunk.text for r in contexts]
    vecs = embedder.encode(sentences + ctx, normalize_embeddings=True, show_progress_bar=False)
    a = vecs[:len(sentences)]
    c = vecs[len(sentences):]
    semantic = a @ c.T
    lexical = np.array([[lexical_support(s, x) for x in ctx] for s in sentences])
    return 0.65 * semantic + 0.35 * lexical


def deterministic_critic(question: str, answer: str, contexts: List[RetrievedChunk], embedder: SentenceTransformer) -> CriticResult:
    if not answer.strip() or "I don't have enough information" in answer:
        return CriticResult(False, False, False, False, False, 0.0, 0.0, 0.0, "The model did not produce a supported answer.", "RETRIEVE_MORE")

    sentences = sentence_parts(answer)
    matrix = support_matrix(sentences, contexts, embedder)
    per_sentence = matrix.max(axis=1) if matrix.size else np.zeros(len(sentences))
    grounded_score = float(np.mean(per_sentence)) if len(per_sentence) else 0.0
    grounded = bool(np.all(per_sentence >= 0.43))

    qvec = embedder.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]
    avec = embedder.encode([strip_citations(answer)], normalize_embeddings=True, show_progress_bar=False)[0]
    relevance_score = float(qvec @ avec)
    relevant = relevance_score >= 0.38

    citations = [int(x) for x in re.findall(r"\[(\d+)\]", answer)]
    citation_valid = bool(citations) and all(1 <= x <= len(contexts) for x in citations)

    # A citation is semantically supported when the cited chunk supports the sentence it appears in.
    citation_supported_flags: List[bool] = []
    for i, sentence in enumerate(sentences):
        cited = [int(x) for x in re.findall(r"\[(\d+)\]", sentence)]
        if not cited:
            citation_supported_flags.append(False)
            continue
        ok = all(1 <= x <= len(contexts) and matrix[i, x - 1] >= 0.43 for x in cited)
        citation_supported_flags.append(ok)
    citation_supported = bool(citation_supported_flags) and all(citation_supported_flags)

    complete = len(strip_citations(answer).split()) >= 4
    confidence = float(np.clip(
        0.45 * grounded_score + 0.30 * max(0.0, relevance_score) + 0.15 * (1.0 if citation_supported else 0.0) + 0.10 * (1.0 if complete else 0.0),
        0, 1,
    ))

    if not grounded:
        action, reason = "RETRIEVE_MORE", "At least one answer claim is weakly supported by the retrieved evidence."
    elif not relevant:
        action, reason = "RETRIEVE_MORE", "The answer is weakly aligned with the user's question."
    elif not citation_valid or not citation_supported:
        action, reason = "REGENERATE", "Citations are missing, invalid, or not supported by the cited evidence."
    else:
        action, reason = "PASS", "Answer is grounded, relevant, complete enough, and supported by citations."

    return CriticResult(grounded, relevant, complete, citation_valid, citation_supported, grounded_score, relevance_score, confidence, reason, action)


def build_graph(retriever: HybridRetriever, llm: LocalLLM, embedder: SentenceTransformer, top_k: int = 5):
    def add_trace(state: GraphState, node: str, message: str, **extra):
        trace = list(state.get("trace", []))
        trace.append({"time": time.strftime("%H:%M:%S"), "node": node, "message": message, **extra})
        return trace

    def retrieve_node(state: GraphState):
        q = state.get("search_query") or state["question"]
        results = retriever.search(q, top_k=top_k)
        return {"contexts": results, "trace": add_trace(state, "RETRIEVER", f"Retrieved {len(results)} chunks using dense + BM25 + RRF.", query=q)}

    def generate_node(state: GraphState):
        started = time.perf_counter()
        answer = llm.generate("You are a precise evidence-grounded assistant. Follow the citation rules exactly.", answer_prompt(state["question"], state["contexts"]))
        latency = (time.perf_counter() - started) * 1000
        return {"answer": answer, "trace": add_trace(state, "GENERATOR", "Generated a candidate answer with evidence citations.", latency_ms=round(latency, 1))}

    def critic_node(state: GraphState):
        started = time.perf_counter()
        result = deterministic_critic(state["question"], state["answer"], state["contexts"], embedder)
        latency = (time.perf_counter() - started) * 1000
        return {"critic": result.__dict__, "trace": add_trace(state, "CRITIC", result.reason, action=result.action, confidence=round(result.confidence, 3), grounded_score=round(result.grounded_score, 3), relevance_score=round(result.relevance_score, 3), citation_supported=result.citation_supported, latency_ms=round(latency, 1))}

    def rewrite_node(state: GraphState):
        started = time.perf_counter()
        old_query = state.get("search_query") or state["question"]
        rewritten = llm.generate(
            "You are a retrieval query rewriting component. Output only one search query.",
            rewrite_prompt(state["question"], state.get("answer", ""), state["critic"]["reason"], old_query),
            max_new_tokens=60,
        )
        rewritten = normalize_text(rewritten).strip('"').strip()
        if not rewritten or rewritten.lower() == old_query.lower():
            # Deterministic fallback prevents a useless retry.
            rewritten = f"{state['question']} evidence policy requirements details"
        return {
            "search_query": rewritten,
            "attempt": state.get("attempt", 0) + 1,
            "trace": add_trace(state, "QUERY_REWRITER", f"Reformulated query: {rewritten}", latency_ms=round((time.perf_counter() - started) * 1000, 1)),
        }

    def route(state: GraphState):
        if state.get("critic", {}).get("action") == "PASS":
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
    workflow.add_conditional_edges("critic", route, {"retry": "rewrite", "final": END})
    workflow.add_edge("rewrite", "retrieve")
    return workflow.compile()


def run_rag(graph, question: str, max_retries: int = 2, mode: str = "Self-Healing RAG"):
    state: GraphState = {
        "question": question,
        "search_query": question,
        "attempt": 0,
        "max_retries": max_retries if mode == "Self-Healing RAG" else 0,
        "mode": mode,
        "trace": [],
    }
    return graph.invoke(state)
