from __future__ import annotations

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
RRF_K = 60
DEFAULT_TOP_K = 5
MAX_CONTEXT_CHARS = 6500

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "and", "in", "on", "for", "with", "from", "by", "or", "as",
    "what", "when", "where", "who", "which", "how", "does", "do", "did",
    "can", "may", "must", "should", "their", "they", "them", "this", "that",
    "company", "employee", "employees", "policy", "according", "following",
}


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
    supported_claim_rate: float
    reason: str
    action: str
    failure_type: str | None


class GraphState(TypedDict, total=False):
    question: str
    search_query: str
    answer: str
    contexts: List[RetrievedChunk]
    attempt: int
    max_retries: int
    critic: Dict[str, Any]
    previous_quality: float
    mode: str
    trace: List[Dict[str, Any]]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z0-9][a-zA-Z0-9_-]*\b", text.lower())


def content_tokens(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1}


def split_text(text: str, chunk_size: int = 180, overlap: int = 35) -> List[str]:
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

    def search(self, query: str, top_k: int = DEFAULT_TOP_K, dense_k: int = 15, bm25_k: int = 15) -> List[RetrievedChunk]:
        qvec = self.embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0].astype("float32")
        dense_scores = self.embeddings @ qvec
        dense_order = np.argsort(-dense_scores)[: min(dense_k, len(self.chunks))]

        bm25_scores = np.asarray(self.bm25.get_scores(tokenize(query)), dtype="float32")
        bm25_order = np.argsort(-bm25_scores)[: min(bm25_k, len(self.chunks))]

        rrf: Dict[int, float] = {}
        for rank, idx in enumerate(dense_order, start=1):
            rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (RRF_K + rank)
        for rank, idx in enumerate(bm25_order, start=1):
            rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (RRF_K + rank)

        ordered = sorted(rrf, key=rrf.get, reverse=True)[: min(top_k, len(self.chunks))]
        return [
            RetrievedChunk(
                chunk=self.chunks[i],
                dense_score=float(dense_scores[i]),
                bm25_score=float(bm25_scores[i]),
                rrf_score=float(rrf[i]),
            )
            for i in ordered
        ]


class LocalLLM:
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
    blocks: List[str] = []
    total = 0
    for i, item in enumerate(contexts, start=1):
        block = f"[{i}] {item.chunk.text}"
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)


def answer_prompt(question: str, contexts: List[RetrievedChunk]) -> str:
    return f"""Answer the question using ONLY the evidence below.

Rules:
- Do not use outside knowledge.
- If the evidence is insufficient, say exactly: I don't have enough information in the provided documents.
- Give a concise answer in 1-3 sentences.
- Do NOT write citation markers. A separate deterministic citation step will add them.
- Do not invent facts, numbers, dates, names, or policies.

Question:
{question}

Evidence:
{format_contexts(contexts)}

Answer:"""


def rewrite_prompt(question: str, previous_answer: str, critic_reason: str, old_query: str) -> str:
    return f"""Rewrite the retrieval query to find evidence that addresses the specific problem below.
Return ONLY one search query and nothing else.

Original question: {question}
Previous query: {old_query}
Previous answer: {previous_answer}
Failure: {critic_reason}

Preserve important entities, numbers, policy terms, and constraints. Use alternative wording when useful."""


def sentence_parts(answer: str) -> List[str]:
    answer = re.sub(r"\s+", " ", answer).strip()
    if not answer:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]


def strip_citations(text: str) -> str:
    return re.sub(r"\s*\[(\d+)\]", "", text).strip()


def lexical_support(sentence: str, context: str) -> float:
    s = content_tokens(strip_citations(sentence))
    c = content_tokens(context)
    if not s:
        return 0.0
    return len(s & c) / max(1, len(s))


def support_matrix(answer_sentences: List[str], contexts: List[RetrievedChunk], embedder: SentenceTransformer) -> np.ndarray:
    if not answer_sentences or not contexts:
        return np.zeros((len(answer_sentences), len(contexts)))
    sentences = [strip_citations(s) for s in answer_sentences]
    ctx = [r.chunk.text for r in contexts]
    vecs = embedder.encode(sentences + ctx, normalize_embeddings=True, show_progress_bar=False)
    a = vecs[: len(sentences)]
    c = vecs[len(sentences):]
    semantic = a @ c.T
    lexical = np.array([[lexical_support(s, x) for x in ctx] for s in sentences])
    # Semantic similarity handles paraphrases; lexical overlap rewards exact factual evidence.
    return 0.65 * semantic + 0.35 * lexical


def attach_citations(answer: str, contexts: List[RetrievedChunk], embedder: SentenceTransformer, support_threshold: float = 0.43) -> tuple[str, List[Dict[str, Any]]]:
    """Attach citations deterministically to each answer sentence.

    This removes dependence on the small local LLM following citation syntax perfectly.
    """
    sentences = sentence_parts(answer)
    if not sentences or not contexts:
        return answer.strip(), []

    matrix = support_matrix(sentences, contexts, embedder)
    output: List[str] = []
    mapping: List[Dict[str, Any]] = []

    for i, sentence in enumerate(sentences):
        scores = matrix[i]
        order = np.argsort(-scores)
        best_idx = int(order[0])
        best_score = float(scores[best_idx])
        citations: List[int] = []
        if best_score >= support_threshold:
            citations.append(best_idx + 1)
            # Add a second source only when it independently clears the threshold and is useful.
            if len(order) > 1 and float(scores[int(order[1])]) >= support_threshold + 0.05:
                citations.append(int(order[1]) + 1)

        clean = strip_citations(sentence)
        suffix = " " + " ".join(f"[{c}]" for c in citations) if citations else ""
        output.append(clean + suffix)
        mapping.append({
            "sentence": clean,
            "citations": citations,
            "best_chunk": best_idx + 1,
            "best_score": round(best_score, 4),
        })

    return " ".join(output), mapping


def citation_metrics(answer: str, contexts: List[RetrievedChunk], embedder: SentenceTransformer, support_threshold: float = 0.43) -> tuple[bool, bool, float, List[Dict[str, Any]]]:
    sentences = sentence_parts(answer)
    if not sentences or not contexts:
        return False, False, 0.0, []
    matrix = support_matrix(sentences, contexts, embedder)
    mapping: List[Dict[str, Any]] = []
    valid_flags: List[bool] = []
    supported_flags: List[bool] = []

    for i, sentence in enumerate(sentences):
        cited = [int(x) for x in re.findall(r"\[(\d+)\]", sentence)]
        valid = bool(cited) and all(1 <= x <= len(contexts) for x in cited)
        supported = valid and all(matrix[i, x - 1] >= support_threshold for x in cited)
        valid_flags.append(valid)
        supported_flags.append(supported)
        mapping.append({
            "sentence": strip_citations(sentence),
            "citations": cited,
            "valid": valid,
            "supported": supported,
            "best_score": round(float(matrix[i].max()), 4),
        })

    valid_rate = float(np.mean(valid_flags)) if valid_flags else 0.0
    supported_rate = float(np.mean(supported_flags)) if supported_flags else 0.0
    return bool(all(valid_flags)), bool(all(supported_flags)), supported_rate, mapping


def critic_for_answer(question: str, answer: str, contexts: List[RetrievedChunk], embedder: SentenceTransformer) -> CriticResult:
    if not answer.strip() or "I don't have enough information" in answer:
        return CriticResult(False, False, False, False, False, 0.0, 0.0, 0.0, 0.0, "No sufficiently informative answer was produced.", "RETRIEVE_MORE", "INSUFFICIENT_ANSWER")

    sentences = sentence_parts(answer)
    matrix = support_matrix(sentences, contexts, embedder)
    per_sentence = matrix.max(axis=1) if matrix.size else np.zeros(len(sentences))
    grounded_score = float(np.mean(per_sentence)) if len(per_sentence) else 0.0

    # A claim is considered supported when at least one retrieved chunk provides enough evidence.
    grounded = bool(len(per_sentence) > 0 and np.mean(per_sentence >= 0.43) >= 0.80)

    qvec = embedder.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]
    avec = embedder.encode([strip_citations(answer)], normalize_embeddings=True, show_progress_bar=False)[0]
    relevance_score = float(qvec @ avec)
    relevant = relevance_score >= 0.38

    citation_valid, citation_supported, supported_claim_rate, _ = citation_metrics(answer, contexts, embedder)
    complete = len(strip_citations(answer).split()) >= 4

    confidence = float(np.clip(
        0.45 * grounded_score
        + 0.30 * max(0.0, relevance_score)
        + 0.15 * supported_claim_rate
        + 0.10 * (1.0 if complete else 0.0),
        0.0,
        1.0,
    ))

    # Citation formatting is not a reason to regenerate because citations are deterministic.
    if not grounded:
        action, reason, failure = "RETRIEVE_MORE", "At least one answer claim is weakly supported by the retrieved evidence.", "UNSUPPORTED_CLAIM"
    elif not relevant:
        action, reason, failure = "RETRIEVE_MORE", "The answer is weakly aligned with the question.", "LOW_RELEVANCE"
    elif supported_claim_rate < 0.80:
        action, reason, failure = "RETRIEVE_MORE", "Too many answer claims lack sufficient evidence in the current context.", "WEAK_EVIDENCE"
    else:
        action, reason, failure = "PASS", "Answer is sufficiently grounded and relevant.", None

    return CriticResult(
        grounded,
        relevant,
        complete,
        citation_valid,
        citation_supported,
        grounded_score,
        relevance_score,
        confidence,
        supported_claim_rate,
        reason,
        action,
        failure,
    )


def build_graph(retriever: HybridRetriever, llm: LocalLLM, embedder: SentenceTransformer, top_k: int = DEFAULT_TOP_K):
    def add_trace(state: GraphState, node: str, message: str, **extra):
        trace = list(state.get("trace", []))
        trace.append({"time": time.strftime("%H:%M:%S"), "node": node, "message": message, **extra})
        return trace

    def retrieve_node(state: GraphState):
        q = state.get("search_query") or state["question"]
        results = retriever.search(q, top_k=top_k)
        return {
            "contexts": results,
            "trace": add_trace(state, "RETRIEVER", f"Retrieved {len(results)} chunks using dense + BM25 + RRF.", query=q),
        }

    def generate_node(state: GraphState):
        started = time.perf_counter()
        raw = llm.generate(
            "You are a precise evidence-grounded assistant. Use only the supplied evidence. Never invent facts.",
            answer_prompt(state["question"], state["contexts"]),
        )
        # Remove accidental citations before deterministic citation attachment.
        raw = strip_citations(raw)
        cited_answer, mapping = attach_citations(raw, state["contexts"], embedder)
        latency = (time.perf_counter() - started) * 1000
        return {
            "answer": cited_answer,
            "trace": add_trace(
                state,
                "GENERATOR",
                "Generated answer and attached evidence citations deterministically.",
                latency_ms=round(latency, 1),
                citation_mapping=mapping,
            ),
        }

    def critic_node(state: GraphState):
        started = time.perf_counter()
        result = critic_for_answer(state["question"], state["answer"], state["contexts"], embedder)
        latency = (time.perf_counter() - started) * 1000
        return {
            "critic": result.__dict__,
            "trace": add_trace(
                state,
                "CRITIC",
                result.reason,
                action=result.action,
                failure_type=result.failure_type,
                confidence=round(result.confidence, 3),
                grounded_score=round(result.grounded_score, 3),
                relevance_score=round(result.relevance_score, 3),
                supported_claim_rate=round(result.supported_claim_rate, 3),
                latency_ms=round(latency, 1),
            ),
        }

    def rewrite_node(state: GraphState):
        started = time.perf_counter()
        old_query = state.get("search_query") or state["question"]
        rewritten = llm.generate(
            "You rewrite retrieval queries. Output only one concise search query.",
            rewrite_prompt(state["question"], state.get("answer", ""), state["critic"]["reason"], old_query),
            max_new_tokens=60,
        )
        rewritten = normalize_text(rewritten).strip('"').strip()
        if not rewritten or rewritten.lower() == old_query.lower():
            # Deterministic fallback: emphasize the failure type and evidence requirement.
            failure = state["critic"].get("failure_type", "evidence")
            rewritten = f"{state['question']} {failure.replace('_', ' ').lower()} evidence requirements details"
        return {
            "search_query": rewritten,
            "attempt": state.get("attempt", 0) + 1,
            "trace": add_trace(
                state,
                "QUERY_REWRITER",
                f"Reformulated query: {rewritten}",
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
            ),
        }

    def route(state: GraphState):
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
