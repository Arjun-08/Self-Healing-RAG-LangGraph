from pathlib import Path

from sentence_transformers import SentenceTransformer

from rag_engine import EMBED_MODEL, GEN_MODEL, HybridRetriever, LocalLLM, build_graph, load_document_bytes, run_rag

ROOT = Path(__file__).parent
DOC = ROOT / "data" / "sample_company_handbook.txt"


def main():
    chunks = load_document_bytes(DOC.name, DOC.read_bytes())
    embedder = SentenceTransformer(EMBED_MODEL)
    llm = LocalLLM(GEN_MODEL)
    retriever = HybridRetriever(chunks, embedder)
    graph = build_graph(retriever, llm, embedder)

    print("Self-Healing RAG")
    print("Type 'exit' to stop.\n")
    while True:
        question = input("Question: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        result = run_rag(graph, question, max_retries=2, mode="Self-Healing RAG")
        print("\nAnswer:")
        print(result.get("answer", ""))
        print("\nCritic:")
        print(result.get("critic", {}))
        print("\nTrace:")
        for event in result.get("trace", []):
            print(f"- {event['node']}: {event['message']}")
        print()


if __name__ == "__main__":
    main()
