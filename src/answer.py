"""End-to-end RAG answer generation entry points."""

from __future__ import annotations

import argparse

from src.llm_gemini import generate_with_gemini
from src.prompt import render_baseline_prompt
from src.router import route_query
from src.retrieve import retrieve_documents


def format_docs(docs: list) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("filename", "unknown")
        category = doc.metadata.get("category", "unknown")
        parts.append(f"[Chunk {i} | {category} | {source}]\n{doc.page_content}")
    return "\n\n".join(parts)


def categories_for_route(route: str) -> list[str] | None:
    if route == "method":
        return ["causal_methods"]
    if route == "project":
        return ["netflix_did", "project_notes"]
    if route == "mixed":
        return ["causal_methods", "netflix_did", "project_notes"]
    return None


def answer_question(
    question: str,
    index_dir: str = "artifacts/faiss_index",
    embedding_model: str = "all-MiniLM-L6-v2",
    k: int = 4,
    llm_model: str = "gemini-3-flash-preview",
) -> tuple[str, list, str]:
    route = route_query(question)
    docs = retrieve_documents(
        query=question,
        index_dir=index_dir,
        embedding_model=embedding_model,
        k=k,
        categories=categories_for_route(route),
    )
    context = format_docs(docs)
    prompt = render_baseline_prompt(context=context, question=question)
    answer = generate_with_gemini(prompt=prompt, model=llm_model)
    return answer, docs, route


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the baseline RAG system a question.")
    parser.add_argument("question")
    parser.add_argument("--index-dir", default="artifacts/faiss_index")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--llm-model", default="gemini-3-flash-preview")
    args = parser.parse_args()

    answer, docs, route = answer_question(
        question=args.question,
        index_dir=args.index_dir,
        embedding_model=args.embedding_model,
        k=args.k,
        llm_model=args.llm_model,
    )
    print(f"=== Route ===\n{route}")
    print("=== Retrieved Chunks ===")
    print(format_docs(docs))
    print("\n=== Answer ===")
    print(answer)


if __name__ == "__main__":
    main()
