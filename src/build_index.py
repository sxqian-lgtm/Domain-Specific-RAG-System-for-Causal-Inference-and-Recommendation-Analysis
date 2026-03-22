"""Build and persist vector indexes for domain knowledge."""

from __future__ import annotations

import argparse
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from src.load_data import load_documents
from src.split_docs import split_documents


class SentenceTransformerEmbeddings(Embeddings):
    """LangChain-compatible wrapper around Sentence Transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", local_files_only: bool = True) -> None:
        self.model = SentenceTransformer(model_name, local_files_only=local_files_only)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()


def build_index(
    data_dir: str = "data",
    index_dir: str = "artifacts/faiss_index",
    embedding_model: str = "all-MiniLM-L6-v2",
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> tuple[int, int]:
    docs = load_documents(data_dir=data_dir)
    if not docs:
        raise ValueError(
            "No public-safe documents found. Add .txt/.md/.pdf files under data/"
            " excluding data/raw_private/."
        )

    splits = split_documents(
        docs=docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

    target_dir = Path(index_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(target_dir))
    return len(docs), len(splits)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a FAISS index from local data.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--index-dir", default="artifacts/faiss_index")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    args = parser.parse_args()

    raw_count, chunk_count = build_index(
        data_dir=args.data_dir,
        index_dir=args.index_dir,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"Loaded {raw_count} documents and indexed {chunk_count} chunks.")


if __name__ == "__main__":
    main()
