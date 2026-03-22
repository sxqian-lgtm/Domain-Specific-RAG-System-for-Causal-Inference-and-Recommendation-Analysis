"""Retrieve relevant chunks from the vector store."""

from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.build_index import SentenceTransformerEmbeddings


def load_vectorstore(
    index_dir: str = "artifacts/faiss_index",
    embedding_model: str = "all-MiniLM-L6-v2",
) -> FAISS:
    embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
    return FAISS.load_local(
        folder_path=index_dir,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )


def retrieve_documents(
    query: str,
    index_dir: str = "artifacts/faiss_index",
    embedding_model: str = "all-MiniLM-L6-v2",
    k: int = 4,
    categories: list[str] | None = None,
) -> list[Document]:
    vectorstore = load_vectorstore(index_dir=index_dir, embedding_model=embedding_model)
    docs = vectorstore.similarity_search(query, k=max(k * 3, 12))
    if categories:
        docs = [doc for doc in docs if doc.metadata.get("category") in categories]
    return docs[:k]
