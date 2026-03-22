"""Load source documents for the domain-specific RAG system."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}
EXCLUDED_FILENAMES = {"README.md", ".gitkeep"}


def _iter_files(data_dir: Path) -> Iterable[Path]:
    for path in sorted(data_dir.rglob("*")):
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_SUFFIXES
            and path.name not in EXCLUDED_FILENAMES
        ):
            yield path


def _category_for_path(path: Path, base_dir: Path) -> str:
    try:
        return path.relative_to(base_dir).parts[0]
    except (ValueError, IndexError):
        return "unknown"


def _load_one(path: Path, base_dir: Path) -> list[Document]:
    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
    else:
        loader = TextLoader(str(path), encoding="utf-8")

    docs = loader.load()
    category = _category_for_path(path, base_dir)
    for doc in docs:
        doc.metadata.update(
            {
                "source": str(path),
                "filename": path.name,
                "category": category,
            }
        )
    return docs


def load_documents(data_dir: str = "data") -> list[Document]:
    base_dir = Path(data_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {base_dir}")

    documents: list[Document] = []
    for path in _iter_files(base_dir):
        if "raw_private" in path.parts or "eval_questions" in path.parts:
            continue
        documents.extend(_load_one(path, base_dir))

    return documents
