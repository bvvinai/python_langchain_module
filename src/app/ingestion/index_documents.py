from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".pdf"}


def load_documents(
    data_dir: str | Path,
    include_paths: Sequence[str | Path] | None = None,
) -> List[Document]:
    from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader

    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")

    selected_paths = None
    if include_paths is not None:
        selected_paths = {Path(path).resolve() for path in include_paths}

    docs: List[Document] = []
    for file_path in root.rglob("*"):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if selected_paths is not None and file_path.resolve() not in selected_paths:
            continue

        if file_path.suffix.lower() == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
        elif file_path.suffix.lower() == ".md":
            loader = TextLoader(str(file_path), encoding="utf-8")
        elif file_path.suffix.lower() == ".csv":
            loader = CSVLoader(str(file_path))
        else:
            loader = PyPDFLoader(str(file_path))

        docs.extend(loader.load())

    return docs


def split_documents(
    documents: Iterable[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(list(documents))
