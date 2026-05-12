from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from app.config.models import ResolvedConfig


def build_vector_store(
    config: ResolvedConfig,
    embeddings: Embeddings,
    documents: Sequence[Document] | None = None,
) -> VectorStore:
    provider = config.active.vector_db_provider
    cfg = config.vector_db_config

    if provider == "chroma":
        from langchain_chroma import Chroma

        persist_directory = cfg.get("persist_directory", "./.chroma")
        collection_name = cfg.get("collection_name", "docs")

        if documents:
            return Chroma.from_documents(
                documents=list(documents),
                embedding=embeddings,
                persist_directory=persist_directory,
                collection_name=collection_name,
            )

        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )

    if provider == "faiss":
        from langchain_community.vectorstores import FAISS

        index_path = Path(cfg.get("index_path", "./.faiss/index"))
        index_path.parent.mkdir(parents=True, exist_ok=True)

        if documents:
            vector_store = FAISS.from_documents(list(documents), embeddings)
            vector_store.save_local(str(index_path))
            return vector_store

        if index_path.exists():
            return FAISS.load_local(
                str(index_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )

        raise FileNotFoundError(
            "FAISS index not found. Run indexing first with the `index` command."
        )

    if provider == "qdrant":
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient

        url = os.getenv("QDRANT_URL", cfg.get("url", "http://localhost:6333"))
        api_key = os.getenv("QDRANT_API_KEY")
        collection_name = cfg.get("collection_name", "docs")
        prefer_grpc = bool(cfg.get("prefer_grpc", False))

        if documents:
            return QdrantVectorStore.from_documents(
                documents=list(documents),
                embedding=embeddings,
                url=url,
                api_key=api_key,
                collection_name=collection_name,
                prefer_grpc=prefer_grpc,
            )

        client = QdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc)
        return QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )

    raise ValueError(f"Unsupported vector DB provider: {provider}")
