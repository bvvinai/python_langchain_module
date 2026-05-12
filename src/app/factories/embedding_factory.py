from __future__ import annotations

import os
from typing import Any, Dict

from langchain_core.embeddings import Embeddings

from app.config.models import ResolvedConfig


def build_embeddings(config: ResolvedConfig) -> Embeddings:
    provider = config.active.embedding_provider
    cfg = config.embedding_config

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        openai_base_url = os.getenv("OPENAI_BASE_URL") or cfg.get("base_url")
        openai_kwargs: Dict[str, Any] = {}
        if openai_base_url:
            openai_kwargs["base_url"] = openai_base_url

        return OpenAIEmbeddings(
            model=cfg["model"],
            api_key=os.getenv("OPENAI_API_KEY"),
            **openai_kwargs,
        )

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=cfg["model"])

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=cfg["model"],
            base_url=cfg.get("base_url", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")),
        )

    raise ValueError(f"Unsupported embedding provider: {provider}")
