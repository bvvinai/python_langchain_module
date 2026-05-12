from __future__ import annotations

import os
from typing import Any, Dict

from langchain_core.language_models.chat_models import BaseChatModel

from app.config.models import ResolvedConfig


def _optional_args(cfg: Dict[str, Any]) -> Dict[str, Any]:
    args: Dict[str, Any] = {}
    for key in ("temperature", "max_tokens"):
        if key in cfg and cfg[key] is not None:
            args[key] = cfg[key]
    return args


def _get_env(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value else None


def build_llm(config: ResolvedConfig) -> BaseChatModel:
    provider = config.active.llm_provider
    cfg = config.llm_config

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        openai_base_url = _get_env("OPENAI_BASE_URL") or cfg.get("base_url")
        openai_kwargs: Dict[str, Any] = {}
        if openai_base_url:
            openai_kwargs["base_url"] = openai_base_url

        return ChatOpenAI(
            model=cfg["model"],
            api_key=_get_env("OPENAI_API_KEY"),
            **_optional_args(cfg),
            **openai_kwargs,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        anthropic_base_url = (
            _get_env("ANTHROPIC_API_URL")
            or _get_env("ANTHROPIC_BASE_URL")
            or cfg.get("base_url")
        )
        anthropic_kwargs: Dict[str, Any] = {}
        if anthropic_base_url:
            anthropic_kwargs["base_url"] = anthropic_base_url

        return ChatAnthropic(
            model=cfg["model"],
            anthropic_api_key=_get_env("ANTHROPIC_API_KEY"),
            **_optional_args(cfg),
            **anthropic_kwargs,
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=cfg["model"],
            google_api_key=_get_env("GOOGLE_API_KEY"),
            **_optional_args(cfg),
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=cfg["model"],
            base_url=cfg.get("base_url", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")),
            **_optional_args(cfg),
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")
