from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

from .models import ActiveProviders, AppConfig, ResolvedConfig


def _read_yaml(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _active_provider(env_key: str, default: str) -> str:
    return os.getenv(env_key, default).strip()


def load_app_config(config_path: str | None = None) -> ResolvedConfig:
    load_dotenv()

    path = Path(
        config_path
        or os.getenv("APP_CONFIG_PATH", "config/providers.yaml")
    ).expanduser()

    config_data = _read_yaml(path)
    parsed = AppConfig.model_validate(config_data)

    active = ActiveProviders(
        llm_provider=_active_provider("ACTIVE_LLM_PROVIDER", parsed.defaults.llm_provider),
        embedding_provider=_active_provider(
            "ACTIVE_EMBEDDING_PROVIDER", parsed.defaults.embedding_provider
        ),
        vector_db_provider=_active_provider(
            "ACTIVE_VECTOR_DB_PROVIDER", parsed.defaults.vector_db_provider
        ),
    )

    llm_config = parsed.llm.get(active.llm_provider)
    embedding_config = parsed.embeddings.get(active.embedding_provider)
    vector_db_config = parsed.vector_db.get(active.vector_db_provider)

    if llm_config is None:
        raise ValueError(f"LLM provider '{active.llm_provider}' not found in config")
    if embedding_config is None:
        raise ValueError(
            f"Embedding provider '{active.embedding_provider}' not found in config"
        )
    if vector_db_config is None:
        raise ValueError(
            f"Vector DB provider '{active.vector_db_provider}' not found in config"
        )

    return ResolvedConfig(
        active=active,
        llm_config=llm_config,
        embedding_config=embedding_config,
        vector_db_config=vector_db_config,
        raw=parsed,
    )
