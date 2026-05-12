from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class DefaultsConfig(BaseModel):
    llm_provider: str
    embedding_provider: str
    vector_db_provider: str


class AppConfig(BaseModel):
    defaults: DefaultsConfig
    llm: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    embeddings: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    vector_db: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class ActiveProviders(BaseModel):
    llm_provider: str
    embedding_provider: str
    vector_db_provider: str


class ResolvedConfig(BaseModel):
    active: ActiveProviders
    llm_config: Dict[str, Any]
    embedding_config: Dict[str, Any]
    vector_db_config: Dict[str, Any]
    raw: AppConfig

    def get(self, section: str, key: str, default: Optional[Any] = None) -> Any:
        if section == "llm":
            return self.llm_config.get(key, default)
        if section == "embeddings":
            return self.embedding_config.get(key, default)
        if section == "vector_db":
            return self.vector_db_config.get(key, default)
        return default
