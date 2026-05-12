from pathlib import Path

from app.config.loader import load_app_config


def test_load_app_config_defaults(tmp_path: Path, monkeypatch):
    config_file = tmp_path / "providers.yaml"
    config_file.write_text(
        """
defaults:
  llm_provider: openai
  embedding_provider: openai
  vector_db_provider: chroma
llm:
  openai:
    model: gpt-4o-mini
embeddings:
  openai:
    model: text-embedding-3-small
vector_db:
  chroma:
    collection_name: docs
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.delenv("ACTIVE_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ACTIVE_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("ACTIVE_VECTOR_DB_PROVIDER", raising=False)

    cfg = load_app_config(str(config_file))

    assert cfg.active.llm_provider == "openai"
    assert cfg.active.embedding_provider == "openai"
    assert cfg.active.vector_db_provider == "chroma"
    assert cfg.llm_config["model"] == "gpt-4o-mini"
