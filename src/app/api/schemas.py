from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Prompt for the configured LLM")


class ChatResponse(BaseModel):
    response: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question for RAG")
    k: int = Field(4, ge=1, le=100, description="Number of retrieved chunks")
    ranking_strategy: Literal["similarity", "mmr"] = Field(
        "similarity",
        description="Ranking strategy for retrieval",
    )
    fetch_k: int | None = Field(
        None,
        ge=1,
        le=100,
        description="Candidate pool size used by MMR ranking",
    )
    lambda_mult: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="MMR diversity parameter (0=max diversity, 1=max relevance)",
    )


class RankedSource(BaseModel):
    rank: int = Field(..., ge=1, description="1-based rank of the chunk")
    source: str | None = Field(None, description="Document source path when available")
    preview: str = Field(..., description="Short preview of chunk content")


class AskResponse(BaseModel):
    answer: str
    ranking_strategy: Literal["similarity", "mmr"]
    sources: list[RankedSource] = Field(default_factory=list)


class IndexRequest(BaseModel):
    data_dir: str = Field("data", description="Directory containing .txt, .md, .csv files")
    chunk_size: int = Field(1000, ge=100, le=5000)
    chunk_overlap: int = Field(150, ge=0, le=1000)


class IndexResponse(BaseModel):
    chunks_indexed: int


class ProvidersResponse(BaseModel):
    llm_provider: str
    embedding_provider: str
    vector_db_provider: str


class UploadResponse(BaseModel):
    file_name: str
    saved_path: str
    size_bytes: int
    indexed_chunks: int
