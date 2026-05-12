from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Prompt for the configured LLM")


class ChatResponse(BaseModel):
    response: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question for RAG")
    k: int = Field(4, ge=1, le=20, description="Number of retrieved chunks")


class AskResponse(BaseModel):
    answer: str


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
