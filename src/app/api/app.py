from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi import File, Form, UploadFile

from app.api.schemas import (
    AskRequest,
    AskResponse,
    ChatRequest,
    ChatResponse,
    IndexRequest,
    IndexResponse,
    ProvidersResponse,
    UploadResponse,
)
from app.chains.rag_chain import build_rag_chain
from app.config.loader import load_app_config
from app.factories import build_embeddings, build_llm, build_vector_store
from app.ingestion.index_documents import SUPPORTED_EXTENSIONS, load_documents, split_documents


def create_app() -> FastAPI:
    app = FastAPI(
        title="LangChain Module API",
        version="0.1.0",
        description="FastAPI endpoints for provider-aware LangChain chat and RAG",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/providers", response_model=ProvidersResponse)
    def providers() -> ProvidersResponse:
        cfg = load_app_config()
        return ProvidersResponse(
            llm_provider=cfg.active.llm_provider,
            embedding_provider=cfg.active.embedding_provider,
            vector_db_provider=cfg.active.vector_db_provider,
        )

    @app.post("/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest) -> ChatResponse:
        try:
            cfg = load_app_config()
            llm = build_llm(cfg)
            response = llm.invoke(payload.prompt)
            content = getattr(response, "content", str(response))
            return ChatResponse(response=str(content))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc

    @app.post("/index", response_model=IndexResponse)
    def index(payload: IndexRequest) -> IndexResponse:
        try:
            cfg = load_app_config()
            embeddings = build_embeddings(cfg)

            docs = load_documents(payload.data_dir)
            if not docs:
                raise HTTPException(
                    status_code=400,
                    detail="No supported documents found in the provided data directory",
                )

            chunks = split_documents(
                docs,
                chunk_size=payload.chunk_size,
                chunk_overlap=payload.chunk_overlap,
            )
            build_vector_store(cfg, embeddings, documents=chunks)
            return IndexResponse(chunks_indexed=len(chunks))
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc

    @app.post("/ask", response_model=AskResponse)
    def ask(payload: AskRequest) -> AskResponse:
        try:
            cfg = load_app_config()
            embeddings = build_embeddings(cfg)
            vector_store = build_vector_store(cfg, embeddings)
            retriever = vector_store.as_retriever(search_kwargs={"k": payload.k})

            llm = build_llm(cfg)
            rag = build_rag_chain(llm, retriever)
            result = rag.invoke({"input": payload.question})
            answer = result.get("answer", "No answer returned")
            return AskResponse(answer=answer)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"RAG query failed: {exc}") from exc

    @app.post("/upload", response_model=UploadResponse)
    def upload_file(
        file: UploadFile = File(...),
        data_dir: str = Form("data"),
    ) -> UploadResponse:
        try:
            original_name = file.filename or ""
            safe_name = Path(original_name).name
            if not safe_name:
                raise HTTPException(status_code=400, detail="Uploaded file name is missing")

            extension = Path(safe_name).suffix.lower()
            if extension not in SUPPORTED_EXTENSIONS:
                allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type '{extension}'. Allowed: {allowed}",
                )

            target_dir = Path(data_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            destination = target_dir / safe_name

            with destination.open("wb") as output:
                shutil.copyfileobj(file.file, output)

            cfg = load_app_config()
            embeddings = build_embeddings(cfg)
            docs = load_documents(data_dir=target_dir, include_paths=[destination])
            chunks = split_documents(docs)
            build_vector_store(cfg, embeddings, documents=chunks)

            size_bytes = destination.stat().st_size
            return UploadResponse(
                file_name=safe_name,
                saved_path=str(destination),
                size_bytes=size_bytes,
                indexed_chunks=len(chunks),
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc
        finally:
            file.file.close()

    return app


app = create_app()
