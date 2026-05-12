from __future__ import annotations

import typer
from rich.console import Console

from app.chains.rag_chain import build_rag_chain
from app.config.loader import load_app_config
from app.factories import build_embeddings, build_llm, build_vector_store
from app.ingestion.index_documents import load_documents, split_documents

app = typer.Typer(help="LangChain project CLI")
console = Console()


@app.command("providers")
def show_providers() -> None:
    cfg = load_app_config()
    console.print("[bold green]Active providers[/bold green]")
    console.print(f"LLM: {cfg.active.llm_provider}")
    console.print(f"Embeddings: {cfg.active.embedding_provider}")
    console.print(f"Vector DB: {cfg.active.vector_db_provider}")


@app.command("chat")
def chat(prompt: str = typer.Argument(..., help="Prompt for the active LLM")) -> None:
    cfg = load_app_config()
    llm = build_llm(cfg)
    response = llm.invoke(prompt)
    console.print(response.content)


@app.command("index")
def index_data(
    data_dir: str = typer.Option("data", help="Directory containing .txt, .md, .csv files"),
    chunk_size: int = typer.Option(1000, help="Text chunk size"),
    chunk_overlap: int = typer.Option(150, help="Chunk overlap"),
) -> None:
    cfg = load_app_config()
    embeddings = build_embeddings(cfg)

    docs = load_documents(data_dir)
    if not docs:
        raise typer.BadParameter("No supported documents found in data directory")

    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    build_vector_store(cfg, embeddings, documents=chunks)

    console.print(f"[bold green]Indexed {len(chunks)} chunks[/bold green]")


@app.command("ask")
def ask(
    question: str = typer.Argument(..., help="Question to ask over indexed documents"),
    k: int = typer.Option(4, help="Number of retrieved chunks"),
) -> None:
    cfg = load_app_config()
    embeddings = build_embeddings(cfg)
    vector_store = build_vector_store(cfg, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    llm = build_llm(cfg)
    rag = build_rag_chain(llm, retriever)
    result = rag.invoke({"input": question})

    console.print(result.get("answer", "No answer returned"))


if __name__ == "__main__":
    app()
