from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


def _format_documents(documents) -> str:
    return "\n\n".join(doc.page_content for doc in documents)


def _resolve_documents(payload: dict, retriever: BaseRetriever):
    documents = payload.get("documents")
    if documents is not None:
        return documents
    return retriever.invoke(payload["input"])


def build_rag_chain(llm: BaseChatModel, retriever: BaseRetriever):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Use only the retrieved context to answer. "
                "If context is missing, say you do not know.",
            ),
            (
                "human",
                "Question: {input}\n\nContext:\n{context}",
            ),
        ]
    )

    rag_pipeline = (
        RunnablePassthrough.assign(
            context=(
                RunnableLambda(lambda x: _resolve_documents(x, retriever))
                | RunnableLambda(_format_documents)
            )
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_pipeline | RunnableLambda(lambda answer: {"answer": answer})
