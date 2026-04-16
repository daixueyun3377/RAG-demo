# LLM / Embedding / Langfuse 初始化
import logging

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:
    from langfuse.callback import CallbackHandler as LangfuseCallbackHandler
except (ImportError, ModuleNotFoundError):
    LangfuseCallbackHandler = None

from app.config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
    EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL,
    LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_HOST,
)

logger = logging.getLogger(__name__)


def get_llm():
    return ChatOpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        temperature=0.3,
        max_tokens=1024,
    )


def get_embeddings():
    return OpenAIEmbeddings(
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_BASE_URL,
        model=EMBEDDING_MODEL,
    )


def get_langfuse_handler():
    if LangfuseCallbackHandler and LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY:
        return LangfuseCallbackHandler(
            secret_key=LANGFUSE_SECRET_KEY,
            public_key=LANGFUSE_PUBLIC_KEY,
            host=LANGFUSE_HOST,
        )
    return None
