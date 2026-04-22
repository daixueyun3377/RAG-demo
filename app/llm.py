# LLM / Embedding / Langfuse 初始化
import os
import logging

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
except (ImportError, ModuleNotFoundError):
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

# Langfuse v4 通过环境变量读取配置，确保设置
if LANGFUSE_SECRET_KEY:
    os.environ.setdefault("LANGFUSE_SECRET_KEY", LANGFUSE_SECRET_KEY)
if LANGFUSE_PUBLIC_KEY:
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", LANGFUSE_PUBLIC_KEY)
if LANGFUSE_HOST:
    os.environ.setdefault("LANGFUSE_HOST", LANGFUSE_HOST)


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


def _is_langfuse_enabled() -> bool:
    """检查 Langfuse 是否已配置且可用"""
    return bool(LangfuseCallbackHandler and LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY)


def get_langfuse_handler(
    trace_name: str = "rag-query",
    session_id: str | None = None,
    user_id: str | None = None,
    metadata: dict | None = None,
):
    """
    创建 Langfuse CallbackHandler。

    Langfuse v4 通过环境变量读取 secret_key/public_key/host，
    构造函数不再接受这些参数。直接创建即可。
    """
    if not _is_langfuse_enabled():
        return None

    try:
        handler = LangfuseCallbackHandler()
        return handler
    except Exception as e:
        logger.warning(f"Langfuse handler 创建失败: {e}")
        return None
