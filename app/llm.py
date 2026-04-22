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
    创建 Langfuse CallbackHandler，支持 trace 级别的上下文。

    同一次 RAG 查询应共享同一个 handler，这样所有 LLM 调用
    会归到同一条 trace 下，在 Langfuse 面板中可以看到完整链路。

    Args:
        trace_name: trace 名称，用于在 Langfuse 中标识查询类型
        session_id: 会话 ID，同一用户的多次查询可归到同一 session
        user_id: 用户标识
        metadata: 附加元数据（如 retrieval_mode, query_transform 等）
    """
    if not _is_langfuse_enabled():
        return None

    try:
        handler = LangfuseCallbackHandler(
            secret_key=LANGFUSE_SECRET_KEY,
            public_key=LANGFUSE_PUBLIC_KEY,
            host=LANGFUSE_HOST,
            trace_name=trace_name,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata or {},
        )
        return handler
    except Exception as e:
        logger.warning(f"Langfuse handler 创建失败: {e}")
        return None
