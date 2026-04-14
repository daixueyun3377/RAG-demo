"""
长期记忆 (Long-Term Memory) — VectorStoreMemory
=================================================
类比：人的"长期记忆"，通过关联检索回忆过去的经验

核心洞察：
  你在 Day 3 做的 RAG 向量检索 = Agent 长期记忆的底层能力！
  - RAG：文档 → embedding → 向量库 → 语义检索 → 生成回答
  - 长期记忆：对话/事实 → embedding → 向量库 → 语义检索 → 注入上下文

区别只在于"存什么"：
  - RAG 存的是外部文档
  - 长期记忆存的是对话历史、用户偏好、学到的事实

实现：复用项目已有的 Chroma 向量库，单独建一个 collection 存记忆
"""
import logging
import time
from dataclasses import dataclass, field

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from app.config import (
    EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
)

logger = logging.getLogger(__name__)

MEMORY_COLLECTION = "agent_long_term_memory"


def _get_memory_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_BASE_URL,
        model=EMBEDDING_MODEL,
    )


@dataclass
class VectorMemory:
    """
    向量长期记忆 — 语义检索过去的对话和事实
    
    工作原理：
    1. 每轮对话结束后，把「用户问题 + AI 回答」存入向量库
    2. 新对话开始时，用当前问题去向量库检索相关的历史记忆
    3. 把检索到的记忆注入 prompt，让 LLM "回忆"起相关内容
    
    与 RAG 的关系：
    - RAG: query → 检索文档 → 生成回答
    - 长期记忆: query → 检索历史对话 → 注入上下文 → 生成回答
    - 底层能力完全一样！只是数据源不同
    """
    user_id: str = "default"
    _vectorstore: Chroma = field(default=None, repr=False)

    def __post_init__(self):
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                collection_name=MEMORY_COLLECTION,
                embedding_function=_get_memory_embeddings(),
                persist_directory=CHROMA_PERSIST_DIR,
            )

    def store_conversation(self, user_msg: str, ai_msg: str, metadata: dict = None):
        """
        存储一轮对话到长期记忆
        
        存储格式：把 Q+A 拼成一个文档，这样检索时能同时匹配问题和回答
        metadata 记录时间戳、用户 ID 等，方便后续过滤
        """
        content = f"用户: {user_msg}\nAI: {ai_msg}"
        meta = {
            "user_id": self.user_id,
            "timestamp": time.time(),
            "type": "conversation",
            **(metadata or {}),
        }
        doc = Document(page_content=content, metadata=meta)
        self._vectorstore.add_documents([doc])
        logger.info(f"长期记忆已存储: {content[:80]}...")

    def store_fact(self, fact: str, source: str = "user"):
        """
        存储一个事实/偏好到长期记忆
        
        例如："用户喜欢用 Python"、"用户的项目叫 AgentMark"
        这些是从对话中提取的结构化知识
        """
        meta = {
            "user_id": self.user_id,
            "timestamp": time.time(),
            "type": "fact",
            "source": source,
        }
        doc = Document(page_content=fact, metadata=meta)
        self._vectorstore.add_documents([doc])
        logger.info(f"事实已存储: {fact[:80]}...")

    def recall(self, query: str, top_k: int = 3) -> list[dict]:
        """
        回忆：根据当前问题，检索相关的历史记忆
        
        这就是 RAG 的检索步骤！只不过检索的是记忆而不是文档
        """
        results = self._vectorstore.similarity_search_with_score(
            query,
            k=top_k,
            filter={"user_id": self.user_id} if self.user_id != "default" else None,
        )
        memories = []
        for doc, score in results:
            memories.append({
                "content": doc.page_content,
                "type": doc.metadata.get("type", "unknown"),
                "timestamp": doc.metadata.get("timestamp"),
                "relevance_score": round(1 - score, 4),  # Chroma 返回的是距离，转成相似度
            })
        return memories

    def recall_formatted(self, query: str, top_k: int = 3) -> str:
        """回忆并格式化为可注入 prompt 的文本"""
        memories = self.recall(query, top_k)
        if not memories:
            return ""
        lines = ["[相关历史记忆]"]
        for i, mem in enumerate(memories, 1):
            lines.append(f"{i}. {mem['content']}")
        return "\n".join(lines)

    def get_all_memories(self, limit: int = 20) -> list[dict]:
        """获取所有记忆（调试用）"""
        results = self._vectorstore.get(
            where={"user_id": self.user_id} if self.user_id != "default" else None,
            limit=limit,
        )
        memories = []
        if results and results["documents"]:
            for doc, meta in zip(results["documents"], results["metadatas"]):
                memories.append({
                    "content": doc,
                    "type": meta.get("type", "unknown"),
                    "timestamp": meta.get("timestamp"),
                })
        return memories

    def clear(self):
        """清空长期记忆"""
        self._vectorstore.delete_collection()
        self._vectorstore = Chroma(
            collection_name=MEMORY_COLLECTION,
            embedding_function=_get_memory_embeddings(),
            persist_directory=CHROMA_PERSIST_DIR,
        )
        logger.info("长期记忆已清空")
