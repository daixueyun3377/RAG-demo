"""
短期记忆 (Short-Term Memory)
=============================
类比：人的"对话记忆"，记住最近说了什么

两种实现：
1. ConversationBufferMemory — 完整保留对话历史（简单但 token 爆炸）
2. ConversationSummaryMemory — LLM 自动摘要压缩（省 token，损失细节）

核心洞察：
- BufferMemory 适合短对话（< 10 轮）
- SummaryMemory 适合长对话，用 LLM 把历史压缩成摘要
- 实际生产中常用 ConversationSummaryBufferMemory（混合模式：最近 N 轮保留原文 + 更早的压缩成摘要）
"""
import logging
from dataclasses import dataclass, field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

logger = logging.getLogger(__name__)


# ========== 1. ConversationBufferMemory ==========
# 最简单的记忆：把所有对话原封不动存下来

@dataclass
class BufferMemory:
    """
    完整对话缓冲区 — 所有消息原样保留
    
    优点：信息零损失，实现简单
    缺点：对话越长 token 越多，成本线性增长
    适用：短对话、需要精确回溯的场景
    """
    messages: list = field(default_factory=list)
    max_messages: int = 50  # 安全上限，防止 token 爆炸

    def add_user_message(self, content: str):
        self.messages.append(HumanMessage(content=content))
        self._trim()

    def add_ai_message(self, content: str):
        self.messages.append(AIMessage(content=content))
        self._trim()

    def get_messages(self) -> list:
        return self.messages.copy()

    def clear(self):
        self.messages.clear()

    def _trim(self):
        """超过上限时丢弃最早的消息"""
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    @property
    def token_estimate(self) -> int:
        """粗略估算 token 数（中文约 1.5 token/字）"""
        total_chars = sum(len(m.content) for m in self.messages)
        return int(total_chars * 1.5)


# ========== 2. ConversationSummaryMemory ==========
# 用 LLM 把对话历史压缩成摘要，省 token

@dataclass
class SummaryMemory:
    """
    摘要记忆 — LLM 自动压缩对话历史
    
    工作原理：
    1. 维护一个 running summary（持续更新的摘要）
    2. 每次新消息进来，让 LLM 把「旧摘要 + 新消息」压缩成新摘要
    3. 对话上下文 = 摘要 + 最近几轮原文
    
    优点：token 消耗恒定，适合超长对话
    缺点：摘要过程本身消耗 LLM 调用，早期细节可能丢失
    """
    summary: str = ""
    recent_messages: list = field(default_factory=list)
    recent_window: int = 4  # 保留最近 N 轮原文
    _llm: ChatOpenAI = field(default=None, repr=False)

    def __post_init__(self):
        if self._llm is None:
            self._llm = ChatOpenAI(
                api_key=LLM_API_KEY,
                base_url=LLM_BASE_URL,
                model=LLM_MODEL,
                temperature=0,
                max_tokens=512,
            )

    def add_user_message(self, content: str):
        self.recent_messages.append(HumanMessage(content=content))
        self._maybe_compress()

    def add_ai_message(self, content: str):
        self.recent_messages.append(AIMessage(content=content))
        self._maybe_compress()

    def _maybe_compress(self):
        """当最近消息超过窗口大小时，把溢出部分压缩进摘要"""
        if len(self.recent_messages) > self.recent_window * 2:
            # 取出要压缩的消息
            to_compress = self.recent_messages[:-self.recent_window * 2]
            self.recent_messages = self.recent_messages[-self.recent_window * 2:]
            self._update_summary(to_compress)

    def _update_summary(self, new_messages: list):
        """让 LLM 把旧摘要 + 新消息压缩成新摘要"""
        messages_text = "\n".join(
            f"{'用户' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in new_messages
        )
        prompt = ChatPromptTemplate.from_template(
            "请将以下对话历史压缩成简洁的摘要。保留关键信息、用户偏好和重要决策。\n\n"
            "已有摘要：\n{existing_summary}\n\n"
            "新的对话：\n{new_messages}\n\n"
            "更新后的摘要："
        )
        chain = prompt | self._llm | StrOutputParser()
        try:
            self.summary = chain.invoke({
                "existing_summary": self.summary or "（无）",
                "new_messages": messages_text,
            })
            logger.info(f"摘要已更新: {self.summary[:100]}...")
        except Exception as e:
            logger.error(f"摘要压缩失败: {e}")

    def get_context(self) -> dict:
        """返回完整上下文：摘要 + 最近消息"""
        return {
            "summary": self.summary,
            "recent_messages": [
                {"role": "user" if isinstance(m, HumanMessage) else "ai", "content": m.content}
                for m in self.recent_messages
            ],
        }

    def get_messages_for_prompt(self) -> list:
        """构造给 LLM 的消息列表：系统摘要 + 最近原文"""
        messages = []
        if self.summary:
            messages.append(SystemMessage(content=f"对话历史摘要：{self.summary}"))
        messages.extend(self.recent_messages)
        return messages

    def clear(self):
        self.summary = ""
        self.recent_messages.clear()


# ========== 3. 带记忆的对话 Chain ==========

class ConversationWithMemory:
    """
    完整的带记忆对话链 — 演示 Memory 如何接入 LLM
    
    支持两种记忆模式：
    - buffer: 完整保留（短对话）
    - summary: 摘要压缩（长对话）
    """

    SYSTEM_PROMPT = "你是一个有记忆的 AI 助手。你能记住之前的对话内容，并据此给出连贯的回答。"

    def __init__(self, memory_type: str = "buffer"):
        self.memory_type = memory_type
        self.memory = BufferMemory() if memory_type == "buffer" else SummaryMemory()
        self.llm = ChatOpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            model=LLM_MODEL,
            temperature=0.7,
            max_tokens=1024,
        )

    def chat(self, user_input: str) -> str:
        """一轮对话：加载记忆 → 生成回答 → 保存记忆"""
        # 1. 记录用户消息
        self.memory.add_user_message(user_input)

        # 2. 构造 prompt（带历史上下文）
        if self.memory_type == "buffer":
            messages = [SystemMessage(content=self.SYSTEM_PROMPT)]
            messages.extend(self.memory.get_messages())
        else:
            messages = [SystemMessage(content=self.SYSTEM_PROMPT)]
            messages.extend(self.memory.get_messages_for_prompt())

        # 3. 调用 LLM
        response = self.llm.invoke(messages)
        ai_reply = response.content

        # 4. 保存 AI 回复到记忆
        self.memory.add_ai_message(ai_reply)

        return ai_reply

    def get_memory_state(self) -> dict:
        """查看当前记忆状态"""
        if self.memory_type == "buffer":
            return {
                "type": "buffer",
                "message_count": len(self.memory.messages),
                "token_estimate": self.memory.token_estimate,
                "messages": [
                    {"role": "user" if isinstance(m, HumanMessage) else "ai", "content": m.content}
                    for m in self.memory.messages
                ],
            }
        else:
            ctx = self.memory.get_context()
            return {
                "type": "summary",
                **ctx,
            }

    def clear_memory(self):
        self.memory.clear()
