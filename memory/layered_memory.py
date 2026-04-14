from __future__ import annotations

"""
分层记忆架构 (Layered Memory Architecture)
==========================================
灵感来源：MemGPT / Mem0 — 模拟人类记忆的分层管理

核心思想：
  人脑不是一个大杂烩，而是分层管理记忆的：
  - 感觉记忆（毫秒级）→ 对应 Agent 的原始输入
  - 短期记忆（秒~分钟）→ 对应 BufferMemory（当前对话）
  - 工作记忆（当前任务）→ 对应 WorkingMemory（任务上下文）
  - 长期记忆（天~年）→ 对应 VectorMemory（向量检索）

MemGPT 的关键创新：
  让 LLM 自己决定什么时候"存记忆"和"取记忆"
  - 传统方式：每轮对话都自动存/取（被动）
  - MemGPT 方式：LLM 通过 function call 主动管理记忆（主动）
  
  类比：
  - 传统 = 自动录像机，什么都录
  - MemGPT = 人脑，主动决定记住什么、忘记什么、什么时候回忆

本模块实现一个简化版的分层记忆系统，把三层记忆统一管理。
"""
import logging
import time
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from memory.buffer_memory import BufferMemory
from memory.vector_memory import VectorMemory
from memory.working_memory import WorkingMemory

logger = logging.getLogger(__name__)


# ========== 记忆重要性评估 ==========

IMPORTANCE_PROMPT = ChatPromptTemplate.from_template(
    "请评估以下对话内容的重要性（1-10 分）。\n"
    "高分标准：包含用户偏好、重要决策、关键事实、个人信息。\n"
    "低分标准：闲聊、重复内容、临时性问题。\n\n"
    "对话内容：\n用户: {user_msg}\nAI: {ai_msg}\n\n"
    "只输出一个数字（1-10）："
)

FACT_EXTRACTION_PROMPT = ChatPromptTemplate.from_template(
    "从以下对话中提取值得长期记住的事实或用户偏好。\n"
    "如果没有值得记住的内容，输出「无」。\n"
    "每条事实一行，简洁明了。\n\n"
    "对话内容：\n用户: {user_msg}\nAI: {ai_msg}\n\n"
    "提取的事实："
)


@dataclass
class LayeredMemory:
    """
    分层记忆系统 — 统一管理三层记忆
    
    架构：
    ┌─────────────────────────────────────┐
    │  LLM (大脑)                          │
    │  ┌───────────┐  ┌────────────────┐  │
    │  │ 工作记忆   │  │ 短期记忆        │  │
    │  │ (任务状态) │  │ (当前对话)      │  │
    │  └───────────┘  └────────────────┘  │
    │         ↕ 主动存取                    │
    │  ┌────────────────────────────────┐  │
    │  │ 长期记忆 (向量库)               │  │
    │  │ - 历史对话                      │  │
    │  │ - 用户偏好                      │  │
    │  │ - 学到的事实                    │  │
    │  └────────────────────────────────┘  │
    └─────────────────────────────────────┘
    
    记忆流转：
    1. 用户输入 → 短期记忆
    2. 短期记忆溢出 → 评估重要性 → 重要的存入长期记忆
    3. 新对话开始 → 从长期记忆检索相关内容 → 注入工作记忆
    4. LLM 生成回答时，同时看到三层记忆的内容
    """
    user_id: str = "default"
    # 三层记忆
    short_term: BufferMemory = field(default_factory=lambda: BufferMemory(max_messages=20))
    long_term: VectorMemory = field(default=None)
    working: WorkingMemory = field(default_factory=WorkingMemory)
    # 配置
    auto_memorize: bool = True  # 是否自动评估并存储重要记忆
    importance_threshold: int = 6  # 重要性阈值（>=6 才存入长期记忆）
    # 内部
    _llm: ChatOpenAI = field(default=None, repr=False)

    def __post_init__(self):
        if self.long_term is None:
            self.long_term = VectorMemory(user_id=self.user_id)
        if self._llm is None:
            self._llm = ChatOpenAI(
                api_key=LLM_API_KEY,
                base_url=LLM_BASE_URL,
                model=LLM_MODEL,
                temperature=0,
                max_tokens=256,
            )

    def chat(self, user_input: str) -> dict:
        """
        完整的分层记忆对话流程
        
        流程：
        1. 从长期记忆检索相关内容（recall）
        2. 构造带三层记忆的 prompt
        3. LLM 生成回答
        4. 更新短期记忆
        5. 评估是否需要存入长期记忆（memorize）
        """
        # 1. Recall — 从长期记忆检索
        long_term_context = self.long_term.recall_formatted(user_input, top_k=3)

        # 2. 构造 prompt
        messages = self._build_prompt(user_input, long_term_context)

        # 3. 生成回答
        llm = ChatOpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            model=LLM_MODEL,
            temperature=0.7,
            max_tokens=1024,
        )
        response = llm.invoke(messages)
        ai_reply = response.content

        # 4. 更新短期记忆
        self.short_term.add_user_message(user_input)
        self.short_term.add_ai_message(ai_reply)

        # 5. 自动记忆管理（异步评估重要性）
        memorize_result = None
        if self.auto_memorize:
            memorize_result = self._auto_memorize(user_input, ai_reply)

        return {
            "reply": ai_reply,
            "memory_recall": long_term_context or "(无相关历史记忆)",
            "memorize_action": memorize_result,
        }

    def _build_prompt(self, user_input: str, long_term_context: str) -> list:
        """
        构造带三层记忆的 prompt
        
        这是分层记忆的核心：把三层记忆的信息结构化地注入 LLM
        """
        system_parts = [
            "你是一个拥有分层记忆系统的 AI 助手。",
            "你能记住之前的对话，理解当前任务，并利用长期记忆提供个性化回答。",
        ]

        # 注入工作记忆（当前任务上下文）
        working_ctx = self.working.to_prompt_context()
        if working_ctx:
            system_parts.append(f"\n{working_ctx}")

        # 注入长期记忆（检索到的相关历史）
        if long_term_context:
            system_parts.append(f"\n{long_term_context}")

        messages = [SystemMessage(content="\n".join(system_parts))]

        # 注入短期记忆（最近对话）
        messages.extend(self.short_term.get_messages())

        # 当前用户输入
        messages.append(HumanMessage(content=user_input))

        return messages

    def _auto_memorize(self, user_msg: str, ai_msg: str) -> dict | None:
        """
        自动记忆管理 — MemGPT 的核心思想简化版
        
        流程：
        1. 评估这轮对话的重要性
        2. 如果重要，存入长期记忆
        3. 尝试提取事实/偏好
        """
        try:
            # 评估重要性
            chain = IMPORTANCE_PROMPT | self._llm | StrOutputParser()
            score_str = chain.invoke({"user_msg": user_msg, "ai_msg": ai_msg}).strip()
            score = int("".join(c for c in score_str if c.isdigit())[:2] or "5")
            score = min(max(score, 1), 10)

            result = {"importance_score": score, "stored": False, "facts": []}

            if score >= self.importance_threshold:
                # 存入长期记忆
                self.long_term.store_conversation(user_msg, ai_msg, {"importance": score})
                result["stored"] = True

                # 尝试提取事实
                fact_chain = FACT_EXTRACTION_PROMPT | self._llm | StrOutputParser()
                facts_text = fact_chain.invoke({"user_msg": user_msg, "ai_msg": ai_msg})
                if facts_text.strip() and facts_text.strip() != "无":
                    facts = [f.strip() for f in facts_text.strip().split("\n") if f.strip() and f.strip() != "无"]
                    for fact in facts[:3]:  # 最多存 3 条
                        self.long_term.store_fact(fact, source="auto_extract")
                    result["facts"] = facts

                logger.info(f"记忆已存储 (重要性={score}): {user_msg[:50]}...")

            return result

        except Exception as e:
            logger.warning(f"自动记忆评估失败: {e}")
            return None

    # ---- 手动记忆操作（模拟 MemGPT 的 function call）----

    def memorize(self, content: str, memory_type: str = "fact"):
        """手动存入长期记忆"""
        if memory_type == "fact":
            self.long_term.store_fact(content, source="manual")
        else:
            self.long_term.store_conversation(content, "", {"source": "manual"})

    def recall(self, query: str, top_k: int = 3) -> list[dict]:
        """手动检索长期记忆"""
        return self.long_term.recall(query, top_k)

    def set_task(self, task: str, goal: str = "", steps: list[str] = None):
        """设置工作记忆的任务"""
        self.working.set_task(task, goal)
        if steps:
            self.working.set_plan(steps)

    # ---- 状态查看 ----

    def get_state(self) -> dict:
        """获取三层记忆的完整状态"""
        return {
            "short_term": {
                "message_count": len(self.short_term.messages),
                "token_estimate": self.short_term.token_estimate,
            },
            "long_term": {
                "memories": self.long_term.get_all_memories(limit=10),
            },
            "working": self.working.snapshot(),
        }

    def clear_all(self):
        """清空所有记忆"""
        self.short_term.clear()
        self.long_term.clear()
        self.working.clear()
