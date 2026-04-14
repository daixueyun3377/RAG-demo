"""
Memory API 路由 — 演示三层记忆架构
"""
from typing import Literal, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from memory.buffer_memory import ConversationWithMemory
from memory.vector_memory import VectorMemory
from memory.working_memory import WorkingMemory
from memory.layered_memory import LayeredMemory

router = APIRouter(prefix="/memory", tags=["Memory 三层架构"])

# ========== 会话存储（内存中，演示用）==========
_sessions: dict[str, ConversationWithMemory] = {}
_layered_sessions: dict[str, LayeredMemory] = {}


# ========== 请求模型 ==========

class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str

class MemoryTypeRequest(BaseModel):
    session_id: str = "default"
    memory_type: Literal["buffer", "summary"] = "buffer"

class StoreFactRequest(BaseModel):
    user_id: str = "default"
    fact: str

class RecallRequest(BaseModel):
    user_id: str = "default"
    query: str
    top_k: int = 3

class TaskRequest(BaseModel):
    session_id: str = "default"
    task: str
    goal: str = ""
    steps: list[str] = []

class ScratchpadRequest(BaseModel):
    session_id: str = "default"
    key: str
    value: str
    ttl: float = 0

class LayeredChatRequest(BaseModel):
    session_id: str = "default"
    message: str
    auto_memorize: bool = True


# ========== 1. 短期记忆 API ==========

@router.post("/short-term/create")
async def create_short_term_session(req: MemoryTypeRequest):
    """
    创建带短期记忆的对话会话
    
    - buffer: 完整保留所有对话（适合短对话）
    - summary: LLM 自动摘要压缩（适合长对话）
    """
    _sessions[req.session_id] = ConversationWithMemory(memory_type=req.memory_type)
    return {"session_id": req.session_id, "memory_type": req.memory_type, "message": "会话已创建"}


@router.post("/short-term/chat")
async def short_term_chat(req: ChatRequest):
    """
    短期记忆对话 — 体验 BufferMemory / SummaryMemory 的效果
    
    试试连续问几个相关问题，观察 AI 是否记住了之前的内容
    """
    if req.session_id not in _sessions:
        _sessions[req.session_id] = ConversationWithMemory(memory_type="buffer")

    session = _sessions[req.session_id]
    reply = session.chat(req.message)
    state = session.get_memory_state()

    return {
        "reply": reply,
        "memory_state": state,
    }


@router.get("/short-term/state/{session_id}")
async def get_short_term_state(session_id: str = "default"):
    """查看短期记忆状态"""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "会话不存在")
    return session.get_memory_state()


@router.delete("/short-term/{session_id}")
async def clear_short_term(session_id: str = "default"):
    """清空短期记忆"""
    session = _sessions.get(session_id)
    if session:
        session.clear_memory()
    return {"message": "短期记忆已清空"}


# ========== 2. 长期记忆 API ==========

@router.post("/long-term/store-fact")
async def store_fact(req: StoreFactRequest):
    """
    手动存入一条事实到长期记忆
    
    例如："用户喜欢 Python"、"项目使用 Spring Boot"
    """
    vm = VectorMemory(user_id=req.user_id)
    vm.store_fact(req.fact)
    return {"message": "事实已存入长期记忆", "fact": req.fact}


@router.post("/long-term/recall")
async def recall_memory(req: RecallRequest):
    """
    从长期记忆中检索 — 这就是 RAG！
    
    输入一个问题，看看能检索到哪些相关的历史记忆
    """
    vm = VectorMemory(user_id=req.user_id)
    memories = vm.recall(req.query, top_k=req.top_k)
    return {
        "query": req.query,
        "memories": memories,
        "insight": "这个检索过程和 RAG 的文档检索完全一样，只是数据源从文档变成了记忆",
    }


@router.get("/long-term/all/{user_id}")
async def get_all_memories(user_id: str = "default"):
    """查看所有长期记忆"""
    vm = VectorMemory(user_id=user_id)
    return {"memories": vm.get_all_memories()}


@router.delete("/long-term/{user_id}")
async def clear_long_term(user_id: str = "default"):
    """清空长期记忆"""
    vm = VectorMemory(user_id=user_id)
    vm.clear()
    return {"message": "长期记忆已清空"}


# ========== 3. 工作记忆 API ==========

_working_memories: dict[str, WorkingMemory] = {}


def _get_working_memory(session_id: str) -> WorkingMemory:
    if session_id not in _working_memories:
        _working_memories[session_id] = WorkingMemory()
    return _working_memories[session_id]


@router.post("/working/set-task")
async def set_task(req: TaskRequest):
    """
    设置工作记忆的当前任务
    
    工作记忆 = 你脑子里正在处理的事情
    """
    wm = _get_working_memory(req.session_id)
    wm.set_task(req.task, req.goal)
    if req.steps:
        wm.set_plan(req.steps)
    return {"message": "任务已设置", "state": wm.snapshot()}


@router.post("/working/scratchpad")
async def write_scratchpad(req: ScratchpadRequest):
    """往工作记忆的草稿纸写入中间结果"""
    wm = _get_working_memory(req.session_id)
    wm.write(req.key, req.value, ttl=req.ttl)
    return {"message": f"已写入: {req.key}", "state": wm.snapshot()}


@router.post("/working/complete-step/{session_id}/{step_num}")
async def complete_step(session_id: str, step_num: int, result: str = ""):
    """标记计划中的某步完成"""
    wm = _get_working_memory(session_id)
    wm.complete_step(step_num, result)
    return {"progress": wm.get_progress(), "state": wm.snapshot()}


@router.get("/working/state/{session_id}")
async def get_working_state(session_id: str = "default"):
    """查看工作记忆状态"""
    wm = _get_working_memory(session_id)
    return {
        "snapshot": wm.snapshot(),
        "prompt_context": wm.to_prompt_context(),
    }


# ========== 4. 分层记忆 API（MemGPT 风格）==========

@router.post("/layered/chat")
async def layered_chat(req: LayeredChatRequest):
    """
    分层记忆对话 — 三层记忆协同工作
    
    流程：
    1. 从长期记忆检索相关历史（recall）
    2. 结合短期记忆 + 工作记忆构造 prompt
    3. LLM 生成回答
    4. 自动评估重要性，决定是否存入长期记忆（memorize）
    
    观察返回值中的 memory_recall 和 memorize_action，
    理解记忆是如何流转的
    """
    if req.session_id not in _layered_sessions:
        _layered_sessions[req.session_id] = LayeredMemory(
            user_id=req.session_id,
            auto_memorize=req.auto_memorize,
        )

    session = _layered_sessions[req.session_id]
    result = session.chat(req.message)
    return result


@router.post("/layered/set-task")
async def layered_set_task(req: TaskRequest):
    """为分层记忆会话设置工作任务"""
    if req.session_id not in _layered_sessions:
        _layered_sessions[req.session_id] = LayeredMemory(user_id=req.session_id)

    session = _layered_sessions[req.session_id]
    session.set_task(req.task, req.goal, req.steps or None)
    return {"message": "任务已设置", "working_memory": session.working.snapshot()}


@router.post("/layered/memorize")
async def layered_memorize(req: StoreFactRequest):
    """手动往分层记忆系统存入事实"""
    session_id = req.user_id
    if session_id not in _layered_sessions:
        _layered_sessions[session_id] = LayeredMemory(user_id=session_id)

    session = _layered_sessions[session_id]
    session.memorize(req.fact)
    return {"message": "已存入长期记忆", "fact": req.fact}


@router.get("/layered/state/{session_id}")
async def get_layered_state(session_id: str = "default"):
    """查看分层记忆的完整状态 — 三层记忆一览"""
    session = _layered_sessions.get(session_id)
    if not session:
        raise HTTPException(404, "会话不存在")
    return session.get_state()


@router.delete("/layered/{session_id}")
async def clear_layered(session_id: str = "default"):
    """清空分层记忆"""
    session = _layered_sessions.get(session_id)
    if session:
        session.clear_all()
    return {"message": "所有记忆已清空"}


# ========== 5. 架构说明 API ==========

@router.get("/architecture")
async def memory_architecture():
    """
    Memory 三层架构说明 — 学习指南
    """
    return {
        "title": "Agent Memory 三层架构",
        "layers": {
            "short_term": {
                "name": "短期记忆 (Short-Term Memory)",
                "analogy": "人的对话记忆 — 记住最近说了什么",
                "implementations": [
                    "ConversationBufferMemory — 完整保留，简单但 token 爆炸",
                    "ConversationSummaryMemory — LLM 摘要压缩，省 token 损细节",
                ],
                "api": "/memory/short-term/*",
            },
            "long_term": {
                "name": "长期记忆 (Long-Term Memory)",
                "analogy": "人的长期记忆 — 通过关联回忆过去的经验",
                "key_insight": "VectorStoreMemory = RAG 向量检索！你的 JRag 就是 Agent 长期记忆的底层能力",
                "implementations": [
                    "VectorStoreMemory — 语义检索历史对话和事实",
                ],
                "api": "/memory/long-term/*",
            },
            "working": {
                "name": "工作记忆 (Working Memory)",
                "analogy": "人的工作台 — 当前正在处理的任务上下文",
                "key_insight": "不是记住过去，而是 hold 住当前任务的状态",
                "components": [
                    "task_context — 当前任务描述和目标",
                    "scratchpad — 中间结果暂存区（草稿纸）",
                    "plan — 执行计划和进度追踪",
                ],
                "api": "/memory/working/*",
            },
        },
        "advanced": {
            "MemGPT": "让 LLM 自己决定存/取记忆（主动记忆管理）",
            "Mem0": "个性化记忆层，自动提取用户偏好和事实",
            "layered_api": "/memory/layered/* — 三层记忆协同工作的完整演示",
        },
        "connection_to_rag": {
            "insight": "RAG 的向量检索 = Agent 长期记忆的底层能力",
            "rag_flow": "文档 → embedding → 向量库 → 语义检索 → 生成回答",
            "memory_flow": "对话/事实 → embedding → 向量库 → 语义检索 → 注入上下文",
            "difference": "存什么不同：RAG 存文档，Memory 存对话和事实",
        },
    }
