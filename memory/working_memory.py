"""
工作记忆 (Working Memory)
=========================
类比：人的"工作台"，当前正在处理的任务上下文

与短期记忆的区别：
- 短期记忆 = 对话历史（你说了什么、我说了什么）
- 工作记忆 = 当前任务状态（我在做什么、进展到哪了、还需要什么）

人类例子：
  你在做一道数学题时，脑子里同时 hold 住：
  - 题目条件（输入）
  - 中间计算结果（状态）
  - 下一步要做什么（计划）
  - 最终目标（目标）
  这就是工作记忆。

Agent 的工作记忆：
  - 当前任务描述
  - 已收集的信息 / 中间结果
  - 执行计划和进度
  - 可用工具列表
  - 约束条件
"""
import time
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ScratchpadItem:
    """工作记忆中的一个条目"""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    ttl: float = 0  # 存活时间（秒），0 = 永不过期


@dataclass
class WorkingMemory:
    """
    工作记忆 — Agent 的"工作台"
    
    三个核心区域：
    1. task_context: 当前任务描述和目标
    2. scratchpad: 中间结果暂存区（类似人脑的"草稿纸"）
    3. plan: 执行计划和进度追踪
    
    特点：
    - 容量有限（模拟人脑工作记忆 7±2 限制）
    - 支持 TTL 过期（不重要的信息自动遗忘）
    - 任务切换时可以保存/恢复
    """
    task: str = ""
    goal: str = ""
    scratchpad: dict[str, ScratchpadItem] = field(default_factory=dict)
    plan_steps: list[dict] = field(default_factory=list)
    max_scratchpad_items: int = 10  # 工作记忆容量限制

    # ---- 任务管理 ----

    def set_task(self, task: str, goal: str = ""):
        """设置当前任务"""
        self.task = task
        self.goal = goal
        logger.info(f"工作记忆 - 任务设置: {task}")

    # ---- 草稿纸 (Scratchpad) ----

    def write(self, key: str, value: Any, ttl: float = 0):
        """
        往草稿纸写入一个中间结果
        
        例如：
        - write("user_name", "年年")  # 从对话中提取的信息
        - write("search_results", [...])  # 工具调用的结果
        - write("current_step", 3)  # 当前执行到第几步
        """
        if len(self.scratchpad) >= self.max_scratchpad_items and key not in self.scratchpad:
            # 容量满了，淘汰最旧的
            oldest_key = min(self.scratchpad, key=lambda k: self.scratchpad[k].updated_at)
            del self.scratchpad[oldest_key]
            logger.info(f"工作记忆溢出，淘汰: {oldest_key}")

        self.scratchpad[key] = ScratchpadItem(
            key=key, value=value, ttl=ttl,
            created_at=time.time(), updated_at=time.time(),
        )

    def read(self, key: str, default: Any = None) -> Any:
        """从草稿纸读取"""
        self._evict_expired()
        item = self.scratchpad.get(key)
        if item is None:
            return default
        return item.value

    def erase(self, key: str):
        """擦除草稿纸上的一个条目"""
        self.scratchpad.pop(key, None)

    def _evict_expired(self):
        """清理过期条目"""
        now = time.time()
        expired = [
            k for k, v in self.scratchpad.items()
            if v.ttl > 0 and (now - v.updated_at) > v.ttl
        ]
        for k in expired:
            del self.scratchpad[k]
            logger.info(f"工作记忆过期淘汰: {k}")

    # ---- 计划管理 ----

    def set_plan(self, steps: list[str]):
        """设置执行计划"""
        self.plan_steps = [
            {"step": i + 1, "description": s, "status": "pending", "result": None}
            for i, s in enumerate(steps)
        ]

    def complete_step(self, step_num: int, result: str = ""):
        """标记某步完成"""
        for step in self.plan_steps:
            if step["step"] == step_num:
                step["status"] = "done"
                step["result"] = result
                break

    def get_current_step(self) -> dict | None:
        """获取当前待执行的步骤"""
        for step in self.plan_steps:
            if step["status"] == "pending":
                return step
        return None

    def get_progress(self) -> str:
        """获取计划执行进度"""
        if not self.plan_steps:
            return "无计划"
        done = sum(1 for s in self.plan_steps if s["status"] == "done")
        total = len(self.plan_steps)
        return f"{done}/{total} 步完成"

    # ---- 导出 ----

    def to_prompt_context(self) -> str:
        """
        导出为可注入 prompt 的文本
        
        这是工作记忆的核心价值：把当前任务状态结构化地告诉 LLM
        """
        self._evict_expired()
        parts = []

        if self.task:
            parts.append(f"[当前任务] {self.task}")
        if self.goal:
            parts.append(f"[目标] {self.goal}")

        if self.scratchpad:
            parts.append("[工作台]")
            for key, item in self.scratchpad.items():
                val = str(item.value)
                if len(val) > 200:
                    val = val[:200] + "..."
                parts.append(f"  - {key}: {val}")

        if self.plan_steps:
            parts.append(f"[执行计划] {self.get_progress()}")
            for step in self.plan_steps:
                icon = "✅" if step["status"] == "done" else "⏳"
                parts.append(f"  {icon} Step {step['step']}: {step['description']}")

        return "\n".join(parts) if parts else ""

    def snapshot(self) -> dict:
        """导出完整状态（用于保存/恢复）"""
        self._evict_expired()
        return {
            "task": self.task,
            "goal": self.goal,
            "scratchpad": {
                k: {"value": v.value, "created_at": v.created_at}
                for k, v in self.scratchpad.items()
            },
            "plan": self.plan_steps,
            "progress": self.get_progress(),
        }

    def clear(self):
        """清空工作记忆（任务切换时调用）"""
        self.task = ""
        self.goal = ""
        self.scratchpad.clear()
        self.plan_steps.clear()
