# 智能入库图 — LangGraph 动态选择切分策略 + 质量验证 + 降级重试
import os
import re
import logging
from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

from app.config import CHUNK_SIZE
from app.retriever import load_file, split_documents, ingest_documents

logger = logging.getLogger(__name__)

# 降级顺序：策略失败后，按这个顺序尝试
_FALLBACK_CHAIN = ["recursive", "fixed", "semantic"]

# 切分质量的硬指标阈值
_MIN_CHUNKS = 1           # 至少切出 1 个块
_MAX_EMPTY_RATIO = 0.1    # 空块（<10字符）占比不超过 10%
_MIN_AVG_LENGTH = 50      # 平均块长度不低于 50 字符
_MAX_AVG_LENGTH = 3000    # 平均块长度不超过 3000 字符（太长说明没切开）


class IngestState(TypedDict):
    """智能入库图的状态"""
    file_path: str
    documents: list
    doc_analysis: str
    chosen_strategy: str
    chosen_chunk_size: int
    chosen_chunk_overlap: int
    chunks: list
    quality_pass: bool
    quality_issues: str
    tried_strategies: list
    ingest_count: int
    steps: list


# ================================================================
# 文档特征分析（纯规则，零 LLM 调用）
# ================================================================

def _analyze_doc_features(content: str) -> dict:
    """
    纯规则启发式分析文档特征，零 LLM 调用，零延迟。
    返回 {"strategy": str, "chunk_size": int, "reason": str}
    """
    total_len = len(content)
    if total_len == 0:
        return {"strategy": "recursive", "chunk_size": CHUNK_SIZE, "reason": "空文档，使用默认策略"}

    # ---- 特征提取 ----
    heading_count = len(re.findall(r"^#{1,6}\s+", content, re.MULTILINE))
    list_count = len(re.findall(r"^[\s]*[-*]\s+|^[\s]*\d+\.\s+", content, re.MULTILINE))
    code_block_count = content.count("```") // 2
    table_row_count = len(re.findall(r"^\|.+\|", content, re.MULTILINE))

    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    para_count = len(paragraphs)
    avg_para_len = total_len / max(para_count, 1)

    structure_marks = heading_count + list_count + code_block_count + table_row_count
    structure_density = structure_marks / (total_len / 1000)

    newline_count = content.count("\n")
    newline_density = newline_count / (total_len / 1000)

    # ---- 决策 ----
    reasons = []

    if structure_density > 3:
        strategy = "recursive"
        reasons.append(f"结构化标记密度高({structure_density:.1f}/千字)")
        if code_block_count >= 3:
            chunk_size = 1024
            reasons.append(f"含{code_block_count}个代码块，用大块保持完整性")
        elif total_len < 2000:
            chunk_size = 256
            reasons.append("短文档用小块")
        else:
            chunk_size = 512
            reasons.append("中等长度结构化文档")

    elif structure_density < 1 and avg_para_len > 500:
        strategy = "semantic"
        reasons.append(f"结构标记稀疏({structure_density:.1f}/千字)，段落长({avg_para_len:.0f}字)")
        chunk_size = 512
        reasons.append("长段落少标记，适合语义切分")

    elif structure_density < 0.5 and newline_density < 5:
        strategy = "fixed"
        reasons.append(f"纯文本流(换行密度{newline_density:.1f}/千字，标记密度{structure_density:.1f}/千字)")
        if total_len > 10000:
            chunk_size = 1024
            reasons.append("长文本用大块减少碎片")
        else:
            chunk_size = 512
            reasons.append("中等文本")

    else:
        strategy = "recursive"
        reasons.append(f"通用文档(标记密度{structure_density:.1f}，段落均长{avg_para_len:.0f})")
        chunk_size = 512

    reason = "；".join(reasons)
    return {"strategy": strategy, "chunk_size": chunk_size, "reason": reason}


# ================================================================
# 入库图节点
# ================================================================

def load_document_node(state: IngestState) -> dict:
    """Ingest Node 1: 加载文档"""
    file_path = state["file_path"]
    docs = load_file(file_path)
    return {
        "documents": docs,
        "steps": state.get("steps", []) + [f"load → {len(docs)} docs from {os.path.basename(file_path)}"],
    }


def analyze_document_node(state: IngestState) -> dict:
    """Ingest Node 2: 基于规则的文档特征分析，零延迟零成本"""
    docs = state["documents"]
    content = docs[0].page_content if docs else ""

    result = _analyze_doc_features(content)
    strategy = result["strategy"]
    chunk_size = result["chunk_size"]
    reason = result["reason"]

    return {
        "doc_analysis": reason,
        "chosen_strategy": strategy,
        "chosen_chunk_size": chunk_size,
        "chosen_chunk_overlap": chunk_size // 10,
        "tried_strategies": [strategy],
        "steps": state.get("steps", []) + [f"analyze(rule) → {strategy}(size={chunk_size}): {reason}"],
    }


def split_document_node(state: IngestState) -> dict:
    """Ingest Node 3: 按选定策略切分"""
    docs = state["documents"]
    strategy = state["chosen_strategy"]
    chunk_size = state["chosen_chunk_size"]
    chunk_overlap = state["chosen_chunk_overlap"]

    try:
        chunks = split_documents(docs, strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    except Exception as e:
        logger.warning(f"Split failed with {strategy}: {e}")
        return {
            "chunks": [],
            "quality_pass": False,
            "quality_issues": f"切分执行失败: {e}",
            "steps": state.get("steps", []) + [f"split({strategy}) → FAILED: {e}"],
        }

    return {
        "chunks": chunks,
        "steps": state.get("steps", []) + [f"split({strategy}, size={chunk_size}) → {len(chunks)} chunks"],
    }


def validate_chunks_node(state: IngestState) -> dict:
    """Ingest Node 4: 切分质量验证 — 用硬指标判断"""
    chunks = state["chunks"]
    issues = []

    if len(chunks) < _MIN_CHUNKS:
        issues.append(f"切分结果为空（{len(chunks)} chunks）")

    if chunks:
        lengths = [len(c.page_content) for c in chunks]
        avg_len = sum(lengths) / len(lengths)

        empty_count = sum(1 for l in lengths if l < 10)
        empty_ratio = empty_count / len(chunks)
        if empty_ratio > _MAX_EMPTY_RATIO:
            issues.append(f"空块过多: {empty_count}/{len(chunks)} ({empty_ratio:.0%}) 的块不足10字符")

        if avg_len < _MIN_AVG_LENGTH:
            issues.append(f"切分过碎: 平均长度 {avg_len:.0f} 字符 < {_MIN_AVG_LENGTH}")

        if avg_len > _MAX_AVG_LENGTH:
            issues.append(f"切分不足: 平均长度 {avg_len:.0f} 字符 > {_MAX_AVG_LENGTH}")

    passed = len(issues) == 0
    issues_str = "; ".join(issues) if issues else ""
    step_msg = f"validate → {'PASS' if passed else 'FAIL'}"
    if not passed:
        step_msg += f" ({issues_str})"

    return {
        "quality_pass": passed,
        "quality_issues": issues_str,
        "steps": state.get("steps", []) + [step_msg],
    }


def fallback_strategy_node(state: IngestState) -> dict:
    """Ingest Node 5: 降级到下一个策略"""
    tried = state.get("tried_strategies", [])
    issues = state.get("quality_issues", "")

    next_strategy = None
    for s in _FALLBACK_CHAIN:
        if s not in tried:
            next_strategy = s
            break

    if next_strategy is None:
        next_strategy = "recursive"
        chunk_size = CHUNK_SIZE
        step_msg = f"fallback → 所有策略均已尝试，最终兜底: recursive(size={chunk_size})"
    else:
        chunk_size = CHUNK_SIZE
        step_msg = f"fallback → {state['chosen_strategy']} 质量不合格({issues})，降级到 {next_strategy}(size={chunk_size})"

    return {
        "chosen_strategy": next_strategy,
        "chosen_chunk_size": chunk_size,
        "chosen_chunk_overlap": chunk_size // 10,
        "tried_strategies": tried + [next_strategy],
        "quality_pass": False,
        "chunks": [],
        "steps": state.get("steps", []) + [step_msg],
    }


def store_document_node(state: IngestState) -> dict:
    """Ingest Node 6: 存入向量库"""
    chunks = state["chunks"]
    count = ingest_documents(chunks)
    return {
        "ingest_count": count,
        "steps": state.get("steps", []) + [f"store → {count} chunks ingested"],
    }


# ================================================================
# 条件路由 & 图构建
# ================================================================

def route_after_validation(state: IngestState) -> str:
    """质量验证后的路由"""
    if state.get("quality_pass", False):
        return "store"

    tried = state.get("tried_strategies", [])
    has_untried = any(s not in tried for s in _FALLBACK_CHAIN)

    if has_untried:
        return "fallback"

    if state.get("chunks"):
        return "store"
    return "fallback"


def build_ingest_graph():
    """
    构建智能入库状态图：

    START → load_document → analyze_document → split_document → validate_chunks
                                                                     ↓
                                                           ┌── quality ok? ──┐
                                                           ↓ yes             ↓ no
                                                     store_document    fallback_strategy
                                                           ↓                 ↓
                                                          END          split_document (重试)
    """
    graph = StateGraph(IngestState)

    graph.add_node("load_document", load_document_node)
    graph.add_node("analyze_document", analyze_document_node)
    graph.add_node("split_document", split_document_node)
    graph.add_node("validate_chunks", validate_chunks_node)
    graph.add_node("fallback_strategy", fallback_strategy_node)
    graph.add_node("store_document", store_document_node)

    graph.add_edge(START, "load_document")
    graph.add_edge("load_document", "analyze_document")
    graph.add_edge("analyze_document", "split_document")
    graph.add_edge("split_document", "validate_chunks")

    graph.add_conditional_edges(
        "validate_chunks",
        route_after_validation,
        {"store": "store_document", "fallback": "fallback_strategy"},
    )

    graph.add_edge("fallback_strategy", "split_document")
    graph.add_edge("store_document", END)

    return graph.compile()


# 编译入库图（模块级单例）
ingest_graph = build_ingest_graph()


def smart_ingest_file(file_path: str) -> dict:
    """
    智能入库入口 — LangGraph 版
    规则分析文档特征，动态选择最佳切分策略
    切分后验证质量，不合格自动降级重试
    """
    initial_state: IngestState = {
        "file_path": file_path,
        "documents": [],
        "doc_analysis": "",
        "chosen_strategy": "",
        "chosen_chunk_size": 0,
        "chosen_chunk_overlap": 0,
        "chunks": [],
        "quality_pass": False,
        "quality_issues": "",
        "tried_strategies": [],
        "ingest_count": 0,
        "steps": [],
    }

    final = ingest_graph.invoke(initial_state)

    return {
        "filename": os.path.basename(file_path),
        "chunks": final["ingest_count"],
        "strategy": final["chosen_strategy"],
        "chunk_size": final["chosen_chunk_size"],
        "analysis": final["doc_analysis"],
        "tried_strategies": final.get("tried_strategies", []),
        "quality_issues": final.get("quality_issues", ""),
        "graph_steps": final.get("steps", []),
    }
