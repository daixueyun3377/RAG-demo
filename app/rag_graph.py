# RAG 查询图 — LangGraph 版
# 将 RAG 查询流程建模为显式状态图：
#   query变换 → 检索 → 文档评估 → (重排序) → 生成 → 幻觉检测 → 输出/重试

import re
import logging
from typing import Literal, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

from app.config import TOP_K
from app.llm import get_llm, get_langfuse_handler
from app.retriever import (
    get_vector_retriever, get_bm25_retriever, hybrid_retrieve,
    rerank_documents,
)

logger = logging.getLogger(__name__)


# ================================================================
# LangGraph State 定义
# ================================================================

class RAGState(TypedDict):
    """RAG 图的全局状态"""
    question: str
    search_query: str
    retrieval_mode: str
    query_transform: str
    use_reranker: bool
    top_k: int
    retrieved_docs: list
    relevant_docs: list
    answer: str
    hallucination_pass: bool
    retry_count: int
    sources: list
    steps: list


# ================================================================
# Graph Nodes
# ================================================================

def transform_query(state: RAGState) -> dict:
    """Node 1: Query 变换 — rewrite / hyde / 直接透传"""
    question = state["question"]
    mode = state.get("query_transform", "none")
    llm = get_llm()

    if mode == "rewrite":
        prompt = ChatPromptTemplate.from_template(
            "你是一个搜索查询优化助手。请将用户的问题改写为更适合在知识库中检索的查询语句。"
            "只输出改写后的查询，不要解释。\n\n用户问题：{question}"
        )
        search_query = (prompt | llm | StrOutputParser()).invoke({"question": question})
        return {"search_query": search_query, "steps": state.get("steps", []) + [f"query_rewrite → {search_query}"]}

    elif mode == "hyde":
        prompt = ChatPromptTemplate.from_template(
            "请针对以下问题写一段简短的回答（约100字），即使你不确定也请尝试回答。"
            "这段回答将用于检索相关文档。\n\n问题：{question}"
        )
        search_query = (prompt | llm | StrOutputParser()).invoke({"question": question})
        return {"search_query": search_query, "steps": state.get("steps", []) + [f"hyde → {search_query[:80]}..."]}

    else:
        return {"search_query": question, "steps": state.get("steps", []) + ["query_passthrough"]}


def retrieve(state: RAGState) -> dict:
    """Node 2: 检索 — vector / bm25 / hybrid"""
    query = state["search_query"]
    mode = state.get("retrieval_mode", "hybrid")
    top_k = state.get("top_k", TOP_K)

    if mode == "vector":
        docs = get_vector_retriever(top_k).invoke(query)
    elif mode == "bm25":
        retriever = get_bm25_retriever(top_k)
        docs = retriever.invoke(query) if retriever else []
    else:
        docs = hybrid_retrieve(query, top_k)

    return {
        "retrieved_docs": docs,
        "steps": state.get("steps", []) + [f"retrieve({mode}) → {len(docs)} docs"],
    }


def _parse_grade_result(raw: str, num_docs: int) -> list[bool]:
    """
    解析 LLM 批量评估结果，增强鲁棒性。
    支持格式：
      - 每行 yes/no
      - 带编号：[文档0] yes / 1. yes / 文档0: yes
      - 带额外文字：yes, this is relevant
    返回长度为 num_docs 的 bool 列表，解析失败的位置保守返回 True。
    """
    results = [True] * num_docs  # 默认保守保留

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    parsed_idx = 0

    for line in lines:
        if parsed_idx >= num_docs:
            break
        lower = line.lower()
        # 去掉常见前缀：[文档0]、1.、文档0:
        cleaned = re.sub(r"^(\[?文档\s*\d+\]?[:\s]*|\d+[.)\]:\s]+)", "", lower).strip()
        if cleaned.startswith("yes"):
            results[parsed_idx] = True
            parsed_idx += 1
        elif cleaned.startswith("no"):
            results[parsed_idx] = False
            parsed_idx += 1
        # 跳过无法解析的行（不推进 index）

    return results


def grade_documents(state: RAGState) -> dict:
    """Node 3: 文档相关性评估 — LLM 批量判断文档是否与问题相关"""
    question = state["question"]
    docs = state["retrieved_docs"]

    if not docs:
        return {
            "relevant_docs": [],
            "steps": state.get("steps", []) + ["grade_documents → 0/0 relevant (no docs)"],
        }

    llm = get_llm()

    doc_list = "\n\n".join(
        f"[文档{i}]\n{doc.page_content}" for i, doc in enumerate(docs)
    )
    grade_prompt = ChatPromptTemplate.from_template(
        "你是一个文档相关性评估专家。判断以下每篇文档是否与用户问题相关。\n"
        "请对每篇文档只回答 'yes' 或 'no'，每行一个结果，顺序与文档编号一致。\n"
        "不要解释，不要输出其他内容。\n\n"
        "用户问题：{question}\n\n"
        "文档列表：\n{documents}"
    )
    chain = grade_prompt | llm | StrOutputParser()
    raw_result = chain.invoke({"question": question, "documents": doc_list}).strip()

    grade_results = _parse_grade_result(raw_result, len(docs))
    relevant = [doc for doc, is_relevant in zip(docs, grade_results) if is_relevant]

    return {
        "relevant_docs": relevant,
        "steps": state.get("steps", []) + [f"grade_documents(batch) → {len(relevant)}/{len(docs)} relevant"],
    }


def rerank(state: RAGState) -> dict:
    """Node 4: Reranker 重排序（可选）"""
    question = state["question"]
    docs = state["relevant_docs"]
    top_k = state.get("top_k", TOP_K)

    if not state.get("use_reranker", False) or not docs:
        return {"relevant_docs": docs, "steps": state.get("steps", []) + ["rerank_skipped"]}

    try:
        reranked = rerank_documents(question, docs, top_k)
        return {"relevant_docs": reranked, "steps": state.get("steps", []) + [f"rerank → {len(reranked)} docs"]}
    except Exception as e:
        logger.warning(f"Reranker failed: {e}, using original order")
        return {"relevant_docs": docs, "steps": state.get("steps", []) + [f"rerank_failed: {e}"]}


def generate(state: RAGState) -> dict:
    """Node 5: LLM 生成回答"""
    question = state["question"]
    docs = state["relevant_docs"]
    llm = get_llm()
    langfuse_handler = get_langfuse_handler()
    callbacks = [langfuse_handler] if langfuse_handler else []

    context_parts = []
    for doc in docs:
        source = doc.metadata.get("source", "未知")
        context_parts.append(f"[来源: {source}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_template(
        "你是一个知识库问答助手。请基于以下检索到的内容回答用户问题。\n"
        "如果检索内容中没有相关信息，请诚实说明。回答时标注信息来源。\n\n"
        "检索到的内容：\n{context}\n\n"
        "用户问题：{question}\n\n回答："
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question}, config={"callbacks": callbacks})

    sources = [{"text": doc.page_content[:200], "source": doc.metadata.get("source", "未知")} for doc in docs]

    return {
        "answer": answer,
        "sources": sources,
        "steps": state.get("steps", []) + ["generate"],
    }


def check_hallucination(state: RAGState) -> dict:
    """Node 6: 幻觉检测 — 验证回答是否有文档依据"""
    answer = state["answer"]
    docs = state["relevant_docs"]
    llm = get_llm()

    doc_contents = "\n\n".join([doc.page_content for doc in docs])

    prompt = ChatPromptTemplate.from_template(
        "你是一个事实核查专家。请判断以下回答是否完全基于提供的参考文档，没有编造信息。\n"
        "只回答 'yes'（有依据）或 'no'（存在编造），不要解释。\n\n"
        "参考文档：\n{documents}\n\n"
        "回答：\n{answer}"
    )
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"documents": doc_contents, "answer": answer}).strip().lower()

    passed = result.startswith("yes")
    retry_count = state.get("retry_count", 0)

    return {
        "hallucination_pass": passed,
        "retry_count": retry_count + (0 if passed else 1),
        "steps": state.get("steps", []) + [f"hallucination_check → {'pass' if passed else 'fail'}"],
    }


def fallback(state: RAGState) -> dict:
    """Node 7: 兜底回答"""
    return {
        "answer": "抱歉，知识库中没有找到与您问题相关的可靠信息。请尝试换个问法，或上传更多相关文档。",
        "hallucination_pass": True,
        "steps": state.get("steps", []) + ["fallback"],
    }


# ================================================================
# 条件路由函数
# ================================================================

def route_after_grading(state: RAGState) -> str:
    if state.get("relevant_docs"):
        return "rerank"
    return "fallback"


def route_after_hallucination(state: RAGState) -> str:
    if state.get("hallucination_pass", False):
        return "finish"
    if state.get("retry_count", 0) < 2:
        return "regenerate"
    return "fallback"


# ================================================================
# 构建 LangGraph 状态图
# ================================================================

def build_rag_graph() -> StateGraph:
    """
    构建 RAG 状态图：

    START → transform_query → retrieve → grade_documents
                                              ↓
                                    ┌─── has relevant docs? ───┐
                                    ↓ yes                      ↓ no
                                  rerank                    fallback → END
                                    ↓
                                 generate
                                    ↓
                            check_hallucination
                                    ↓
                          ┌─── grounded? ───────────┐
                          ↓ yes        ↓ no(<2)     ↓ no(≥2)
                         END       generate(retry)  fallback → END
    """
    graph = StateGraph(RAGState)

    graph.add_node("transform_query", transform_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rerank", rerank)
    graph.add_node("generate", generate)
    graph.add_node("check_hallucination", check_hallucination)
    graph.add_node("fallback", fallback)

    graph.add_edge(START, "transform_query")
    graph.add_edge("transform_query", "retrieve")
    graph.add_edge("retrieve", "grade_documents")

    graph.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {"rerank": "rerank", "fallback": "fallback"},
    )

    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", "check_hallucination")

    graph.add_conditional_edges(
        "check_hallucination",
        route_after_hallucination,
        {"finish": END, "regenerate": "generate", "fallback": "fallback"},
    )

    graph.add_edge("fallback", END)

    return graph.compile()


# 编译图（模块级单例）
rag_graph = build_rag_graph()


# ================================================================
# 流式查询图（无幻觉检测，避免已输出内容被重试覆盖）
# ================================================================

def _build_stream_graph():
    """
    流式查询专用图：去掉幻觉检测环节。
    流式场景下 token 已经逐个发送给客户端，如果幻觉检测不通过触发重试，
    用户会看到两段矛盾的输出，体验很差。因此流式模式只走：
    transform_query → retrieve → grade_documents → rerank → generate → END
    """
    graph = StateGraph(RAGState)

    graph.add_node("transform_query", transform_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rerank", rerank)
    graph.add_node("generate", generate)
    graph.add_node("fallback", fallback)

    graph.add_edge(START, "transform_query")
    graph.add_edge("transform_query", "retrieve")
    graph.add_edge("retrieve", "grade_documents")

    graph.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {"rerank": "rerank", "fallback": "fallback"},
    )

    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)

    return graph.compile()


stream_rag_graph = _build_stream_graph()


# ================================================================
# 对外接口
# ================================================================

def query_rag(
    question: str,
    retrieval_mode: Literal["vector", "bm25", "hybrid"] = "hybrid",
    query_transform: Literal["none", "rewrite", "hyde"] = "none",
    use_reranker: bool = False,
    top_k: int = TOP_K,
) -> dict:
    """LangGraph 版 RAG 查询入口"""
    initial_state: RAGState = {
        "question": question,
        "search_query": "",
        "retrieval_mode": retrieval_mode,
        "query_transform": query_transform,
        "use_reranker": use_reranker,
        "top_k": top_k,
        "retrieved_docs": [],
        "relevant_docs": [],
        "answer": "",
        "hallucination_pass": False,
        "retry_count": 0,
        "sources": [],
        "steps": [],
    }

    final_state = rag_graph.invoke(initial_state)

    return {
        "answer": final_state["answer"],
        "sources": final_state.get("sources", []),
        "config": {
            "retrieval_mode": retrieval_mode,
            "query_transform": query_transform,
            "use_reranker": use_reranker,
            "top_k": top_k,
        },
        "graph_steps": final_state.get("steps", []),
    }


async def query_rag_stream(
    question: str,
    retrieval_mode: Literal["vector", "bm25", "hybrid"] = "hybrid",
    query_transform: Literal["none", "rewrite", "hyde"] = "none",
    use_reranker: bool = False,
    top_k: int = TOP_K,
):
    """
    LangGraph 版 RAG 流式查询入口。
    使用专用的 stream_rag_graph（无幻觉检测），避免已输出 token 被重试覆盖。
    """
    import json as _json_stream

    initial_state: RAGState = {
        "question": question,
        "search_query": "",
        "retrieval_mode": retrieval_mode,
        "query_transform": query_transform,
        "use_reranker": use_reranker,
        "top_k": top_k,
        "retrieved_docs": [],
        "relevant_docs": [],
        "answer": "",
        "hallucination_pass": False,
        "retry_count": 0,
        "sources": [],
        "steps": [],
    }

    async for event in stream_rag_graph.astream_events(initial_state, version="v2"):
        kind = event.get("event", "")
        if kind == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                yield f"data: {_json_stream.dumps({'type': 'token', 'content': chunk.content}, ensure_ascii=False)}\n\n"
        elif kind == "on_chain_end" and event.get("name") in (
            "transform_query", "retrieve", "grade_documents", "rerank",
            "generate", "fallback",
        ):
            node_name = event["name"]
            output = event.get("data", {}).get("output", {})
            steps = output.get("steps", [])
            if steps:
                yield f"data: {_json_stream.dumps({'type': 'step', 'node': node_name, 'step': steps[-1]}, ensure_ascii=False)}\n\n"

    yield f"data: {_json_stream.dumps({'type': 'done'})}\n\n"
