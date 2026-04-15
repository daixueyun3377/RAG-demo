# RAG 核心引擎 - LangGraph 版
# 将 RAG 全流程建模为显式状态图：
#   query变换 → 检索 → 文档评估 → (重排序) → 生成 → 幻觉检测 → 输出/重试
#
# 相比 LangChain LCEL 版本的核心升级：
#   1. 文档相关性评估 (grade_documents) — LLM 判断每篇文档是否与问题相关
#   2. 幻觉检测 (check_hallucination) — LLM 验证回答是否有文档依据
#   3. 条件路由 — 无相关文档时走 fallback，检测到幻觉时自动重试
#   4. 全流程可视化 — LangGraph 原生支持 Mermaid 图导出

import os
import logging
import hashlib
import requests
from typing import Literal, TypedDict, Annotated

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

try:
    from langfuse.callback import CallbackHandler as LangfuseCallbackHandler
except (ImportError, ModuleNotFoundError):
    LangfuseCallbackHandler = None

from app.config import *

logger = logging.getLogger(__name__)


# ================================================================
# 初始化
# ================================================================

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


# ================================================================
# 文档加载 & 切分（与原版一致，入库流程不变）
# ================================================================

def load_file(file_path: str) -> list[Document]:
    if file_path.endswith((".md", ".txt")):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")
    return loader.load()


def load_directory(dir_path: str) -> list[Document]:
    loader = DirectoryLoader(dir_path, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    return loader.load()


def split_documents(
    documents: list[Document],
    strategy: Literal["fixed", "recursive", "semantic"] = "recursive",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    if strategy == "fixed":
        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator="\n")
    elif strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""],
        )
    elif strategy == "semantic":
        splitter = SemanticChunker(
            embeddings=get_embeddings(),
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=80,
        )
    else:
        raise ValueError(f"未知切分策略: {strategy}")
    return splitter.split_documents(documents)


# ================================================================
# 向量存储 & 入库
# ================================================================

_vectorstore = None
_all_docs_for_bm25: list[Document] = []


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION,
            embedding_function=get_embeddings(),
            persist_directory=CHROMA_PERSIST_DIR,
        )
    return _vectorstore


def ingest_documents(documents: list[Document]) -> int:
    global _all_docs_for_bm25
    vs = get_vectorstore()
    vs.add_documents(documents)
    _all_docs_for_bm25.extend(documents)
    return len(documents)


def ingest_file(
    file_path: str,
    strategy: str = "recursive",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> dict:
    docs = load_file(file_path)
    chunks = split_documents(docs, strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    count = ingest_documents(chunks)
    return {"filename": os.path.basename(file_path), "chunks": count, "strategy": strategy, "chunk_size": chunk_size}


# ================================================================
# 检索器
# ================================================================

def _get_vector_retriever(top_k: int = TOP_K):
    return get_vectorstore().as_retriever(search_kwargs={"k": top_k})


def _get_bm25_retriever(top_k: int = TOP_K):
    global _all_docs_for_bm25
    if not _all_docs_for_bm25:
        vs = get_vectorstore()
        results = vs.get()
        if results and results["documents"]:
            _all_docs_for_bm25 = [
                Document(page_content=doc, metadata=meta)
                for doc, meta in zip(results["documents"], results["metadatas"])
            ]
    if not _all_docs_for_bm25:
        return None
    return BM25Retriever.from_documents(_all_docs_for_bm25, k=top_k)


def _hybrid_retrieve(query: str, top_k: int = TOP_K) -> list[Document]:
    vector_docs = _get_vector_retriever(top_k).invoke(query)
    bm25 = _get_bm25_retriever(top_k)
    bm25_docs = bm25.invoke(query) if bm25 else []

    rrf_k = 60
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(vector_docs):
        key = hashlib.md5(doc.page_content.encode()).hexdigest()
        scores[key] = scores.get(key, 0) + 0.6 * (1.0 / (rrf_k + rank))
        doc_map[key] = doc

    for rank, doc in enumerate(bm25_docs):
        key = hashlib.md5(doc.page_content.encode()).hexdigest()
        scores[key] = scores.get(key, 0) + 0.4 * (1.0 / (rrf_k + rank))
        doc_map[key] = doc

    sorted_keys = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [doc_map[k] for k in sorted_keys]


# ================================================================
# Reranker
# ================================================================

def _rerank_documents(query: str, documents: list[Document], top_k: int = TOP_K) -> list[Document]:
    if not documents:
        return documents
    resp = requests.post(
        RERANKER_BASE_URL,
        headers={"Authorization": f"Bearer {RERANKER_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": RERANKER_MODEL,
            "query": query,
            "documents": [doc.page_content for doc in documents],
            "top_n": min(top_k, len(documents)),
        },
        timeout=30,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    reranked = []
    for item in sorted(results, key=lambda x: x["relevance_score"], reverse=True):
        doc = documents[item["index"]]
        doc.metadata["rerank_score"] = item["relevance_score"]
        reranked.append(doc)
    return reranked


# ================================================================
# LangGraph State 定义
# ================================================================

class RAGState(TypedDict):
    """RAG 图的全局状态"""
    question: str                # 用户原始问题
    search_query: str            # 变换后的检索 query
    retrieval_mode: str          # vector / bm25 / hybrid
    query_transform: str         # none / rewrite / hyde
    use_reranker: bool
    top_k: int
    retrieved_docs: list         # 检索到的文档
    relevant_docs: list          # 经过相关性评估后的文档
    answer: str                  # 生成的回答
    hallucination_pass: bool     # 幻觉检测是否通过
    retry_count: int             # 重试次数
    sources: list                # 来源信息
    steps: list                  # 执行步骤记录（用于调试）


# ================================================================
# Graph Nodes — 每个 node 接收 state，返回 state 的部分更新
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
        docs = _get_vector_retriever(top_k).invoke(query)
    elif mode == "bm25":
        retriever = _get_bm25_retriever(top_k)
        docs = retriever.invoke(query) if retriever else []
    else:
        docs = _hybrid_retrieve(query, top_k)

    return {
        "retrieved_docs": docs,
        "steps": state.get("steps", []) + [f"retrieve({mode}) → {len(docs)} docs"],
    }


def grade_documents(state: RAGState) -> dict:
    """Node 3: 文档相关性评估 — LLM 逐篇判断文档是否与问题相关（LangGraph 经典模式）"""
    question = state["question"]
    docs = state["retrieved_docs"]
    llm = get_llm()

    grade_prompt = ChatPromptTemplate.from_template(
        "你是一个文档相关性评估专家。判断以下文档是否与用户问题相关。\n"
        "只回答 'yes' 或 'no'，不要解释。\n\n"
        "用户问题：{question}\n\n"
        "文档内容：{document}"
    )
    chain = grade_prompt | llm | StrOutputParser()

    relevant = []
    for doc in docs:
        score = chain.invoke({"question": question, "document": doc.page_content}).strip().lower()
        if score.startswith("yes"):
            relevant.append(doc)

    return {
        "relevant_docs": relevant,
        "steps": state.get("steps", []) + [f"grade_documents → {len(relevant)}/{len(docs)} relevant"],
    }


def rerank(state: RAGState) -> dict:
    """Node 4: Reranker 重排序（可选）"""
    question = state["question"]
    docs = state["relevant_docs"]
    top_k = state.get("top_k", TOP_K)

    if not state.get("use_reranker", False) or not docs:
        return {"relevant_docs": docs, "steps": state.get("steps", []) + ["rerank_skipped"]}

    try:
        reranked = _rerank_documents(question, docs, top_k)
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

    # 格式化文档
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
    """Node 7: 兜底回答 — 无相关文档或幻觉检测多次失败时触发"""
    return {
        "answer": "抱歉，知识库中没有找到与您问题相关的可靠信息。请尝试换个问法，或上传更多相关文档。",
        "hallucination_pass": True,  # 标记为通过，终止循环
        "steps": state.get("steps", []) + ["fallback"],
    }


# ================================================================
# 条件路由函数
# ================================================================

def route_after_grading(state: RAGState) -> str:
    """文档评估后的路由：有相关文档 → rerank，无相关文档 → fallback"""
    if state.get("relevant_docs"):
        return "rerank"
    return "fallback"


def route_after_hallucination(state: RAGState) -> str:
    """幻觉检测后的路由：通过 → 结束，未通过且重试<2次 → 重新生成，否则 → fallback"""
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

    # 添加节点
    graph.add_node("transform_query", transform_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rerank", rerank)
    graph.add_node("generate", generate)
    graph.add_node("check_hallucination", check_hallucination)
    graph.add_node("fallback", fallback)

    # 添加边
    graph.add_edge(START, "transform_query")
    graph.add_edge("transform_query", "retrieve")
    graph.add_edge("retrieve", "grade_documents")

    # 条件边：文档评估后
    graph.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {"rerank": "rerank", "fallback": "fallback"},
    )

    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", "check_hallucination")

    # 条件边：幻觉检测后
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
# 对外接口
# ================================================================

def query_rag(
    question: str,
    retrieval_mode: Literal["vector", "bm25", "hybrid"] = "hybrid",
    query_transform: Literal["none", "rewrite", "hyde"] = "none",
    use_reranker: bool = False,
    top_k: int = TOP_K,
) -> dict:
    """
    LangGraph 版 RAG 查询入口
    接口签名与原版完全一致，内部走状态图
    """
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

    # 执行图
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


# ================================================================
# 切分策略对比（不变）
# ================================================================

def compare_chunk_strategies(file_path: str) -> dict:
    docs = load_file(file_path)
    results = {}

    for strategy in ["fixed", "recursive"]:
        for size in [256, 512, 1024]:
            chunks = split_documents(docs, strategy=strategy, chunk_size=size, chunk_overlap=size // 10)
            key = f"{strategy}_{size}"
            results[key] = {
                "strategy": strategy,
                "chunk_size": size,
                "num_chunks": len(chunks),
                "avg_length": round(sum(len(c.page_content) for c in chunks) / max(len(chunks), 1), 1),
                "sample": chunks[0].page_content[:200] if chunks else "",
            }

    try:
        semantic_chunks = split_documents(docs, strategy="semantic")
        results["semantic"] = {
            "strategy": "semantic",
            "chunk_size": "auto (embedding-based)",
            "num_chunks": len(semantic_chunks),
            "avg_length": round(sum(len(c.page_content) for c in semantic_chunks) / max(len(semantic_chunks), 1), 1),
            "sample": semantic_chunks[0].page_content[:200] if semantic_chunks else "",
        }
    except Exception as e:
        results["semantic"] = {"strategy": "semantic", "error": str(e)}

    return results
