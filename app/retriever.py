# 向量存储、检索器、Reranker
import os
import logging
import hashlib
import threading
from typing import Literal

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document

from app.config import (
    CHROMA_COLLECTION, CHROMA_PERSIST_DIR, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K,
    RERANKER_API_KEY, RERANKER_BASE_URL, RERANKER_MODEL,
)
from app.llm import get_embeddings

logger = logging.getLogger(__name__)


# ================================================================
# 文档加载 & 切分
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
# 向量存储 & 入库（线程安全）
# ================================================================

_vectorstore = None
_all_docs_for_bm25: list[Document] = []
_store_lock = threading.Lock()


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        with _store_lock:
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
    with _store_lock:
        vs.add_documents(documents)
        _all_docs_for_bm25 = _all_docs_for_bm25 + documents  # 创建新列表，避免读端竞争
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

def get_vector_retriever(top_k: int = TOP_K):
    return get_vectorstore().as_retriever(search_kwargs={"k": top_k})


def get_bm25_retriever(top_k: int = TOP_K):
    global _all_docs_for_bm25
    with _store_lock:
        local_docs = _all_docs_for_bm25  # 快照引用，读取安全
    if not local_docs:
        vs = get_vectorstore()
        results = vs.get()
        if results and results["documents"]:
            local_docs = [
                Document(page_content=doc, metadata=meta)
                for doc, meta in zip(results["documents"], results["metadatas"])
            ]
            with _store_lock:
                _all_docs_for_bm25 = local_docs
    if not local_docs:
        return None
    return BM25Retriever.from_documents(local_docs, k=top_k)


def hybrid_retrieve(query: str, top_k: int = TOP_K) -> list[Document]:
    vector_docs = get_vector_retriever(top_k).invoke(query)
    bm25 = get_bm25_retriever(top_k)
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
# Reranker（带重试和超时降级）
# ================================================================

def _build_reranker_session() -> requests.Session:
    """构建带自动重试的 requests Session"""
    session = requests.Session()
    retries = Retry(total=2, backoff_factor=0.5, status_forcelist=[502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_reranker_session = _build_reranker_session()


def rerank_documents(query: str, documents: list[Document], top_k: int = TOP_K) -> list[Document]:
    if not documents:
        return documents
    try:
        resp = _reranker_session.post(
            RERANKER_BASE_URL,
            headers={"Authorization": f"Bearer {RERANKER_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": RERANKER_MODEL,
                "query": query,
                "documents": [doc.page_content for doc in documents],
                "top_n": min(top_k, len(documents)),
            },
            timeout=15,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Reranker 请求失败，降级使用原始顺序: {e}")
        return documents

    results = resp.json().get("results", [])
    reranked = []
    for item in sorted(results, key=lambda x: x["relevance_score"], reverse=True):
        doc = documents[item["index"]]
        doc.metadata["rerank_score"] = item["relevance_score"]
        reranked.append(doc)
    return reranked


# ================================================================
# 切分策略对比
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
