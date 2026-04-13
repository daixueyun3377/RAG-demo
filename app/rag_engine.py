# RAG 核心引擎 - LangChain 版
# 覆盖：DocumentLoader, TextSplitter(多策略), Chroma, BM25混合检索, Reranker, RetrievalQA
import os
from typing import Literal

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langfuse import Langfuse
from langfuse.callback import CallbackHandler as LangfuseCallbackHandler

from app.config import *


# ========== 初始化 ==========

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
    if LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY:
        return LangfuseCallbackHandler(
            secret_key=LANGFUSE_SECRET_KEY,
            public_key=LANGFUSE_PUBLIC_KEY,
            host=LANGFUSE_HOST,
        )
    return None


# ========== 文档加载 (DocumentLoader) ==========

def load_file(file_path: str) -> list[Document]:
    """使用 LangChain DocumentLoader 加载文档"""
    if file_path.endswith(".md"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")
    return loader.load()


def load_directory(dir_path: str) -> list[Document]:
    """批量加载目录下所有文档"""
    loader = DirectoryLoader(dir_path, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    return loader.load()


# ========== 文本切分 (TextSplitter) ==========

def split_documents(
    documents: list[Document],
    strategy: Literal["fixed", "recursive"] = "recursive",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    多策略文本切分
    - fixed: 固定长度切分 (CharacterTextSplitter)
    - recursive: 递归切分 (RecursiveCharacterTextSplitter) — 推荐
    """
    if strategy == "fixed":
        splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n",
        )
    elif strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""],
        )
    else:
        raise ValueError(f"未知切分策略: {strategy}")

    return splitter.split_documents(documents)


# ========== 向量存储 (Chroma) ==========

_vectorstore = None
_all_docs_for_bm25 = []  # BM25 需要维护文档列表


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
    """将切分后的文档存入 Chroma"""
    global _all_docs_for_bm25
    vs = get_vectorstore()
    vs.add_documents(documents)
    _all_docs_for_bm25.extend(documents)
    return len(documents)


# ========== 文件入库完整流程 ==========

def ingest_file(
    file_path: str,
    strategy: str = "recursive",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> dict:
    """加载文件 → 切分 → 入库"""
    docs = load_file(file_path)
    chunks = split_documents(docs, strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    count = ingest_documents(chunks)
    return {
        "filename": os.path.basename(file_path),
        "chunks": count,
        "strategy": strategy,
        "chunk_size": chunk_size,
    }


# ========== 检索器 (Retriever) ==========

def get_vector_retriever(top_k: int = TOP_K):
    """纯向量检索"""
    vs = get_vectorstore()
    return vs.as_retriever(search_kwargs={"k": top_k})


def get_bm25_retriever(top_k: int = TOP_K):
    """BM25 关键词检索"""
    global _all_docs_for_bm25
    if not _all_docs_for_bm25:
        # 从 Chroma 加载已有文档
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


def get_hybrid_retriever(top_k: int = TOP_K):
    """混合检索：向量 + BM25，RRF 融合"""
    vector_retriever = get_vector_retriever(top_k)
    bm25_retriever = get_bm25_retriever(top_k)
    if bm25_retriever is None:
        return vector_retriever
    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.6, 0.4],  # 向量权重 60%，BM25 权重 40%
    )


# ========== Query 改写 / HyDE ==========

def rewrite_query(query: str) -> str:
    """Query 改写：让 LLM 优化用户问题，使其更适合检索"""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        "你是一个搜索查询优化助手。请将用户的问题改写为更适合在知识库中检索的查询语句。"
        "只输出改写后的查询，不要解释。\n\n用户问题：{question}"
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": query})


def hyde_query(query: str) -> str:
    """HyDE：让 LLM 先生成一个假设性回答，用这个回答去检索（而不是用原始问题）"""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        "请针对以下问题写一段简短的回答（约100字），即使你不确定也请尝试回答。"
        "这段回答将用于检索相关文档。\n\n问题：{question}"
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": query})


# ========== RAG Chain ==========

RAG_PROMPT = ChatPromptTemplate.from_template(
    """你是一个知识库问答助手。请基于以下检索到的内容回答用户问题。
如果检索内容中没有相关信息，请诚实说明。回答时标注信息来源。

检索到的内容：
{context}

用户问题：{question}

回："""
)


def format_docs(docs: list[Document]) -> str:
    """格式化检索结果"""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "未知")
        formatted.append(f"[来源: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def query_rag(
    question: str,
    retrieval_mode: Literal["vector", "bm25", "hybrid"] = "hybrid",
    query_transform: Literal["none", "rewrite", "hyde"] = "none",
    top_k: int = TOP_K,
) -> dict:
    """
    完整 RAG 查询流程
    - retrieval_mode: vector(纯向量) / bm25(纯关键词) / hybrid(混合)
    - query_transform: none(原始) / rewrite(改写) / hyde(假设性回答检索)
    """
    langfuse_handler = get_langfuse_handler()
    callbacks = [langfuse_handler] if langfuse_handler else []

    # 1. Query 变换
    search_query = question
    transform_info = "none"
    if query_transform == "rewrite":
        search_query = rewrite_query(question)
        transform_info = f"rewrite → {search_query}"
    elif query_transform == "hyde":
        search_query = hyde_query(question)
        transform_info = f"hyde → {search_query[:100]}..."

    # 2. 选择检索器
    if retrieval_mode == "vector":
        retriever = get_vector_retriever(top_k)
    elif retrieval_mode == "bm25":
        retriever = get_bm25_retriever(top_k)
        if retriever is None:
            return {"answer": "知识库为空，请先上传文档", "sources": [], "config": {}}
    else:
        retriever = get_hybrid_retriever(top_k)

    # 3. 检索
    retrieved_docs = retriever.invoke(search_query)

    # 4. 生成回答（LCEL chain）
    llm = get_llm()
    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke(
        {"context": format_docs(retrieved_docs), "question": question},
        config={"callbacks": callbacks},
    )

    # 5. 构造返回
    sources = []
    for doc in retrieved_docs:
        sources.append({
            "text": doc.page_content[:200],
            "source": doc.metadata.get("source", "未知"),
        })

    return {
        "answer": answer,
        "sources": sources,
        "config": {
            "retrieval_mode": retrieval_mode,
            "query_transform": transform_info,
            "top_k": top_k,
        },
    }


# ========== 切分策略对比工具 ==========

def compare_chunk_strategies(file_path: str) -> dict:
    """对比不同切分策略和 chunk_size 的效果"""
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
                "avg_length": sum(len(c.page_content) for c in chunks) / max(len(chunks), 1),
                "sample": chunks[0].page_content[:200] if chunks else "",
            }

    return results
