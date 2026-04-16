# -*- coding: utf-8 -*-
"""
RAG LangGraph 集成测试 — 基于真实服务
依赖：LLM (localhost:8001)、硅基流动 Embedding、硅基流动 Reranker
运行：pytest tests/test_integration.py -v -s
"""

import os
import sys
import shutil
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from app.config import CHROMA_PERSIST_DIR


# ================================================================
# Fixtures
# ================================================================

INTEGRATION_CHROMA_DIR = "./chroma_db_integration_test"
SAMPLE_FILE = "docs/sample.md"


@pytest.fixture(scope="module", autouse=True)
def setup_isolated_chroma():
    """使用独立的 Chroma 目录，避免污染正式数据"""
    import app.retriever as rt
    # 保存原始值
    original_dir = rt.CHROMA_PERSIST_DIR
    original_vs = rt._vectorstore
    original_bm25 = rt._all_docs_for_bm25

    # 切换到测试目录
    rt.CHROMA_PERSIST_DIR = INTEGRATION_CHROMA_DIR
    rt._vectorstore = None
    rt._all_docs_for_bm25 = []

    yield

    # 恢复
    rt.CHROMA_PERSIST_DIR = original_dir
    rt._vectorstore = original_vs
    rt._all_docs_for_bm25 = original_bm25

    # 清理测试数据
    if os.path.exists(INTEGRATION_CHROMA_DIR):
        shutil.rmtree(INTEGRATION_CHROMA_DIR)


# ================================================================
# 1. 服务连通性测试
# ================================================================

class TestServiceConnectivity:
    """验证真实服务可用"""

    def test_llm_service(self):
        from app.llm import get_llm
        llm = get_llm()
        result = llm.invoke("回答'OK'两个字母")
        assert result.content is not None
        assert len(result.content) > 0
        print(f"  LLM 响应: {result.content[:50]}")

    def test_embedding_service(self):
        from app.llm import get_embeddings
        embeddings = get_embeddings()
        vector = embeddings.embed_query("测试文本")
        assert len(vector) == 1024
        print(f"  Embedding 维度: {len(vector)}")

    def test_reranker_service(self):
        from app.retriever import rerank_documents
        docs = [
            Document(page_content="RAG是检索增强生成技术"),
            Document(page_content="今天天气很好"),
        ]
        result = rerank_documents("什么是RAG", docs, top_k=2)
        assert len(result) == 2
        assert "RAG" in result[0].page_content
        print(f"  Reranker top1 score: {result[0].metadata.get('rerank_score', 'N/A')}")


# ================================================================
# 2. 文档入库集成测试
# ================================================================

class TestIngestIntegration:

    def test_ingest_sample_file(self):
        from app.retriever import ingest_file
        result = ingest_file(SAMPLE_FILE, strategy="recursive", chunk_size=512, chunk_overlap=50)
        assert result["chunks"] > 0
        assert result["filename"] == "sample.md"
        print(f"  入库: {result['chunks']} chunks, 策略: {result['strategy']}")

    def test_smart_ingest_sample_file(self):
        from app.ingest import smart_ingest_file
        result = smart_ingest_file(SAMPLE_FILE)
        assert result["chunks"] > 0
        assert result["strategy"] in ("recursive", "fixed", "semantic")
        assert len(result["graph_steps"]) > 0
        print(f"  智能入库: {result['chunks']} chunks")
        print(f"  选择策略: {result['strategy']}")
        print(f"  分析: {result['analysis']}")
        print(f"  步骤: {result['graph_steps']}")

    def test_vectorstore_has_data_after_ingest(self):
        from app.retriever import get_vectorstore
        vs = get_vectorstore()
        count = vs._collection.count()
        assert count > 0
        print(f"  Chroma 文档数: {count}")


# ================================================================
# 3. 检索集成测试
# ================================================================

class TestRetrievalIntegration:

    def test_vector_retrieval(self):
        from app.retriever import get_vector_retriever
        retriever = get_vector_retriever(top_k=3)
        docs = retriever.invoke("RAG的核心流程是什么")
        assert len(docs) > 0
        print(f"  向量检索: {len(docs)} docs")
        print(f"  Top1: {docs[0].page_content[:80]}...")

    def test_hybrid_retrieval(self):
        from app.retriever import hybrid_retrieve
        docs = hybrid_retrieve("RAG有什么优势", top_k=3)
        assert len(docs) > 0
        print(f"  混合检索: {len(docs)} docs")

    def test_retrieval_relevance(self):
        from app.retriever import hybrid_retrieve
        docs = hybrid_retrieve("向量数据库有哪些", top_k=3)
        assert len(docs) > 0
        all_content = " ".join(d.page_content for d in docs)
        assert any(kw in all_content for kw in ["向量", "Chroma", "Milvus", "embedding", "Embedding"])
        print(f"  相关性验证通过，内容包含向量相关关键词")


# ================================================================
# 4. RAG 查询图端到端集成测试
# ================================================================

class TestQueryGraphIntegration:

    def test_basic_query(self):
        from app.rag_graph import query_rag
        result = query_rag("什么是RAG？", retrieval_mode="hybrid", query_transform="none")
        assert result["answer"]
        assert len(result["answer"]) > 10
        assert "graph_steps" in result
        assert len(result["graph_steps"]) > 0
        print(f"  回答: {result['answer'][:100]}...")
        print(f"  步骤: {result['graph_steps']}")

    def test_query_with_rewrite(self):
        from app.rag_graph import query_rag
        result = query_rag("RAG咋回事", retrieval_mode="hybrid", query_transform="rewrite")
        assert result["answer"]
        steps_str = " ".join(result["graph_steps"])
        assert "rewrite" in steps_str.lower()
        print(f"  改写后回答: {result['answer'][:100]}...")
        print(f"  步骤: {result['graph_steps']}")

    def test_query_with_hyde(self):
        from app.rag_graph import query_rag
        result = query_rag("RAG的优势", retrieval_mode="hybrid", query_transform="hyde")
        assert result["answer"]
        steps_str = " ".join(result["graph_steps"])
        assert "hyde" in steps_str.lower()
        print(f"  HyDE 回答: {result['answer'][:100]}...")

    def test_query_vector_only(self):
        from app.rag_graph import query_rag
        result = query_rag("Embedding模型有哪些", retrieval_mode="vector")
        assert result["answer"]
        assert result["config"]["retrieval_mode"] == "vector"
        print(f"  向量检索回答: {result['answer'][:100]}...")

    def test_query_with_reranker(self):
        from app.rag_graph import query_rag
        result = query_rag("RAG的核心流程", retrieval_mode="hybrid", use_reranker=True)
        assert result["answer"]
        assert result["config"]["use_reranker"] is True
        print(f"  Reranker 回答: {result['answer'][:100]}...")
        print(f"  步骤: {result['graph_steps']}")

    def test_query_irrelevant_question(self):
        from app.rag_graph import query_rag
        result = query_rag("量子计算机的工作原理是什么？")
        assert result["answer"]
        print(f"  不相关问题回答: {result['answer'][:100]}...")
        print(f"  步骤: {result['graph_steps']}")

    def test_query_returns_sources(self):
        from app.rag_graph import query_rag
        result = query_rag("RAG有什么优势")
        if "抱歉" not in result["answer"]:
            assert len(result["sources"]) > 0
            assert "source" in result["sources"][0]
            print(f"  来源数: {len(result['sources'])}")
            print(f"  来源: {result['sources'][0]['source']}")


# ================================================================
# 5. 幻觉检测 & 路由集成测试
# ================================================================

class TestHallucinationIntegration:

    def test_grounded_answer_passes(self):
        from app.rag_graph import check_hallucination
        docs = [Document(page_content="RAG是检索增强生成技术，结合了信息检索与大语言模型。")]
        state = {
            "question": "什么是RAG",
            "answer": "RAG是检索增强生成技术，它结合了信息检索与大语言模型。",
            "relevant_docs": docs,
            "retry_count": 0,
            "steps": [],
        }
        result = check_hallucination(state)
        assert result["hallucination_pass"] is True
        print(f"  有依据回答: 幻觉检测通过 ✓")

    def test_fabricated_answer_fails(self):
        from app.rag_graph import check_hallucination
        docs = [Document(page_content="RAG是检索增强生成技术，结合了信息检索与大语言模型。")]
        state = {
            "question": "什么是RAG",
            "answer": "RAG是一种量子计算加速技术，由NASA在2019年发明，主要用于太空探索。",
            "relevant_docs": docs,
            "retry_count": 0,
            "steps": [],
        }
        result = check_hallucination(state)
        assert result["hallucination_pass"] is False
        print(f"  编造回答: 幻觉检测拦截 ✓")


# ================================================================
# 6. 文档评估集成测试
# ================================================================

class TestGradeDocumentsIntegration:

    def test_relevant_doc_graded_yes(self):
        from app.rag_graph import grade_documents
        docs = [Document(page_content="RAG（Retrieval-Augmented Generation）即检索增强生成，是一种将信息检索与大语言模型结合的技术。")]
        state = {
            "question": "什么是RAG？",
            "retrieved_docs": docs,
            "steps": [],
        }
        result = grade_documents(state)
        assert len(result["relevant_docs"]) == 1
        print(f"  相关文档评估: 1/1 relevant ✓")

    def test_irrelevant_doc_graded_no(self):
        from app.rag_graph import grade_documents
        docs = [Document(page_content="今天北京的天气是晴天，最高温度25度，适合户外活动。")]
        state = {
            "question": "什么是RAG？",
            "retrieved_docs": docs,
            "steps": [],
        }
        result = grade_documents(state)
        assert len(result["relevant_docs"]) == 0
        print(f"  不相关文档评估: 0/1 relevant ✓")
