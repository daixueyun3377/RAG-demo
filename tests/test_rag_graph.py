# -*- coding: utf-8 -*-
"""
RAG LangGraph 方案全面测试
覆盖：
  1. 状态定义 & 图结构
  2. 各节点单元测试（mock LLM/Embedding）
  3. 条件路由逻辑
  4. 查询图端到端流程
  5. 智能入库图端到端流程
  6. 文档分析规则引擎
  7. 切分质量验证
  8. 降级重试机制
  9. 边界情况 & 异常处理
  10. FastAPI API 层测试
  11. Retriever 模块测试
  12. LLM 模块测试
  13. 流式查询测试
"""

import os
import sys
import json
import pytest
import asyncio
from io import BytesIO
from unittest.mock import patch, MagicMock, AsyncMock, PropertyMock
from langchain_core.documents import Document

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag_graph import (
    transform_query,
    retrieve,
    grade_documents,
    rerank,
    generate,
    check_hallucination,
    fallback,
    _parse_grade_result,
    route_after_grading,
    route_after_hallucination,
    build_rag_graph,
    _build_stream_graph,
    RAGState,
    query_rag,
    query_rag_stream,
)

from app.ingest import (
    build_ingest_graph,
    IngestState,
    load_document_node,
    analyze_document_node,
    split_document_node,
    validate_chunks_node,
    fallback_strategy_node,
    store_document_node,
    route_after_validation,
    _analyze_doc_features,
    _MIN_CHUNKS,
    _MAX_EMPTY_RATIO,
    _MIN_AVG_LENGTH,
    _MAX_AVG_LENGTH,
    smart_ingest_file,
)


# ================================================================
# Fixtures & Helpers
# ================================================================

def _make_rag_state(**overrides) -> RAGState:
    """构造一个默认的 RAGState，可按需覆盖字段"""
    base: RAGState = {
        "question": "什么是RAG？",
        "search_query": "",
        "retrieval_mode": "hybrid",
        "query_transform": "none",
        "use_reranker": False,
        "top_k": 5,
        "retrieved_docs": [],
        "relevant_docs": [],
        "answer": "",
        "hallucination_pass": False,
        "retry_count": 0,
        "sources": [],
        "steps": [],
    }
    base.update(overrides)
    return base


def _make_ingest_state(**overrides) -> IngestState:
    """构造一个默认的 IngestState"""
    base: IngestState = {
        "file_path": "docs/sample.md",
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
    base.update(overrides)
    return base


def _make_docs(n=3, content_prefix="文档内容"):
    """快速生成 n 个 Document"""
    return [
        Document(page_content=f"{content_prefix} {i}: RAG是检索增强生成技术。", metadata={"source": f"test_{i}.md"})
        for i in range(n)
    ]


class FakeLLMResponse:
    """模拟 LLM 的 invoke 返回"""
    def __init__(self, text):
        self.content = text


# ================================================================
# 1. 图结构测试
# ================================================================

class TestGraphStructure:

    def test_rag_graph_compiles(self):
        graph = build_rag_graph()
        assert graph is not None

    def test_ingest_graph_compiles(self):
        graph = build_ingest_graph()
        assert graph is not None

    def test_rag_graph_has_expected_nodes(self):
        graph = build_rag_graph()
        graph_obj = graph.get_graph()
        node_ids = set(graph_obj.nodes.keys())
        expected = {"transform_query", "retrieve", "grade_documents", "rerank",
                    "generate", "check_hallucination", "fallback"}
        assert expected.issubset(node_ids), f"缺少节点: {expected - node_ids}"

    def test_ingest_graph_has_expected_nodes(self):
        graph = build_ingest_graph()
        graph_obj = graph.get_graph()
        node_ids = set(graph_obj.nodes.keys())
        expected = {"load_document", "analyze_document", "split_document",
                    "validate_chunks", "fallback_strategy", "store_document"}
        assert expected.issubset(node_ids), f"缺少节点: {expected - node_ids}"

    def test_stream_graph_compiles(self):
        """流式查询图能正常编译"""
        graph = _build_stream_graph()
        assert graph is not None

    def test_stream_graph_has_no_hallucination_node(self):
        """流式图不包含幻觉检测节点"""
        graph = _build_stream_graph()
        graph_obj = graph.get_graph()
        node_ids = set(graph_obj.nodes.keys())
        assert "check_hallucination" not in node_ids
        assert "transform_query" in node_ids
        assert "generate" in node_ids


# ================================================================
# 2. Query 变换节点测试
# ================================================================

class TestTransformQuery:

    def test_passthrough_mode(self):
        state = _make_rag_state(question="什么是RAG？", query_transform="none")
        result = transform_query(state)
        assert result["search_query"] == "什么是RAG？"
        assert "query_passthrough" in result["steps"][0]

    @patch("app.rag_graph.get_llm")
    def test_rewrite_mode(self, mock_get_llm):
        mock_chain_result = "RAG 检索增强生成 技术原理"
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        state = _make_rag_state(question="什么是RAG？", query_transform="rewrite")
        with patch("app.rag_graph.ChatPromptTemplate") as mock_prompt_cls:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_chain_result
            mock_prompt_cls.from_template.return_value.__or__ = MagicMock(return_value=MagicMock(
                __or__=MagicMock(return_value=mock_chain)
            ))
            result = transform_query(state)
            assert result["search_query"] == mock_chain_result
            assert "query_rewrite" in result["steps"][0]

    @patch("app.rag_graph.get_llm")
    def test_hyde_mode(self, mock_get_llm):
        mock_chain_result = "RAG是一种将检索与生成结合的技术..."
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        state = _make_rag_state(question="什么是RAG？", query_transform="hyde")
        with patch("app.rag_graph.ChatPromptTemplate") as mock_prompt_cls:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_chain_result
            mock_prompt_cls.from_template.return_value.__or__ = MagicMock(return_value=MagicMock(
                __or__=MagicMock(return_value=mock_chain)
            ))
            result = transform_query(state)
            assert result["search_query"] == mock_chain_result
            assert "hyde" in result["steps"][0]

    def test_passthrough_preserves_existing_steps(self):
        """透传模式保留已有步骤"""
        state = _make_rag_state(question="test", query_transform="none", steps=["prev_step"])
        result = transform_query(state)
        assert len(result["steps"]) == 2
        assert result["steps"][0] == "prev_step"


# ================================================================
# 3. 检索节点测试
# ================================================================

class TestRetrieve:

    @patch("app.rag_graph.get_vector_retriever")
    def test_vector_mode(self, mock_retriever_fn):
        docs = _make_docs(2)
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = docs
        mock_retriever_fn.return_value = mock_retriever

        state = _make_rag_state(search_query="RAG", retrieval_mode="vector")
        result = retrieve(state)
        assert len(result["retrieved_docs"]) == 2
        assert "retrieve(vector)" in result["steps"][0]

    @patch("app.rag_graph.get_bm25_retriever")
    def test_bm25_mode(self, mock_bm25_fn):
        docs = _make_docs(2)
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = docs
        mock_bm25_fn.return_value = mock_retriever

        state = _make_rag_state(search_query="RAG", retrieval_mode="bm25")
        result = retrieve(state)
        assert len(result["retrieved_docs"]) == 2
        assert "retrieve(bm25)" in result["steps"][0]

    @patch("app.rag_graph.hybrid_retrieve")
    def test_hybrid_mode(self, mock_hybrid):
        docs = _make_docs(3)
        mock_hybrid.return_value = docs

        state = _make_rag_state(search_query="RAG", retrieval_mode="hybrid")
        result = retrieve(state)
        assert len(result["retrieved_docs"]) == 3
        assert "retrieve(hybrid)" in result["steps"][0]

    @patch("app.rag_graph.get_bm25_retriever")
    def test_bm25_returns_none(self, mock_bm25_fn):
        mock_bm25_fn.return_value = None
        state = _make_rag_state(search_query="RAG", retrieval_mode="bm25")
        result = retrieve(state)
        assert result["retrieved_docs"] == []

    @patch("app.rag_graph.hybrid_retrieve")
    def test_retrieve_uses_custom_top_k(self, mock_hybrid):
        """检索使用自定义 top_k"""
        mock_hybrid.return_value = []
        state = _make_rag_state(search_query="RAG", retrieval_mode="hybrid", top_k=10)
        retrieve(state)
        mock_hybrid.assert_called_once_with("RAG", 10)


# ================================================================
# 4. 文档评估节点测试
# ================================================================

class TestGradeDocuments:

    @patch("app.rag_graph.get_llm")
    def test_all_relevant(self, mock_get_llm):
        docs = _make_docs(3)
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        with patch("app.rag_graph.ChatPromptTemplate") as mock_prompt_cls:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "yes\nyes\nyes"
            mock_prompt_cls.from_template.return_value.__or__ = MagicMock(return_value=MagicMock(
                __or__=MagicMock(return_value=mock_chain)
            ))
            state = _make_rag_state(retrieved_docs=docs)
            result = grade_documents(state)
            assert len(result["relevant_docs"]) == 3
            assert "3/3 relevant" in result["steps"][0]

    @patch("app.rag_graph.get_llm")
    def test_partial_relevant(self, mock_get_llm):
        docs = _make_docs(3)
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        with patch("app.rag_graph.ChatPromptTemplate") as mock_prompt_cls:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "yes\nno\nyes"
            mock_prompt_cls.from_template.return_value.__or__ = MagicMock(return_value=MagicMock(
                __or__=MagicMock(return_value=mock_chain)
            ))
            state = _make_rag_state(retrieved_docs=docs)
            result = grade_documents(state)
            assert len(result["relevant_docs"]) == 2
            assert "2/3 relevant" in result["steps"][0]

    @patch("app.rag_graph.get_llm")
    def test_none_relevant(self, mock_get_llm):
        docs = _make_docs(2)
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        with patch("app.rag_graph.ChatPromptTemplate") as mock_prompt_cls:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "no\nno"
            mock_prompt_cls.from_template.return_value.__or__ = MagicMock(return_value=MagicMock(
                __or__=MagicMock(return_value=mock_chain)
            ))
            state = _make_rag_state(retrieved_docs=docs)
            result = grade_documents(state)
            assert len(result["relevant_docs"]) == 0

    def test_empty_docs(self):
        state = _make_rag_state(retrieved_docs=[])
        result = grade_documents(state)
        assert len(result["relevant_docs"]) == 0
        assert "no docs" in result["steps"][0]

    @patch("app.rag_graph.get_llm")
    def test_llm_returns_fewer_lines(self, mock_get_llm):
        docs = _make_docs(3)
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        with patch("app.rag_graph.ChatPromptTemplate") as mock_prompt_cls:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "no\nyes"
            mock_prompt_cls.from_template.return_value.__or__ = MagicMock(return_value=MagicMock(
                __or__=MagicMock(return_value=mock_chain)
            ))
            state = _make_rag_state(retrieved_docs=docs)
            result = grade_documents(state)
            assert len(result["relevant_docs"]) == 2


# ================================================================
# 4b. _parse_grade_result 鲁棒性测试
# ================================================================

class TestParseGradeResult:

    def test_standard_format(self):
        assert _parse_grade_result("yes\nno\nyes", 3) == [True, False, True]

    def test_with_prefix_brackets(self):
        assert _parse_grade_result("[文档0] yes\n[文档1] no\n[文档2] yes", 3) == [True, False, True]

    def test_with_numbered_prefix(self):
        assert _parse_grade_result("1. yes\n2. no\n3. yes", 3) == [True, False, True]

    def test_with_extra_text(self):
        assert _parse_grade_result("yes, this is relevant\nno, not related", 2) == [True, False]

    def test_fewer_lines_than_docs(self):
        result = _parse_grade_result("yes\nno", 4)
        assert result == [True, False, True, True]

    def test_empty_result(self):
        result = _parse_grade_result("", 3)
        assert result == [True, True, True]

    def test_with_blank_lines(self):
        result = _parse_grade_result("yes\n\nno\n\nyes", 3)
        assert result == [True, False, True]

    def test_case_insensitive(self):
        assert _parse_grade_result("Yes\nNO\nYES", 3) == [True, False, True]

    def test_with_colon_prefix(self):
        """带冒号前缀格式"""
        assert _parse_grade_result("文档0: yes\n文档1: no", 2) == [True, False]

    def test_more_lines_than_docs(self):
        """LLM 返回行数超过文档数，只取前 N 个"""
        result = _parse_grade_result("yes\nno\nyes\nyes\nno", 3)
        assert result == [True, False, True]

    def test_unparseable_lines_skipped(self):
        """无法解析的行被跳过"""
        result = _parse_grade_result("yes\n这是一段解释\nno", 2)
        assert result == [True, False]


# ================================================================
# 5. Rerank 节点测试
# ================================================================

class TestRerank:

    def test_rerank_skipped_when_disabled(self):
        docs = _make_docs(3)
        state = _make_rag_state(relevant_docs=docs, use_reranker=False)
        result = rerank(state)
        assert result["relevant_docs"] == docs
        assert "rerank_skipped" in result["steps"][0]

    def test_rerank_skipped_when_no_docs(self):
        state = _make_rag_state(relevant_docs=[], use_reranker=True)
        result = rerank(state)
        assert result["relevant_docs"] == []

    @patch("app.rag_graph.rerank_documents")
    def test_rerank_success(self, mock_rerank):
        docs = _make_docs(3)
        reranked = list(reversed(docs))
        mock_rerank.return_value = reranked

        state = _make_rag_state(relevant_docs=docs, use_reranker=True)
        result = rerank(state)
        assert result["relevant_docs"] == reranked
        assert "rerank" in result["steps"][0]

    @patch("app.rag_graph.rerank_documents")
    def test_rerank_failure_fallback(self, mock_rerank):
        docs = _make_docs(3)
        mock_rerank.side_effect = Exception("Reranker API timeout")

        state = _make_rag_state(relevant_docs=docs, use_reranker=True)
        result = rerank(state)
        assert result["relevant_docs"] == docs
        assert "rerank_failed" in result["steps"][0]


# ================================================================
# 6. 生成节点测试
# ================================================================

class TestGenerate:

    @patch("app.rag_graph.get_langfuse_handler", return_value=None)
    @patch("app.rag_graph.get_llm")
    def test_generate_answer(self, mock_get_llm, mock_langfuse):
        docs = _make_docs(2)
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        expected_answer = "RAG是检索增强生成技术，它结合了信息检索和大语言模型。"
        with patch("app.rag_graph.ChatPromptTemplate") as mock_prompt_cls:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = expected_answer
            mock_prompt_cls.from_template.return_value.__or__ = MagicMock(return_value=MagicMock(
                __or__=MagicMock(return_value=mock_chain)
            ))
            state = _make_rag_state(relevant_docs=docs)
            result = generate(state)
            assert result["answer"] == expected_answer
            assert len(result["sources"]) == 2
            assert "generate" in result["steps"][0]

    @patch("app.rag_graph.get_langfuse_handler")
    @patch("app.rag_graph.get_llm")
    def test_generate_with_langfuse(self, mock_get_llm, mock_langfuse):
        """生成时带 langfuse handler"""
        mock_handler = MagicMock()
        mock_langfuse.return_value = mock_handler
        docs = _make_docs(1)
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        with patch("app.rag_graph.ChatPromptTemplate") as mock_prompt_cls:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "回答"
            mock_prompt_cls.from_template.return_value.__or__ = MagicMock(return_value=MagicMock(
                __or__=MagicMock(return_value=mock_chain)
            ))
            state = _make_rag_state(relevant_docs=docs)
            result = generate(state)
            assert result["answer"] == "回答"
            # 验证 invoke 被调用时传入了 callbacks
            mock_chain.invoke.assert_called_once()
            call_kwargs = mock_chain.invoke.call_args
            assert "config" in call_kwargs.kwargs or len(call_kwargs.args) > 1

    @patch("app.rag_graph.get_langfuse_handler", return_value=None)
    @patch("app.rag_graph.get_llm")
    def test_generate_sources_metadata(self, mock_get_llm, mock_langfuse):
        """生成结果包含正确的来源元数据"""
        docs = [
            Document(page_content="A" * 300, metadata={"source": "file_a.md"}),
            Document(page_content="B" * 100, metadata={}),
        ]
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        with patch("app.rag_graph.ChatPromptTemplate") as mock_prompt_cls:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "answer"
            mock_prompt_cls.from_template.return_value.__or__ = MagicMock(return_value=MagicMock(
                __or__=MagicMock(return_value=mock_chain)
            ))
            state = _make_rag_state(relevant_docs=docs)
            result = generate(state)
            assert result["sources"][0]["source"] == "file_a.md"
            assert result["sources"][1]["source"] == "未知"
            # text 被截断到 200 字符
            assert len(result["sources"][0]["text"]) == 200


# ================================================================
# 7. 幻觉检测节点测试
# ================================================================

class TestCheckHallucination:

    @patch("app.rag_graph.get_llm")
    def test_hallucination_pass(self, mock_get_llm):
        docs = _make_docs(2)
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        with patch("app.rag_graph.ChatPromptTemplate") as mock_prompt_cls:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "yes"
            mock_prompt_cls.from_template.return_value.__or__ = MagicMock(return_value=MagicMock(
                __or__=MagicMock(return_value=mock_chain)
            ))
            state = _make_rag_state(answer="RAG是检索增强生成技术", relevant_docs=docs, retry_count=0)
            result = check_hallucination(state)
            assert result["hallucination_pass"] is True
            assert result["retry_count"] == 0

    @patch("app.rag_graph.get_llm")
    def test_hallucination_fail(self, mock_get_llm):
        docs = _make_docs(2)
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        with patch("app.rag_graph.ChatPromptTemplate") as mock_prompt_cls:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "no"
            mock_prompt_cls.from_template.return_value.__or__ = MagicMock(return_value=MagicMock(
                __or__=MagicMock(return_value=mock_chain)
            ))
            state = _make_rag_state(answer="RAG是量子计算技术", relevant_docs=docs, retry_count=0)
            result = check_hallucination(state)
            assert result["hallucination_pass"] is False
            assert result["retry_count"] == 1

    @patch("app.rag_graph.get_llm")
    def test_hallucination_retry_count_increments(self, mock_get_llm):
        """幻觉检测失败时 retry_count 正确递增"""
        docs = _make_docs(1)
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        with patch("app.rag_graph.ChatPromptTemplate") as mock_prompt_cls:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "no"
            mock_prompt_cls.from_template.return_value.__or__ = MagicMock(return_value=MagicMock(
                __or__=MagicMock(return_value=mock_chain)
            ))
            state = _make_rag_state(answer="编造", relevant_docs=docs, retry_count=1)
            result = check_hallucination(state)
            assert result["retry_count"] == 2


# ================================================================
# 8. Fallback 节点测试
# ================================================================

class TestFallback:

    def test_fallback_returns_default_answer(self):
        state = _make_rag_state()
        result = fallback(state)
        assert "抱歉" in result["answer"]
        assert result["hallucination_pass"] is True
        assert "fallback" in result["steps"][0]


# ================================================================
# 9. 条件路由测试
# ================================================================

class TestRouting:

    def test_route_after_grading_has_docs(self):
        state = _make_rag_state(relevant_docs=_make_docs(2))
        assert route_after_grading(state) == "rerank"

    def test_route_after_grading_no_docs(self):
        state = _make_rag_state(relevant_docs=[])
        assert route_after_grading(state) == "fallback"

    def test_route_hallucination_pass(self):
        state = _make_rag_state(hallucination_pass=True)
        assert route_after_hallucination(state) == "finish"

    def test_route_hallucination_fail_retry(self):
        state = _make_rag_state(hallucination_pass=False, retry_count=1)
        assert route_after_hallucination(state) == "regenerate"

    def test_route_hallucination_fail_max_retry(self):
        state = _make_rag_state(hallucination_pass=False, retry_count=2)
        assert route_after_hallucination(state) == "fallback"

    def test_route_hallucination_fail_zero_retry(self):
        state = _make_rag_state(hallucination_pass=False, retry_count=0)
        assert route_after_hallucination(state) == "regenerate"

    def test_route_hallucination_fail_over_max_retry(self):
        """retry_count 超过 2 也走 fallback"""
        state = _make_rag_state(hallucination_pass=False, retry_count=5)
        assert route_after_hallucination(state) == "fallback"


# ================================================================
# 10. 文档特征分析规则引擎测试
# ================================================================

class TestAnalyzeDocFeatures:

    def test_empty_document(self):
        result = _analyze_doc_features("")
        assert result["strategy"] == "recursive"

    def test_highly_structured_markdown(self):
        content = "\n".join([
            "# 标题1", "## 子标题1", "- 列表项1", "- 列表项2",
            "## 子标题2", "```python", "print('hello')", "```",
            "### 子标题3", "- 列表项3", "| 列1 | 列2 |", "| --- | --- |",
            "| 值1 | 值2 |",
        ] * 5)
        result = _analyze_doc_features(content)
        assert result["strategy"] == "recursive"

    def test_long_paragraphs_low_structure(self):
        long_para = "这是一段很长的纯文本内容，没有任何标题或列表标记。" * 50
        content = long_para + "\n\n" + long_para
        result = _analyze_doc_features(content)
        assert result["strategy"] == "semantic"

    def test_plain_text_stream(self):
        short_paras = ["短段落内容。"] * 20
        content = "\n\n".join(short_paras)
        result = _analyze_doc_features(content)
        assert result["strategy"] in ("recursive", "fixed")

    def test_code_heavy_document(self):
        content = "\n".join([
            "# API 文档",
            "## 函数1",
            "```python\ndef func1():\n    pass\n```",
            "## 函数2",
            "```python\ndef func2():\n    pass\n```",
            "## 函数3",
            "```python\ndef func3():\n    pass\n```",
            "## 函数4",
            "```python\ndef func4():\n    pass\n```",
        ] * 3)
        result = _analyze_doc_features(content)
        assert result["strategy"] == "recursive"
        assert result["chunk_size"] >= 512

    def test_returns_reason(self):
        result = _analyze_doc_features("# 标题\n内容")
        assert "reason" in result
        assert len(result["reason"]) > 0

    def test_pure_text_stream_long(self):
        """纯文本流（长文本，无标记，段落长）走 semantic 策略"""
        content = "a" * 15000
        result = _analyze_doc_features(content)
        assert result["strategy"] == "semantic"

    def test_fixed_strategy_low_density(self):
        """低标记密度、低换行密度、段落均长适中的文本走 fixed 策略"""
        content = "\n\n".join(["abcde " * 80] * 5)
        result = _analyze_doc_features(content)
        assert result["strategy"] == "fixed"

    def test_short_structured_doc(self):
        """短结构化文档用小块"""
        content = "# 标题\n## 子标题\n- 列表\n- 列表\n```code\nprint(1)\n```"
        result = _analyze_doc_features(content)
        assert result["strategy"] == "recursive"
        assert result["chunk_size"] == 256  # 短文档


# ================================================================
# 11. 切分质量验证测试
# ================================================================

class TestValidateChunks:

    def test_valid_chunks_pass(self):
        chunks = [Document(page_content="A" * 200) for _ in range(5)]
        state = _make_ingest_state(chunks=chunks, chosen_strategy="recursive")
        result = validate_chunks_node(state)
        assert result["quality_pass"] is True
        assert result["quality_issues"] == ""

    def test_empty_chunks_fail(self):
        state = _make_ingest_state(chunks=[], chosen_strategy="recursive")
        result = validate_chunks_node(state)
        assert result["quality_pass"] is False
        assert "为空" in result["quality_issues"]

    def test_too_many_empty_chunks_fail(self):
        chunks = [Document(page_content="A" * 200) for _ in range(5)]
        chunks += [Document(page_content="短") for _ in range(10)]
        state = _make_ingest_state(chunks=chunks, chosen_strategy="recursive")
        result = validate_chunks_node(state)
        assert result["quality_pass"] is False
        assert "空块" in result["quality_issues"]

    def test_too_short_avg_fail(self):
        chunks = [Document(page_content="A" * 20) for _ in range(10)]
        state = _make_ingest_state(chunks=chunks, chosen_strategy="recursive")
        result = validate_chunks_node(state)
        assert result["quality_pass"] is False
        assert "过碎" in result["quality_issues"]

    def test_too_long_avg_fail(self):
        chunks = [Document(page_content="A" * 5000) for _ in range(3)]
        state = _make_ingest_state(chunks=chunks, chosen_strategy="recursive")
        result = validate_chunks_node(state)
        assert result["quality_pass"] is False
        assert "不足" in result["quality_issues"]

    def test_multiple_issues(self):
        """同时存在多个质量问题"""
        chunks = [Document(page_content="A" * 5) for _ in range(2)]
        chunks += [Document(page_content="") for _ in range(10)]
        state = _make_ingest_state(chunks=chunks, chosen_strategy="recursive")
        result = validate_chunks_node(state)
        assert result["quality_pass"] is False


# ================================================================
# 12. 降级重试机制测试
# ================================================================

class TestFallbackStrategy:

    def test_fallback_to_next_strategy(self):
        state = _make_ingest_state(
            tried_strategies=["recursive"],
            chosen_strategy="recursive",
            quality_issues="切分过碎",
        )
        result = fallback_strategy_node(state)
        assert result["chosen_strategy"] == "fixed"
        assert "fixed" in result["tried_strategies"]

    def test_fallback_skips_tried(self):
        state = _make_ingest_state(
            tried_strategies=["recursive", "fixed"],
            chosen_strategy="fixed",
            quality_issues="切分不足",
        )
        result = fallback_strategy_node(state)
        assert result["chosen_strategy"] == "semantic"

    def test_fallback_all_tried(self):
        state = _make_ingest_state(
            tried_strategies=["recursive", "fixed", "semantic"],
            chosen_strategy="semantic",
            quality_issues="仍然不合格",
        )
        result = fallback_strategy_node(state)
        assert result["chosen_strategy"] == "recursive"
        assert "所有策略均已尝试" in result["steps"][-1]

    def test_fallback_resets_chunks(self):
        """降级后 chunks 被清空"""
        state = _make_ingest_state(
            tried_strategies=["recursive"],
            chosen_strategy="recursive",
            quality_issues="问题",
            chunks=[Document(page_content="old")],
        )
        result = fallback_strategy_node(state)
        assert result["chunks"] == []
        assert result["quality_pass"] is False


# ================================================================
# 13. 入库图路由测试
# ================================================================

class TestIngestRouting:

    def test_route_quality_pass(self):
        state = _make_ingest_state(quality_pass=True)
        assert route_after_validation(state) == "store"

    def test_route_quality_fail_has_untried(self):
        state = _make_ingest_state(quality_pass=False, tried_strategies=["recursive"])
        assert route_after_validation(state) == "fallback"

    def test_route_quality_fail_all_tried_has_chunks(self):
        state = _make_ingest_state(
            quality_pass=False,
            tried_strategies=["recursive", "fixed", "semantic"],
            chunks=[Document(page_content="content")],
        )
        assert route_after_validation(state) == "store"

    def test_route_quality_fail_all_tried_no_chunks(self):
        state = _make_ingest_state(
            quality_pass=False,
            tried_strategies=["recursive", "fixed", "semantic"],
            chunks=[],
        )
        assert route_after_validation(state) == "fallback"


# ================================================================
# 14. 入库节点单元测试
# ================================================================

class TestIngestNodes:

    def test_load_document_node(self):
        state = _make_ingest_state(file_path="docs/sample.md")
        result = load_document_node(state)
        assert len(result["documents"]) > 0
        assert "load" in result["steps"][0]

    def test_analyze_document_node(self):
        doc = Document(page_content="# 标题\n## 子标题\n- 列表\n内容" * 10)
        state = _make_ingest_state(documents=[doc])
        result = analyze_document_node(state)
        assert result["chosen_strategy"] in ("recursive", "fixed", "semantic")
        assert result["chosen_chunk_size"] > 0
        assert len(result["tried_strategies"]) == 1
        assert "analyze" in result["steps"][0]

    def test_analyze_document_node_empty(self):
        """空文档分析"""
        doc = Document(page_content="")
        state = _make_ingest_state(documents=[doc])
        result = analyze_document_node(state)
        assert result["chosen_strategy"] == "recursive"

    def test_analyze_document_node_no_docs(self):
        """无文档时分析空内容"""
        state = _make_ingest_state(documents=[])
        result = analyze_document_node(state)
        assert result["chosen_strategy"] == "recursive"

    def test_split_document_node_success(self):
        doc = Document(page_content="这是一段测试内容。" * 100)
        state = _make_ingest_state(
            documents=[doc],
            chosen_strategy="recursive",
            chosen_chunk_size=512,
            chosen_chunk_overlap=50,
        )
        result = split_document_node(state)
        assert len(result["chunks"]) > 0
        assert "split" in result["steps"][0]

    def test_split_document_node_failure(self):
        state = _make_ingest_state(
            documents=[],
            chosen_strategy="semantic",
            chosen_chunk_size=512,
            chosen_chunk_overlap=50,
        )
        result = split_document_node(state)
        assert "split" in result["steps"][0]

    @patch("app.ingest.ingest_documents")
    def test_store_document_node(self, mock_ingest):
        mock_ingest.return_value = 5
        chunks = _make_docs(5)
        state = _make_ingest_state(chunks=chunks)
        result = store_document_node(state)
        assert result["ingest_count"] == 5
        assert "store" in result["steps"][0]

    def test_split_document_node_fixed_strategy(self):
        """fixed 策略切分"""
        doc = Document(page_content="这是一段测试内容。\n" * 100)
        state = _make_ingest_state(
            documents=[doc],
            chosen_strategy="fixed",
            chosen_chunk_size=256,
            chosen_chunk_overlap=25,
        )
        result = split_document_node(state)
        assert len(result["chunks"]) > 0


# ================================================================
# 15. 端到端查询图测试（全 mock）
# ================================================================

class TestQueryEndToEnd:

    @patch("app.rag_graph.get_langfuse_handler", return_value=None)
    @patch("app.rag_graph.get_llm")
    @patch("app.rag_graph.hybrid_retrieve")
    def test_full_query_flow(self, mock_hybrid, mock_get_llm, mock_langfuse):
        """完整查询流程：检索 → 评估 → 生成 → 幻觉检测"""
        docs = _make_docs(3)
        mock_hybrid.return_value = docs

        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        with patch("app.rag_graph.ChatPromptTemplate") as mock_prompt_cls:
            call_count = [0]

            def chain_invoke_side_effect(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    return "yes\nyes\nyes"
                elif call_count[0] == 2:
                    return "RAG是检索增强生成技术。"
                else:
                    return "yes"

            mock_chain = MagicMock()
            mock_chain.invoke.side_effect = chain_invoke_side_effect
            mock_prompt_cls.from_template.return_value.__or__ = MagicMock(return_value=MagicMock(
                __or__=MagicMock(return_value=mock_chain)
            ))

            result = query_rag("什么是RAG？")
            assert "answer" in result
            assert "sources" in result
            assert "graph_steps" in result
            assert len(result["graph_steps"]) > 0

    @patch("app.rag_graph.get_langfuse_handler", return_value=None)
    @patch("app.rag_graph.get_llm")
    @patch("app.rag_graph.hybrid_retrieve")
    def test_query_no_relevant_docs_fallback(self, mock_hybrid, mock_get_llm, mock_langfuse):
        """无相关文档时走 fallback"""
        docs = _make_docs(2)
        mock_hybrid.return_value = docs

        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        with patch("app.rag_graph.ChatPromptTemplate") as mock_prompt_cls:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "no\nno"
            mock_prompt_cls.from_template.return_value.__or__ = MagicMock(return_value=MagicMock(
                __or__=MagicMock(return_value=mock_chain)
            ))

            result = query_rag("完全无关的问题xyz")
            assert "抱歉" in result["answer"]

    def test_query_returns_config(self):
        with patch("app.rag_graph.rag_graph") as mock_graph:
            mock_graph.invoke.return_value = {
                "answer": "test",
                "sources": [],
                "steps": ["step1"],
            }
            result = query_rag("test", retrieval_mode="vector", query_transform="rewrite")
            assert result["config"]["retrieval_mode"] == "vector"
            assert result["config"]["query_transform"] == "rewrite"

    @patch("app.rag_graph.get_langfuse_handler", return_value=None)
    @patch("app.rag_graph.get_llm")
    @patch("app.rag_graph.hybrid_retrieve")
    def test_query_with_reranker(self, mock_hybrid, mock_get_llm, mock_langfuse):
        """带 reranker 的查询流程"""
        docs = _make_docs(3)
        mock_hybrid.return_value = docs

        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        with patch("app.rag_graph.ChatPromptTemplate") as mock_prompt_cls:
            call_count = [0]

            def chain_invoke_side_effect(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    return "yes\nyes\nyes"
                elif call_count[0] == 2:
                    return "RAG回答"
                else:
                    return "yes"

            mock_chain = MagicMock()
            mock_chain.invoke.side_effect = chain_invoke_side_effect
            mock_prompt_cls.from_template.return_value.__or__ = MagicMock(return_value=MagicMock(
                __or__=MagicMock(return_value=mock_chain)
            ))

            with patch("app.rag_graph.rerank_documents", return_value=docs):
                result = query_rag("什么是RAG？", use_reranker=True)
                assert "answer" in result
                assert result["config"]["use_reranker"] is True


# ================================================================
# 16. 端到端入库图测试
# ================================================================

class TestIngestEndToEnd:

    @patch("app.ingest.ingest_documents")
    @patch("app.retriever.get_embeddings")
    def test_smart_ingest_sample_file(self, mock_embeddings, mock_ingest):
        mock_ingest.return_value = 5
        mock_emb = MagicMock()
        mock_embeddings.return_value = mock_emb

        result = smart_ingest_file("docs/sample.md")
        assert "filename" in result
        assert result["filename"] == "sample.md"
        assert "chunks" in result
        assert "strategy" in result
        assert "graph_steps" in result
        assert len(result["graph_steps"]) > 0
        assert result["chunks"] > 0

    @patch("app.ingest.ingest_documents")
    def test_smart_ingest_returns_analysis(self, mock_ingest):
        mock_ingest.return_value = 3
        result = smart_ingest_file("docs/sample.md")
        assert "analysis" in result
        assert len(result["analysis"]) > 0
        assert "tried_strategies" in result


# ================================================================
# 17. 边界情况测试
# ================================================================

class TestEdgeCases:

    def test_rag_state_type_completeness(self):
        state = _make_rag_state()
        required_keys = {
            "question", "search_query", "retrieval_mode", "query_transform",
            "use_reranker", "top_k", "retrieved_docs", "relevant_docs",
            "answer", "hallucination_pass", "retry_count", "sources", "steps",
        }
        assert required_keys == set(state.keys())

    def test_ingest_state_type_completeness(self):
        state = _make_ingest_state()
        required_keys = {
            "file_path", "documents", "doc_analysis", "chosen_strategy",
            "chosen_chunk_size", "chosen_chunk_overlap", "chunks",
            "quality_pass", "quality_issues", "tried_strategies",
            "ingest_count", "steps",
        }
        assert required_keys == set(state.keys())

    def test_steps_accumulate(self):
        state = _make_rag_state(steps=["step1", "step2"])
        result = fallback(state)
        assert len(result["steps"]) == 3
        assert result["steps"][:2] == ["step1", "step2"]

    def test_fallback_terminates_loop(self):
        state = _make_rag_state()
        result = fallback(state)
        assert result["hallucination_pass"] is True

    def test_analyze_features_short_doc(self):
        result = _analyze_doc_features("# 标题\n简短内容")
        assert result["strategy"] in ("recursive", "fixed", "semantic")
        assert result["chunk_size"] > 0

    def test_validate_single_good_chunk(self):
        chunks = [Document(page_content="A" * 100)]
        state = _make_ingest_state(chunks=chunks, chosen_strategy="recursive")
        result = validate_chunks_node(state)
        assert result["quality_pass"] is True


# ================================================================
# 18. Mermaid 图导出测试
# ================================================================

class TestGraphVisualization:

    def test_rag_graph_mermaid_export(self):
        graph = build_rag_graph()
        mermaid = graph.get_graph().draw_mermaid()
        assert "transform_query" in mermaid
        assert "retrieve" in mermaid
        assert "grade_documents" in mermaid

    def test_ingest_graph_mermaid_export(self):
        graph = build_ingest_graph()
        mermaid = graph.get_graph().draw_mermaid()
        assert "load_document" in mermaid
        assert "analyze_document" in mermaid
        assert "store_document" in mermaid


# ================================================================
# 19. Retriever 模块测试
# ================================================================

class TestRetrieverModule:

    def test_load_file_md(self):
        """加载 markdown 文件"""
        from app.retriever import load_file
        docs = load_file("docs/sample.md")
        assert len(docs) > 0
        assert len(docs[0].page_content) > 0

    def test_load_file_unsupported_format(self):
        """不支持的文件格式抛出异常"""
        from app.retriever import load_file
        with pytest.raises(ValueError, match="不支持的文件格式"):
            load_file("test.pdf")

    def test_split_documents_recursive(self):
        """recursive 策略切分"""
        from app.retriever import split_documents
        docs = [Document(page_content="这是一段测试内容。\n\n" * 100)]
        chunks = split_documents(docs, strategy="recursive", chunk_size=256, chunk_overlap=25)
        assert len(chunks) > 1

    def test_split_documents_fixed(self):
        """fixed 策略切分"""
        from app.retriever import split_documents
        docs = [Document(page_content="这是一段测试内容。\n" * 100)]
        chunks = split_documents(docs, strategy="fixed", chunk_size=256, chunk_overlap=25)
        assert len(chunks) > 1

    def test_split_documents_unknown_strategy(self):
        """未知策略抛出异常"""
        from app.retriever import split_documents
        docs = [Document(page_content="test")]
        with pytest.raises(ValueError, match="未知切分策略"):
            split_documents(docs, strategy="unknown")

    @patch("app.retriever.get_embeddings")
    def test_get_vectorstore(self, mock_embeddings):
        """获取向量存储实例"""
        import app.retriever as rt
        mock_emb = MagicMock()
        mock_embeddings.return_value = mock_emb

        # 重置单例
        original_vs = rt._vectorstore
        rt._vectorstore = None
        try:
            vs = rt.get_vectorstore()
            assert vs is not None
            # 第二次调用返回同一实例
            vs2 = rt.get_vectorstore()
            assert vs is vs2
        finally:
            rt._vectorstore = original_vs

    @patch("app.retriever.get_vectorstore")
    def test_get_vector_retriever(self, mock_vs_fn):
        """获取向量检索器"""
        from app.retriever import get_vector_retriever
        mock_vs = MagicMock()
        mock_retriever = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever
        mock_vs_fn.return_value = mock_vs

        retriever = get_vector_retriever(top_k=3)
        mock_vs.as_retriever.assert_called_once_with(search_kwargs={"k": 3})
        assert retriever == mock_retriever

    @patch("app.retriever.get_vectorstore")
    def test_ingest_documents(self, mock_vs_fn):
        """文档入库"""
        from app.retriever import ingest_documents
        import app.retriever as rt

        mock_vs = MagicMock()
        mock_vs_fn.return_value = mock_vs

        original_bm25 = rt._all_docs_for_bm25
        rt._all_docs_for_bm25 = []
        try:
            docs = _make_docs(3)
            count = ingest_documents(docs)
            assert count == 3
            mock_vs.add_documents.assert_called_once_with(docs)
            assert len(rt._all_docs_for_bm25) == 3
        finally:
            rt._all_docs_for_bm25 = original_bm25

    @patch("app.retriever.get_vectorstore")
    def test_get_bm25_retriever_from_vectorstore(self, mock_vs_fn):
        """BM25 检索器从向量库加载文档"""
        import app.retriever as rt

        original_bm25 = rt._all_docs_for_bm25
        rt._all_docs_for_bm25 = []

        mock_vs = MagicMock()
        mock_vs.get.return_value = {
            "documents": ["doc1 content", "doc2 content"],
            "metadatas": [{"source": "a.md"}, {"source": "b.md"}],
        }
        mock_vs_fn.return_value = mock_vs

        try:
            retriever = rt.get_bm25_retriever(top_k=2)
            assert retriever is not None
        finally:
            rt._all_docs_for_bm25 = original_bm25

    @patch("app.retriever.get_vectorstore")
    def test_get_bm25_retriever_empty_store(self, mock_vs_fn):
        """向量库为空时 BM25 返回 None"""
        import app.retriever as rt

        original_bm25 = rt._all_docs_for_bm25
        rt._all_docs_for_bm25 = []

        mock_vs = MagicMock()
        mock_vs.get.return_value = {"documents": [], "metadatas": []}
        mock_vs_fn.return_value = mock_vs

        try:
            retriever = rt.get_bm25_retriever(top_k=2)
            assert retriever is None
        finally:
            rt._all_docs_for_bm25 = original_bm25

    @patch("app.retriever.get_bm25_retriever")
    @patch("app.retriever.get_vector_retriever")
    def test_hybrid_retrieve(self, mock_vec_fn, mock_bm25_fn):
        """混合检索 RRF 融合"""
        from app.retriever import hybrid_retrieve

        vec_docs = [
            Document(page_content="向量文档1", metadata={"source": "a.md"}),
            Document(page_content="向量文档2", metadata={"source": "b.md"}),
        ]
        bm25_docs = [
            Document(page_content="BM25文档1", metadata={"source": "c.md"}),
            Document(page_content="向量文档1", metadata={"source": "a.md"}),  # 重叠
        ]

        mock_vec_retriever = MagicMock()
        mock_vec_retriever.invoke.return_value = vec_docs
        mock_vec_fn.return_value = mock_vec_retriever

        mock_bm25_retriever = MagicMock()
        mock_bm25_retriever.invoke.return_value = bm25_docs
        mock_bm25_fn.return_value = mock_bm25_retriever

        results = hybrid_retrieve("test query", top_k=3)
        assert len(results) > 0
        assert len(results) <= 3

    @patch("app.retriever.get_bm25_retriever")
    @patch("app.retriever.get_vector_retriever")
    def test_hybrid_retrieve_no_bm25(self, mock_vec_fn, mock_bm25_fn):
        """BM25 不可用时混合检索只用向量"""
        from app.retriever import hybrid_retrieve

        vec_docs = _make_docs(2)
        mock_vec_retriever = MagicMock()
        mock_vec_retriever.invoke.return_value = vec_docs
        mock_vec_fn.return_value = mock_vec_retriever
        mock_bm25_fn.return_value = None

        results = hybrid_retrieve("test", top_k=2)
        assert len(results) == 2

    @patch("app.retriever._reranker_session")
    def test_rerank_documents_success(self, mock_session):
        """Reranker 成功"""
        from app.retriever import rerank_documents

        docs = _make_docs(2)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.5},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_session.post.return_value = mock_resp

        result = rerank_documents("query", docs, top_k=2)
        assert len(result) == 2
        assert result[0].metadata["rerank_score"] == 0.9

    @patch("app.retriever._reranker_session")
    def test_rerank_documents_api_failure(self, mock_session):
        """Reranker API 失败降级"""
        from app.retriever import rerank_documents
        import requests

        docs = _make_docs(2)
        mock_session.post.side_effect = requests.RequestException("timeout")

        result = rerank_documents("query", docs, top_k=2)
        assert result == docs  # 降级返回原始顺序

    def test_rerank_documents_empty(self):
        """空文档列表直接返回"""
        from app.retriever import rerank_documents
        result = rerank_documents("query", [], top_k=2)
        assert result == []

    @patch("app.retriever.get_vectorstore")
    def test_ingest_file(self, mock_vs_fn):
        """ingest_file 完整流程"""
        from app.retriever import ingest_file
        import app.retriever as rt

        mock_vs = MagicMock()
        mock_vs_fn.return_value = mock_vs

        original_bm25 = rt._all_docs_for_bm25
        rt._all_docs_for_bm25 = []
        try:
            result = ingest_file("docs/sample.md", strategy="recursive", chunk_size=256, chunk_overlap=25)
            assert result["filename"] == "sample.md"
            assert result["chunks"] > 0
            assert result["strategy"] == "recursive"
            assert result["chunk_size"] == 256
        finally:
            rt._all_docs_for_bm25 = original_bm25

    def test_compare_chunk_strategies(self):
        """切分策略对比"""
        from app.retriever import compare_chunk_strategies

        with patch("app.retriever.split_documents") as mock_split:
            mock_split.return_value = [Document(page_content="A" * 200)]

            with patch("app.retriever.load_file") as mock_load:
                mock_load.return_value = [Document(page_content="test content" * 100)]
                result = compare_chunk_strategies("docs/sample.md")
                assert "fixed_256" in result
                assert "recursive_512" in result


# ================================================================
# 20. LLM 模块测试
# ================================================================

class TestLLMModule:

    def test_get_llm_returns_instance(self):
        """get_llm 返回 ChatOpenAI 实例"""
        from app.llm import get_llm
        llm = get_llm()
        assert llm is not None
        assert llm.temperature == 0.3
        assert llm.max_tokens == 1024

    def test_get_embeddings_returns_instance(self):
        """get_embeddings 返回 OpenAIEmbeddings 实例"""
        from app.llm import get_embeddings
        emb = get_embeddings()
        assert emb is not None

    def test_get_langfuse_handler_no_keys(self):
        """无 langfuse 配置时返回 None"""
        from app.llm import get_langfuse_handler
        with patch("app.llm.LANGFUSE_SECRET_KEY", ""), \
             patch("app.llm.LANGFUSE_PUBLIC_KEY", ""):
            result = get_langfuse_handler()
            assert result is None

    def test_get_langfuse_handler_no_module(self):
        """langfuse 模块不可用时返回 None"""
        from app.llm import get_langfuse_handler
        with patch("app.llm.LangfuseCallbackHandler", None):
            result = get_langfuse_handler()
            assert result is None

    @patch("app.llm.LangfuseCallbackHandler")
    def test_get_langfuse_handler_with_keys(self, mock_handler_cls):
        """有 langfuse 配置时返回 handler"""
        from app.llm import get_langfuse_handler
        mock_handler = MagicMock()
        mock_handler_cls.return_value = mock_handler

        with patch("app.llm.LANGFUSE_SECRET_KEY", "sk-test"), \
             patch("app.llm.LANGFUSE_PUBLIC_KEY", "pk-test"), \
             patch("app.llm.LANGFUSE_HOST", "http://localhost:3000"):
            result = get_langfuse_handler()
            assert result == mock_handler
            mock_handler_cls.assert_called_once_with(
                secret_key="sk-test",
                public_key="pk-test",
                host="http://localhost:3000",
                trace_name="rag-query",
                session_id=None,
                user_id=None,
                metadata={},
            )


# ================================================================
# 21. FastAPI API 层测试
# ================================================================

class TestFastAPIEndpoints:

    @pytest.fixture(autouse=True)
    def setup_client(self):
        """创建测试客户端"""
        from fastapi.testclient import TestClient
        from app.main import app
        self.client = TestClient(app)

    def test_health_all_ok(self):
        """健康检查 — 所有服务正常"""
        with patch("app.main.get_llm") as mock_llm_fn, \
             patch("app.main.get_embeddings") as mock_emb_fn, \
             patch("app.main.get_vectorstore") as mock_vs_fn, \
             patch("app.main._is_langfuse_enabled", return_value=True):

            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(content="ok")
            mock_llm_fn.return_value = mock_llm

            mock_emb = MagicMock()
            mock_emb.embed_query.return_value = [0.1] * 10
            mock_emb_fn.return_value = mock_emb

            mock_vs = MagicMock()
            mock_vs._collection.count.return_value = 42
            mock_vs_fn.return_value = mock_vs

            resp = self.client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert "42" in data["services"]["chroma"]
            assert "langfuse" in data["services"]

    def test_health_degraded(self):
        """健康检查 — 部分服务异常"""
        with patch("app.main.get_llm") as mock_llm_fn, \
             patch("app.main.get_embeddings") as mock_emb_fn, \
             patch("app.main.get_vectorstore") as mock_vs_fn:

            mock_llm_fn.side_effect = Exception("LLM down")

            mock_emb = MagicMock()
            mock_emb.embed_query.return_value = [0.1]
            mock_emb_fn.return_value = mock_emb

            mock_vs = MagicMock()
            mock_vs._collection.count.return_value = 0
            mock_vs_fn.return_value = mock_vs

            resp = self.client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "degraded"
            assert "error" in data["services"]["llm"]

    @patch("app.main.query_rag")
    def test_query_endpoint(self, mock_query_rag):
        """POST /query 正常查询"""
        mock_query_rag.return_value = {
            "answer": "RAG是检索增强生成技术",
            "sources": [],
            "config": {"retrieval_mode": "hybrid", "query_transform": "none", "use_reranker": False, "top_k": 5},
            "graph_steps": ["step1"],
        }
        resp = self.client.post("/query", json={"question": "什么是RAG？"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "RAG是检索增强生成技术"
        mock_query_rag.assert_called_once()

    @patch("app.main.query_rag")
    def test_query_endpoint_with_params(self, mock_query_rag):
        """POST /query 带完整参数"""
        mock_query_rag.return_value = {"answer": "ok", "sources": [], "config": {}, "graph_steps": []}
        resp = self.client.post("/query", json={
            "question": "test",
            "retrieval_mode": "vector",
            "query_transform": "rewrite",
            "use_reranker": True,
            "top_k": 10,
        })
        assert resp.status_code == 200
        mock_query_rag.assert_called_once_with(
            question="test",
            retrieval_mode="vector",
            query_transform="rewrite",
            use_reranker=True,
            top_k=10,
            session_id=None,
            user_id=None,
        )

    @patch("app.main.ingest_file")
    def test_upload_document(self, mock_ingest):
        """POST /upload 上传文档"""
        mock_ingest.return_value = {"chunks": 5, "filename": "test.md", "strategy": "recursive", "chunk_size": 512}
        file_content = b"# Test\nThis is test content."
        resp = self.client.post(
            "/upload",
            files={"file": ("test.md", BytesIO(file_content), "text/markdown")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "成功入库" in data["message"]
        assert data["chunks"] == 5

    def test_upload_unsupported_format(self):
        """POST /upload 不支持的文件格式"""
        resp = self.client.post(
            "/upload",
            files={"file": ("test.pdf", BytesIO(b"pdf content"), "application/pdf")},
        )
        assert resp.status_code == 400

    @patch("app.main.smart_ingest_file")
    def test_smart_upload(self, mock_smart_ingest):
        """POST /smart-upload 智能上传"""
        mock_smart_ingest.return_value = {
            "strategy": "recursive", "chunks": 8, "filename": "test.md",
            "analysis": "结构化文档", "tried_strategies": ["recursive"],
            "quality_issues": "", "graph_steps": ["step1"],
        }
        file_content = b"# Test\nContent here."
        resp = self.client.post(
            "/smart-upload",
            files={"file": ("test.md", BytesIO(file_content), "text/markdown")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "智能入库完成" in data["message"]

    def test_smart_upload_unsupported_format(self):
        """POST /smart-upload 不支持的文件格式"""
        resp = self.client.post(
            "/smart-upload",
            files={"file": ("test.csv", BytesIO(b"a,b,c"), "text/csv")},
        )
        assert resp.status_code == 400

    def test_graph_visualization(self):
        """GET /graph 返回 HTML"""
        resp = self.client.get("/graph")
        assert resp.status_code == 200
        assert "mermaid" in resp.text.lower()
        assert "transform_query" in resp.text

    def test_ingest_graph_visualization(self):
        """GET /ingest-graph 返回 HTML"""
        resp = self.client.get("/ingest-graph")
        assert resp.status_code == 200
        assert "mermaid" in resp.text.lower()
        assert "load_document" in resp.text

    @patch("app.main.compare_chunk_strategies")
    def test_compare_chunks(self, mock_compare):
        """POST /compare-chunks"""
        mock_compare.return_value = {"fixed_256": {"num_chunks": 10}}
        file_content = b"# Test\nContent."
        resp = self.client.post(
            "/compare-chunks",
            files={"file": ("test.md", BytesIO(file_content), "text/markdown")},
        )
        assert resp.status_code == 200

    def test_compare_chunks_unsupported_format(self):
        """POST /compare-chunks 不支持的格式"""
        resp = self.client.post(
            "/compare-chunks",
            files={"file": ("test.pdf", BytesIO(b"pdf"), "application/pdf")},
        )
        assert resp.status_code == 400

    @patch("app.main.load_file")
    @patch("app.main.split_documents")
    def test_compare_chunks_detail(self, mock_split, mock_load):
        """POST /compare-chunks-detail"""
        mock_load.return_value = [Document(page_content="test content" * 50)]
        mock_split.return_value = [Document(page_content="chunk1"), Document(page_content="chunk2")]

        file_content = b"# Test\nContent."
        resp = self.client.post(
            "/compare-chunks-detail",
            files={"file": ("test.md", BytesIO(file_content), "text/markdown")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "fixed_256" in data
        assert "recursive_512" in data

    def test_compare_chunks_detail_unsupported_format(self):
        """POST /compare-chunks-detail 不支持的格式"""
        resp = self.client.post(
            "/compare-chunks-detail",
            files={"file": ("test.jpg", BytesIO(b"img"), "image/jpeg")},
        )
        assert resp.status_code == 400

    @patch("app.main.query_rag_stream")
    def test_query_stream_endpoint(self, mock_stream):
        """POST /query/stream 流式查询"""
        async def fake_stream(*args, **kwargs):
            yield 'data: {"type": "token", "content": "hello"}\n\n'
            yield 'data: {"type": "done"}\n\n'

        mock_stream.return_value = fake_stream()
        resp = self.client.post("/query/stream", json={"question": "什么是RAG？"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")


# ================================================================
# 22. 流式查询图测试
# ================================================================

class TestStreamQuery:

    def test_query_rag_stream(self):
        """流式查询通过 FastAPI TestClient 验证"""
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)

        async def fake_stream(*args, **kwargs):
            yield 'data: {"type": "token", "content": "hello"}\n\n'
            yield 'data: {"type": "step", "node": "generate", "step": "generate"}\n\n'
            yield 'data: {"type": "done"}\n\n'

        with patch("app.main.query_rag_stream", return_value=fake_stream()):
            resp = client.post("/query/stream", json={"question": "什么是RAG？"})
            assert resp.status_code == 200
            body = resp.text
            assert "token" in body
            assert "done" in body


# ================================================================
# 23. Config 模块测试
# ================================================================

class TestConfig:

    def test_config_defaults(self):
        """配置模块有合理的默认值"""
        from app.config import (
            CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, DOCS_DIR,
            CHROMA_PERSIST_DIR, CHROMA_COLLECTION,
        )
        assert CHUNK_SIZE > 0
        assert CHUNK_OVERLAP >= 0
        assert TOP_K > 0
        assert len(DOCS_DIR) > 0
        assert len(CHROMA_PERSIST_DIR) > 0
        assert len(CHROMA_COLLECTION) > 0

    def test_config_types(self):
        """配置值类型正确"""
        from app.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
        assert isinstance(CHUNK_SIZE, int)
        assert isinstance(CHUNK_OVERLAP, int)
        assert isinstance(TOP_K, int)
