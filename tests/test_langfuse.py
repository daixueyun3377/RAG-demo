# -*- coding: utf-8 -*-
"""
Langfuse 集成测试
覆盖：
  1. Langfuse 配置检测
  2. Handler 创建（启用/禁用场景）
  3. Handler 参数传递（trace_name, session_id, user_id, metadata）
  4. RAG 节点中 callbacks 传递验证
  5. API 层 session_id / user_id 透传
  6. 健康检查中 Langfuse 状态
  7. 异常容错（Langfuse 不可用时不影响主流程）
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock, call
from langchain_core.documents import Document

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ================================================================
# Fixtures & Helpers
# ================================================================

def _make_rag_state(**overrides):
    """构造默认 RAGState"""
    from app.rag_graph import RAGState
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
        "langfuse_handler": None,
    }
    base.update(overrides)
    return base


def _make_docs(n=3):
    return [
        Document(page_content=f"文档内容 {i}: RAG是检索增强生成技术。", metadata={"source": f"test_{i}.md"})
        for i in range(n)
    ]


# ================================================================
# 1. Langfuse 配置检测测试
# ================================================================

class TestLangfuseConfig:

    @patch("app.llm.LANGFUSE_SECRET_KEY", "sk-lf-test")
    @patch("app.llm.LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    @patch("app.llm.LangfuseCallbackHandler", MagicMock)
    def test_is_enabled_with_keys(self):
        from app.llm import _is_langfuse_enabled
        assert _is_langfuse_enabled() is True

    @patch("app.llm.LANGFUSE_SECRET_KEY", "")
    @patch("app.llm.LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    def test_is_disabled_without_secret_key(self):
        from app.llm import _is_langfuse_enabled
        assert _is_langfuse_enabled() is False

    @patch("app.llm.LANGFUSE_SECRET_KEY", "sk-lf-test")
    @patch("app.llm.LANGFUSE_PUBLIC_KEY", "")
    def test_is_disabled_without_public_key(self):
        from app.llm import _is_langfuse_enabled
        assert _is_langfuse_enabled() is False

    @patch("app.llm.LANGFUSE_SECRET_KEY", "")
    @patch("app.llm.LANGFUSE_PUBLIC_KEY", "")
    def test_is_disabled_without_both_keys(self):
        from app.llm import _is_langfuse_enabled
        assert _is_langfuse_enabled() is False

    @patch("app.llm.LangfuseCallbackHandler", None)
    @patch("app.llm.LANGFUSE_SECRET_KEY", "sk-lf-test")
    @patch("app.llm.LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    def test_is_disabled_without_sdk(self):
        from app.llm import _is_langfuse_enabled
        assert _is_langfuse_enabled() is False


# ================================================================
# 2. Handler 创建测试
# ================================================================

class TestGetLangfuseHandler:

    @patch("app.llm.LANGFUSE_SECRET_KEY", "")
    def test_returns_none_when_disabled(self):
        from app.llm import get_langfuse_handler
        assert get_langfuse_handler() is None

    @patch("app.llm.LANGFUSE_HOST", "http://localhost:3000")
    @patch("app.llm.LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    @patch("app.llm.LANGFUSE_SECRET_KEY", "sk-lf-test")
    @patch("app.llm.LangfuseCallbackHandler")
    def test_creates_handler_with_defaults(self, MockHandler):
        mock_instance = MagicMock()
        MockHandler.return_value = mock_instance
        from app.llm import get_langfuse_handler
        handler = get_langfuse_handler()
        assert handler is mock_instance
        MockHandler.assert_called_once_with(
            secret_key="sk-lf-test",
            public_key="pk-lf-test",
            host="http://localhost:3000",
            trace_name="rag-query",
            session_id=None,
            user_id=None,
            metadata={},
        )

    @patch("app.llm.LANGFUSE_HOST", "http://localhost:3000")
    @patch("app.llm.LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    @patch("app.llm.LANGFUSE_SECRET_KEY", "sk-lf-test")
    @patch("app.llm.LangfuseCallbackHandler")
    def test_creates_handler_with_custom_params(self, MockHandler):
        mock_instance = MagicMock()
        MockHandler.return_value = mock_instance
        from app.llm import get_langfuse_handler
        handler = get_langfuse_handler(
            trace_name="custom-trace",
            session_id="sess-123",
            user_id="user-456",
            metadata={"mode": "hybrid"},
        )
        assert handler is mock_instance
        MockHandler.assert_called_once_with(
            secret_key="sk-lf-test",
            public_key="pk-lf-test",
            host="http://localhost:3000",
            trace_name="custom-trace",
            session_id="sess-123",
            user_id="user-456",
            metadata={"mode": "hybrid"},
        )

    @patch("app.llm.LANGFUSE_HOST", "http://localhost:3000")
    @patch("app.llm.LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    @patch("app.llm.LANGFUSE_SECRET_KEY", "sk-lf-test")
    @patch("app.llm.LangfuseCallbackHandler")
    def test_returns_none_on_exception(self, MockHandler):
        """Handler 创建异常时返回 None，不影响主流程"""
        MockHandler.side_effect = Exception("Connection refused")
        from app.llm import get_langfuse_handler
        handler = get_langfuse_handler()
        assert handler is None


# ================================================================
# 3. RAG 节点 Callbacks 传递测试
# ================================================================

class TestNodeCallbacksPassing:
    """验证每个 LLM 节点都正确传递 langfuse_handler 作为 callbacks"""

    @patch("app.rag_graph.get_llm")
    def test_transform_query_rewrite_passes_callbacks(self, mock_get_llm):
        """query rewrite 节点传递 callbacks"""
        mock_handler = MagicMock()
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        from app.rag_graph import transform_query
        state = _make_rag_state(
            question="什么是RAG？",
            query_transform="rewrite",
            langfuse_handler=mock_handler,
        )
        with patch("app.rag_graph.ChatPromptTemplate") as mock_prompt_cls:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "改写后的查询"
            mock_prompt_cls.from_template.return_value.__or__ = MagicMock(
                return_value=MagicMock(__or__=MagicMock(return_value=mock_chain))
            )
            transform_query(state)
            # 验证 invoke 被调用时传入了 callbacks
            mock_chain.invoke.assert_called_once()
            call_args = mock_chain.invoke.call_args
            config = call_args.kwargs.get("config") or call_args[1].get("config", {})
            assert mock_handler in config.get("callbacks", [])

    @patch("app.rag_graph.get_llm")
    def test_transform_query_hyde_passes_callbacks(self, mock_get_llm):
        """hyde 节点传递 callbacks"""
        mock_handler = MagicMock()
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        from app.rag_graph import transform_query
        state = _make_rag_state(
            question="什么是RAG？",
            query_transform="hyde",
            langfuse_handler=mock_handler,
        )
        with patch("app.rag_graph.ChatPromptTemplate") as mock_prompt_cls:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "HyDE 回答"
            mock_prompt_cls.from_template.return_value.__or__ = MagicMock(
                return_value=MagicMock(__or__=MagicMock(return_value=mock_chain))
            )
            transform_query(state)
            call_args = mock_chain.invoke.call_args
            config = call_args.kwargs.get("config") or call_args[1].get("config", {})
            assert mock_handler in config.get("callbacks", [])

    def test_transform_query_passthrough_no_llm_call(self):
        """passthrough 模式不调用 LLM，无需 callbacks"""
        from app.rag_graph import transform_query
        state = _make_rag_state(question="test", query_transform="none")
        result = transform_query(state)
        assert result["search_query"] == "test"

    @patch("app.rag_graph.get_llm")
    def test_grade_documents_passes_callbacks(self, mock_get_llm):
        """文档评估节点传递 callbacks"""
        mock_handler = MagicMock()
        docs = _make_docs(2)
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        from app.rag_graph import grade_documents
        state = _make_rag_state(retrieved_docs=docs, langfuse_handler=mock_handler)
        with patch("app.rag_graph.ChatPromptTemplate") as mock_prompt_cls:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "yes\nyes"
            mock_prompt_cls.from_template.return_value.__or__ = MagicMock(
                return_value=MagicMock(__or__=MagicMock(return_value=mock_chain))
            )
            grade_documents(state)
            call_args = mock_chain.invoke.call_args
            config = call_args.kwargs.get("config") or call_args[1].get("config", {})
            assert mock_handler in config.get("callbacks", [])

    @patch("app.rag_graph.get_langfuse_handler", return_value=None)
    @patch("app.rag_graph.get_llm")
    def test_generate_passes_callbacks_from_state(self, mock_get_llm, _):
        """generate 节点使用 state 中的 handler 而非重新创建"""
        mock_handler = MagicMock()
        docs = _make_docs(1)
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        from app.rag_graph import generate
        state = _make_rag_state(relevant_docs=docs, langfuse_handler=mock_handler)
        with patch("app.rag_graph.ChatPromptTemplate") as mock_prompt_cls:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "回答"
            mock_prompt_cls.from_template.return_value.__or__ = MagicMock(
                return_value=MagicMock(__or__=MagicMock(return_value=mock_chain))
            )
            generate(state)
            call_args = mock_chain.invoke.call_args
            config = call_args.kwargs.get("config") or call_args[1].get("config", {})
            assert mock_handler in config.get("callbacks", [])

    @patch("app.rag_graph.get_llm")
    def test_check_hallucination_passes_callbacks(self, mock_get_llm):
        """幻觉检测节点传递 callbacks"""
        mock_handler = MagicMock()
        docs = _make_docs(1)
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        from app.rag_graph import check_hallucination
        state = _make_rag_state(
            answer="RAG是检索增强生成",
            relevant_docs=docs,
            langfuse_handler=mock_handler,
        )
        with patch("app.rag_graph.ChatPromptTemplate") as mock_prompt_cls:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "yes"
            mock_prompt_cls.from_template.return_value.__or__ = MagicMock(
                return_value=MagicMock(__or__=MagicMock(return_value=mock_chain))
            )
            check_hallucination(state)
            call_args = mock_chain.invoke.call_args
            config = call_args.kwargs.get("config") or call_args[1].get("config", {})
            assert mock_handler in config.get("callbacks", [])


# ================================================================
# 4. 无 Handler 时节点正常工作（容错）
# ================================================================

class TestNoHandlerFallback:
    """Langfuse 未配置时，所有节点应正常工作，callbacks 为空列表"""

    @patch("app.rag_graph.get_llm")
    def test_generate_works_without_handler(self, mock_get_llm):
        docs = _make_docs(1)
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        from app.rag_graph import generate
        state = _make_rag_state(relevant_docs=docs, langfuse_handler=None)
        with patch("app.rag_graph.ChatPromptTemplate") as mock_prompt_cls:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "回答"
            mock_prompt_cls.from_template.return_value.__or__ = MagicMock(
                return_value=MagicMock(__or__=MagicMock(return_value=mock_chain))
            )
            result = generate(state)
            assert result["answer"] == "回答"
            # callbacks 应为空列表
            call_args = mock_chain.invoke.call_args
            config = call_args.kwargs.get("config") or call_args[1].get("config", {})
            assert config.get("callbacks", []) == []

    @patch("app.rag_graph.get_llm")
    def test_grade_documents_works_without_handler(self, mock_get_llm):
        docs = _make_docs(2)
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        from app.rag_graph import grade_documents
        state = _make_rag_state(retrieved_docs=docs, langfuse_handler=None)
        with patch("app.rag_graph.ChatPromptTemplate") as mock_prompt_cls:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "yes\nno"
            mock_prompt_cls.from_template.return_value.__or__ = MagicMock(
                return_value=MagicMock(__or__=MagicMock(return_value=mock_chain))
            )
            result = grade_documents(state)
            assert len(result["relevant_docs"]) == 1


# ================================================================
# 5. _get_callbacks 辅助函数测试
# ================================================================

class TestGetCallbacks:

    def test_returns_handler_in_list(self):
        from app.rag_graph import _get_callbacks
        mock_handler = MagicMock()
        state = _make_rag_state(langfuse_handler=mock_handler)
        assert _get_callbacks(state) == [mock_handler]

    def test_returns_empty_list_when_none(self):
        from app.rag_graph import _get_callbacks
        state = _make_rag_state(langfuse_handler=None)
        assert _get_callbacks(state) == []


# ================================================================
# 6. query_rag 入口 Langfuse 集成测试
# ================================================================

class TestQueryRagLangfuse:

    @patch("app.rag_graph.rag_graph")
    @patch("app.rag_graph.get_langfuse_handler")
    def test_query_rag_creates_handler_with_metadata(self, mock_get_handler, mock_graph):
        """query_rag 入口创建 handler 并传入正确的 metadata"""
        mock_handler = MagicMock()
        mock_get_handler.return_value = mock_handler
        mock_graph.invoke.return_value = {
            "answer": "test",
            "sources": [],
            "steps": [],
        }

        from app.rag_graph import query_rag
        query_rag(
            question="什么是RAG",
            retrieval_mode="vector",
            query_transform="rewrite",
            use_reranker=True,
            top_k=3,
            session_id="sess-abc",
            user_id="user-xyz",
        )

        mock_get_handler.assert_called_once_with(
            trace_name="rag-query",
            session_id="sess-abc",
            user_id="user-xyz",
            metadata={
                "retrieval_mode": "vector",
                "query_transform": "rewrite",
                "use_reranker": True,
                "top_k": 3,
            },
        )

        # 验证 handler 被注入到 initial_state
        invoke_args = mock_graph.invoke.call_args[0][0]
        assert invoke_args["langfuse_handler"] is mock_handler

    @patch("app.rag_graph.rag_graph")
    @patch("app.rag_graph.get_langfuse_handler", return_value=None)
    def test_query_rag_works_without_langfuse(self, mock_get_handler, mock_graph):
        """Langfuse 未配置时 query_rag 正常工作"""
        mock_graph.invoke.return_value = {
            "answer": "回答",
            "sources": [],
            "steps": [],
        }

        from app.rag_graph import query_rag
        result = query_rag("什么是RAG")
        assert result["answer"] == "回答"

        invoke_args = mock_graph.invoke.call_args[0][0]
        assert invoke_args["langfuse_handler"] is None


# ================================================================
# 7. API 层测试
# ================================================================

class TestAPILangfuse:

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)

    @patch("app.rag_graph.rag_graph")
    @patch("app.rag_graph.get_langfuse_handler")
    def test_query_endpoint_passes_session_and_user(self, mock_get_handler, mock_graph, client):
        """POST /query 传递 session_id 和 user_id"""
        mock_get_handler.return_value = MagicMock()
        mock_graph.invoke.return_value = {
            "answer": "test answer",
            "sources": [],
            "steps": [],
        }

        resp = client.post("/query", json={
            "question": "什么是RAG",
            "session_id": "sess-001",
            "user_id": "user-001",
        })
        assert resp.status_code == 200

        mock_get_handler.assert_called_once()
        call_kwargs = mock_get_handler.call_args
        assert call_kwargs.kwargs.get("session_id") == "sess-001"
        assert call_kwargs.kwargs.get("user_id") == "user-001"

    @patch("app.rag_graph.rag_graph")
    @patch("app.rag_graph.get_langfuse_handler", return_value=None)
    def test_query_endpoint_works_without_session(self, mock_get_handler, mock_graph, client):
        """不传 session_id 也能正常工作"""
        mock_graph.invoke.return_value = {
            "answer": "test",
            "sources": [],
            "steps": [],
        }

        resp = client.post("/query", json={"question": "什么是RAG"})
        assert resp.status_code == 200

    def test_health_shows_langfuse_status(self, client):
        """健康检查包含 Langfuse 状态"""
        with patch("app.main._is_langfuse_enabled", return_value=False):
            resp = client.get("/health")
            # health 可能因为 LLM 不可用而返回 degraded，但结构应正确
            data = resp.json()
            assert "langfuse" in data["services"]

    def test_health_shows_langfuse_enabled(self, client):
        """Langfuse 启用时健康检查显示 enabled"""
        with patch("app.main._is_langfuse_enabled", return_value=True):
            resp = client.get("/health")
            data = resp.json()
            assert "enabled" in data["services"]["langfuse"]


# ================================================================
# 8. RAGState 包含 langfuse_handler 字段
# ================================================================

class TestRAGStateSchema:

    def test_state_has_langfuse_handler_field(self):
        """RAGState 包含 langfuse_handler 字段"""
        from app.rag_graph import RAGState
        state = _make_rag_state()
        assert "langfuse_handler" in state

    def test_state_langfuse_handler_defaults_to_none(self):
        state = _make_rag_state()
        assert state["langfuse_handler"] is None
