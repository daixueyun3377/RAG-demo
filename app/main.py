# FastAPI 入口 - LangGraph 版（Langfuse 全链路追踪）
import os
import uuid
import shutil
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from app.config import DOCS_DIR
from app.llm import get_llm, get_embeddings, _is_langfuse_enabled
from app.retriever import (
    get_vectorstore, load_file, split_documents, compare_chunk_strategies,
    ingest_file,
)
from app.ingest import smart_ingest_file, ingest_graph
from app.rag_graph import query_rag, query_rag_stream, rag_graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(DOCS_DIR, exist_ok=True)
    yield


app = FastAPI(title="RAG Demo - LangGraph", version="0.7.0-langfuse", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


class QueryRequest(BaseModel):
    question: str
    retrieval_mode: Literal["vector", "bm25", "hybrid"] = "hybrid"
    query_transform: Literal["none", "rewrite", "hyde"] = "none"
    use_reranker: bool = False
    top_k: int = 5
    session_id: str | None = None
    user_id: str | None = None


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    strategy: str = Query("recursive", description="切分策略: fixed / recursive / semantic"),
    chunk_size: int = Query(512, description="切分大小"),
    chunk_overlap: int = Query(50, description="重叠大小"),
):
    """上传文档并入库"""
    if not file.filename.endswith((".txt", ".md")):
        raise HTTPException(400, "目前只支持 .txt 和 .md 文件")

    file_path = os.path.join(DOCS_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = ingest_file(file_path, strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return {"message": f"成功入库 {result['chunks']} 个文本块", **result}


@app.post("/smart-upload")
async def smart_upload_document(file: UploadFile = File(...)):
    """智能上传 — 规则分析文档特征，动态选择最佳切分策略，质量验证+降级重试（LangGraph 版）"""
    if not file.filename.endswith((".txt", ".md")):
        raise HTTPException(400, "目前只支持 .txt 和 .md 文件")

    file_path = os.path.join(DOCS_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = smart_ingest_file(file_path)
    return {"message": f"智能入库完成：{result['strategy']} 策略，{result['chunks']} 个文本块", **result}


@app.post("/query")
async def query(req: QueryRequest):
    """LangGraph RAG 查询（返回结果含 graph_steps 执行轨迹，Langfuse 全链路追踪）"""
    result = query_rag(
        question=req.question,
        retrieval_mode=req.retrieval_mode,
        query_transform=req.query_transform,
        use_reranker=req.use_reranker,
        top_k=req.top_k,
        session_id=req.session_id,
        user_id=req.user_id,
    )
    return result


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """LangGraph RAG 流式查询（SSE），逐 token 输出生成结果，Langfuse 追踪"""
    return StreamingResponse(
        query_rag_stream(
            question=req.question,
            retrieval_mode=req.retrieval_mode,
            query_transform=req.query_transform,
            use_reranker=req.use_reranker,
            top_k=req.top_k,
            session_id=req.session_id,
            user_id=req.user_id,
        ),
        media_type="text/event-stream",
    )


@app.get("/graph", response_class=HTMLResponse)
async def graph_visualization():
    """LangGraph 流程图可视化（Mermaid）"""
    try:
        mermaid_src = rag_graph.get_graph().draw_mermaid()
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>RAG Graph</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
</head><body>
<h2>RAG LangGraph 流程图</h2>
<pre class="mermaid">{mermaid_src}</pre>
<script>mermaid.initialize({{startOnLoad:true}});</script>
</body></html>"""
        return html
    except Exception as e:
        return f"<pre>Graph export error: {e}\n\nMermaid source:\n{rag_graph.get_graph().draw_mermaid()}</pre>"


@app.get("/ingest-graph", response_class=HTMLResponse)
async def ingest_graph_visualization():
    """智能入库 LangGraph 流程图可视化（Mermaid）"""
    try:
        mermaid_src = ingest_graph.get_graph().draw_mermaid()
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Ingest Graph</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
</head><body>
<h2>智能入库 LangGraph 流程图</h2>
<pre class="mermaid">{mermaid_src}</pre>
<script>mermaid.initialize({{startOnLoad:true}});</script>
</body></html>"""
        return html
    except Exception as e:
        return f"<pre>Graph export error: {e}</pre>"


@app.post("/compare-chunks")
async def compare_chunks(file: UploadFile = File(...)):
    """对比不同切分策略的效果"""
    if not file.filename.endswith((".txt", ".md")):
        raise HTTPException(400, "目前只支持 .txt 和 .md 文件")

    file_path = os.path.join(DOCS_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return compare_chunk_strategies(file_path)


@app.post("/compare-chunks-detail")
async def compare_chunks_detail(file: UploadFile = File(...)):
    """对比不同切分策略 — 返回每个 chunk 的完整内容"""
    if not file.filename.endswith((".txt", ".md")):
        raise HTTPException(400, "目前只支持 .txt 和 .md 文件")

    file_path = os.path.join(DOCS_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

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
                "chunks": [c.page_content for c in chunks],
            }

    try:
        semantic_chunks = split_documents(docs, strategy="semantic")
        results["semantic"] = {
            "strategy": "semantic",
            "chunk_size": "auto (embedding-based)",
            "num_chunks": len(semantic_chunks),
            "avg_length": round(sum(len(c.page_content) for c in semantic_chunks) / max(len(semantic_chunks), 1), 1),
            "chunks": [c.page_content for c in semantic_chunks],
        }
    except Exception as e:
        results["semantic"] = {"strategy": "semantic", "error": str(e)}

    return results


@app.get("/health")
async def health():
    """健康检查 — 检测 LLM、Embedding、Chroma 服务状态"""
    checks = {}

    # LLM
    try:
        llm = get_llm()
        llm.invoke("ping")
        checks["llm"] = "ok"
    except Exception as e:
        checks["llm"] = f"error: {e}"

    # Embedding
    try:
        emb = get_embeddings()
        emb.embed_query("ping")
        checks["embedding"] = "ok"
    except Exception as e:
        checks["embedding"] = f"error: {e}"

    # Chroma
    try:
        vs = get_vectorstore()
        count = vs._collection.count()
        checks["chroma"] = f"ok ({count} docs)"
    except Exception as e:
        checks["chroma"] = f"error: {e}"

    # Langfuse
    checks["langfuse"] = "ok (enabled)" if _is_langfuse_enabled() else "disabled (no keys configured)"

    all_ok = all(v.startswith("ok") for v in checks.values())
    return {
        "status": "ok" if all_ok else "degraded",
        "version": "0.7.0-langfuse",
        "services": checks,
    }
