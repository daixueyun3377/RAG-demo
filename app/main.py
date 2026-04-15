# FastAPI 入口 - LangGraph 版
import os
import shutil
from typing import Literal
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from app.config import DOCS_DIR
from app.rag_graph import (
    ingest_file, query_rag, compare_chunk_strategies,
    load_file, split_documents, rag_graph,
)

app = FastAPI(title="RAG Demo - LangGraph", version="0.5.0")
app.mount("/static", StaticFiles(directory="static"), name="static")


class QueryRequest(BaseModel):
    question: str
    retrieval_mode: Literal["vector", "bm25", "hybrid"] = "hybrid"
    query_transform: Literal["none", "rewrite", "hyde"] = "none"
    use_reranker: bool = False
    top_k: int = 5


@app.on_event("startup")
def startup():
    os.makedirs(DOCS_DIR, exist_ok=True)


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


@app.post("/query")
async def query(req: QueryRequest):
    """LangGraph RAG 查询（返回结果含 graph_steps 执行轨迹）"""
    result = query_rag(
        question=req.question,
        retrieval_mode=req.retrieval_mode,
        query_transform=req.query_transform,
        use_reranker=req.use_reranker,
        top_k=req.top_k,
    )
    return result


@app.get("/graph", response_class=HTMLResponse)
async def graph_visualization():
    """LangGraph 流程图可视化（Mermaid）"""
    try:
        mermaid_png = rag_graph.get_graph().draw_mermaid()
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>RAG Graph</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
</head><body>
<h2>RAG LangGraph 流程图</h2>
<pre class="mermaid">{mermaid_png}</pre>
<script>mermaid.initialize({{startOnLoad:true}});</script>
</body></html>"""
        return html
    except Exception as e:
        return f"<pre>Graph export error: {e}\n\nMermaid source:\n{rag_graph.get_graph().draw_mermaid()}</pre>"


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
    return {"status": "ok", "version": "0.5.0-langgraph"}
