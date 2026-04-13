# FastAPI 入口 - LangChain 版
import os
import shutil
from typing import Literal
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
from app.config import DOCS_DIR
from app.rag_engine import ingest_file, query_rag, compare_chunk_strategies

app = FastAPI(title="RAG Demo - LangChain", version="0.2.0")


class QueryRequest(BaseModel):
    question: str
    retrieval_mode: Literal["vector", "bm25", "hybrid"] = "hybrid"
    query_transform: Literal["none", "rewrite", "hyde"] = "none"
    top_k: int = 5


class IngestRequest(BaseModel):
    strategy: Literal["fixed", "recursive"] = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 50


@app.on_event("startup")
def startup():
    os.makedirs(DOCS_DIR, exist_ok=True)


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    strategy: str = Query("recursive", description="切分策略: fixed / recursive"),
    chunk_size: int = Query(512, description="切分大小"),
    chunk_overlap: int = Query(50, description="重叠大小"),
):
    """上传文档并入库（支持选择切分策略和参数）"""
    if not file.filename.endswith((".txt", ".md")):
        raise HTTPException(400, "目前只支持 .txt 和 .md 文件")

    file_path = os.path.join(DOCS_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = ingest_file(file_path, strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return {"message": f"成功入库 {result['chunks']} 个文本块", **result}


@app.post("/query")
async def query(req: QueryRequest):
    """RAG 查询（支持选择检索模式和 query 变换策略）"""
    result = query_rag(
        question=req.question,
        retrieval_mode=req.retrieval_mode,
        query_transform=req.query_transform,
        top_k=req.top_k,
    )
    return result


@app.post("/compare-chunks")
async def compare_chunks(file: UploadFile = File(...)):
    """对比不同切分策略的效果"""
    if not file.filename.endswith((".txt", ".md")):
        raise HTTPException(400, "目前只支持 .txt 和 .md 文件")

    file_path = os.path.join(DOCS_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = compare_chunk_strategies(file_path)
    return result


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.2.0-langchain"}
