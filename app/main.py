# FastAPI 入口
import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.config import DOCS_DIR
from app.rag_engine import connect_milvus, create_collection, ingest_file, query_rag

app = FastAPI(title="RAG Demo", version="0.1.0")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


@app.on_event("startup")
def startup():
    """启动时连接 Milvus 并创建集合"""
    os.makedirs(DOCS_DIR, exist_ok=True)
    connect_milvus()
    create_collection()


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传文档并入库"""
    if not file.filename.endswith((".txt", ".md")):
        raise HTTPException(400, "目前只支持 .txt 和 .md 文件")

    file_path = os.path.join(DOCS_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks_count = ingest_file(file_path)
    return {"filename": file.filename, "chunks": chunks_count, "message": f"成功入库 {chunks_count} 个文本块"}


@app.post("/query")
async def query(req: QueryRequest):
    """RAG 查询"""
    result = query_rag(req.question, req.top_k)
    return result


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok"}
