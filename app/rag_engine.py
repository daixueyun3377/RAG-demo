# RAG 核心引擎
import os
from openai import OpenAI
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from langfuse import Langfuse
from app.config import *


# 初始化客户端
llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
embed_client = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)

# 初始化 Langfuse（可选）
langfuse = None
if LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY:
    langfuse = Langfuse(
        secret_key=LANGFUSE_SECRET_KEY,
        public_key=LANGFUSE_PUBLIC_KEY,
        host=LANGFUSE_HOST,
    )


def connect_milvus():
    """连接 Milvus"""
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)


def create_collection():
    """创建向量集合（如果不存在）"""
    if utility.has_collection(MILVUS_COLLECTION):
        return Collection(MILVUS_COLLECTION)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
    ]
    schema = CollectionSchema(fields, description="RAG document chunks")
    collection = Collection(MILVUS_COLLECTION, schema)

    # 创建索引
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 256},
    }
    collection.create_index("embedding", index_params)
    return collection


def get_embedding(text: str) -> list[float]:
    """调用 Embedding API 获取向量"""
    response = embed_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """简单的文本切分（按字符，带重叠）"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


def ingest_file(file_path: str) -> int:
    """解析文件 → 切分 → Embedding → 存入 Milvus"""
    # 读取文件
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    filename = os.path.basename(file_path)

    # 切分
    chunks = chunk_text(content)
    if not chunks:
        return 0

    # Embedding
    embeddings = []
    for chunk in chunks:
        emb = get_embedding(chunk)
        embeddings.append(emb)

    # 存入 Milvus
    collection = Collection(MILVUS_COLLECTION)
    collection.insert([
        chunks,       # text
        embeddings,   # embedding
        [filename] * len(chunks),  # source
    ])
    collection.flush()

    return len(chunks)


def search(query: str, top_k: int = TOP_K) -> list[dict]:
    """向量检索"""
    query_embedding = get_embedding(query)

    collection = Collection(MILVUS_COLLECTION)
    collection.load()

    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 128}},
        limit=top_k,
        output_fields=["text", "source"],
    )

    docs = []
    for hit in results[0]:
        docs.append({
            "text": hit.entity.get("text"),
            "source": hit.entity.get("source"),
            "score": hit.score,
        })
    return docs


def generate_answer(query: str, contexts: list[dict]) -> str:
    """基于检索结果生成回答"""
    context_text = "\n\n".join(
        [f"[来源: {c['source']}] {c['text']}" for c in contexts]
    )

    messages = [
        {
            "role": "system",
            "content": "你是一个知识库问答助手。请基于以下检索到的内容回答用户问题。"
                       "如果检索内容中没有相关信息，请诚实说明。回答时标注信息来源。",
        },
        {
            "role": "user",
            "content": f"检索到的内容：\n{context_text}\n\n用户问题：{query}",
        },
    ]

    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content


def query_rag(query: str, top_k: int = TOP_K) -> dict:
    """完整的 RAG 查询流程（带 Langfuse 追踪）"""
    trace = None
    if langfuse:
        trace = langfuse.trace(name="rag-query", input=query)

    # 1. 检索
    if trace:
        span = trace.span(name="retrieval", input={"query": query, "top_k": top_k})
    contexts = search(query, top_k)
    if trace:
        span.end(output={"results_count": len(contexts)})

    # 2. 生成
    if trace:
        gen = trace.generation(
            name="llm-generate",
            model=LLM_MODEL,
            input={"query": query, "contexts_count": len(contexts)},
        )
    answer = generate_answer(query, contexts)
    if trace:
        gen.end(output=answer)

    # 3. 完成
    if trace:
        trace.update(output=answer)
        langfuse.flush()

    return {
        "answer": answer,
        "sources": [{"text": c["text"][:200], "source": c["source"], "score": c["score"]} for c in contexts],
    }
