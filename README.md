# RAG Demo

基于 LangChain 的 RAG（检索增强生成）全链路 Demo，覆盖文档加载、多策略切分、向量存储、混合检索、Reranker 重排序、LLM 生成。

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| 语言 | Python 3.11 |
| Web 框架 | FastAPI |
| RAG 框架 | LangChain 1.x |
| 向量数据库 | Chroma（嵌入式） |
| Embedding | BAAI/bge-large-zh-v1.5（硅基流动 API） |
| Reranker | BAAI/bge-reranker-v2-m3（硅基流动 API） |
| LLM | DeepSeek / Claude（OpenAI 兼容接口） |
| 关键词检索 | BM25（rank-bm25） |
| 可观测性 | Langfuse（可选） |
| 基础设施 | Docker Compose（Milvus + Redis + Langfuse） |

## 架构

```
用户问题
  │
  ├─ Query Transform (可选)
  │   ├─ Query Rewrite (LLM 改写)
  │   └─ HyDE (假设性回答检索)
  │
  ├─ 检索
  │   ├─ Vector (Chroma 向量检索)
  │   ├─ BM25 (关键词检索)
  │   └─ Hybrid (RRF 融合)
  │
  ├─ Reranker (可选, bge-reranker-v2-m3)
  │
  └─ LLM 生成回答
```

## 特性

- **文档加载**：LangChain DocumentLoader（支持 .md / .txt）
- **多策略切分**：
  - `fixed` — 固定长度（CharacterTextSplitter）
  - `recursive` — 递归切分（RecursiveCharacterTextSplitter），按语义边界
  - `semantic` — 语义切分（SemanticChunker），基于 embedding 相似度自动判断边界
- **向量存储**：Chroma（嵌入式，零部署）
- **混合检索**：向量 + BM25，RRF（Reciprocal Rank Fusion）融合
- **Query 变换**：Query Rewrite / HyDE
- **Reranker**：硅基流动 bge-reranker-v2-m3 API，检索后二次排序
- **切分策略对比**：一键对比 fixed/recursive/semantic × 不同 chunk_size 的效果
- **可观测性**：Langfuse 全链路追踪（可选）
- **基础设施**：Docker Compose 一键部署（Milvus + Redis + Langfuse）

## 分支说明

| 分支 | 说明 |
|------|------|
| `main` | LangChain 版（Chroma + 混合检索 + Reranker + 语义切分） |
| `dev-p0` | 原生 Python 版（pymilvus + OpenAI SDK，无框架） |
| `dev-langchain` | LangChain 开发分支（已合并到 main） |

## 快速开始

### 1. 环境准备

```bash
git clone https://github.com/daixueyun3377/RAG-demo.git
cd RAG-demo
pip install -r requirements.txt
```

### 2. 配置

复制 `.env.example` 为 `.env`，填入 API Key：

```bash
cp .env.example .env
```

```env
# LLM（DeepSeek / Claude 代理 / 其他 OpenAI 兼容接口）
LLM_API_KEY=your-llm-api-key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat

# Embedding（硅基流动）
EMBEDDING_API_KEY=your-siliconflow-api-key
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5

# Reranker（硅基流动，默认复用 Embedding Key）
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

### 3. 启动

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

访问 Swagger UI：http://localhost:8080/docs

### 4. 使用

**上传文档：**
```bash
curl -X POST "http://localhost:8080/upload?strategy=recursive&chunk_size=512" \
  -F "file=@docs/sample.md"
```

**RAG 查询：**
```bash
curl -X POST "http://localhost:8080/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "RAG的核心流程是什么？",
    "retrieval_mode": "hybrid",
    "query_transform": "none",
    "use_reranker": true
  }'
```

**对比切分策略：**
```bash
curl -X POST "http://localhost:8080/compare-chunks" \
  -F "file=@docs/sample.md"
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/upload` | 上传文档并入库（支持选择切分策略和参数） |
| POST | `/query` | RAG 查询（支持检索模式、Query 变换、Reranker） |
| POST | `/compare-chunks` | 对比不同切分策略效果 |
| GET | `/health` | 健康检查 |

### /query 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `question` | string | 必填 | 用户问题 |
| `retrieval_mode` | string | `hybrid` | 检索模式：`vector` / `bm25` / `hybrid` |
| `query_transform` | string | `none` | Query 变换：`none` / `rewrite` / `hyde` |
| `use_reranker` | bool | `false` | 是否启用 Reranker 重排序 |
| `top_k` | int | `5` | 返回 Top-K 结果 |

## 项目结构

```
RAG-demo/
├── app/
│   ├── __init__.py
│   ├── config.py          # 配置管理
│   ├── main.py            # FastAPI 入口 + 路由
│   └── rag_engine.py      # RAG 核心引擎
├── docs/
│   └── sample.md          # 示例文档
├── rag-infra/
│   └── docker-compose.yml # 基础设施（Milvus + Redis + Langfuse）
├── .env.example           # 环境变量模板
├── .gitignore
├── requirements.txt
└── README.md
```

## 检索优化策略

### 混合检索（Hybrid Retrieval）
向量检索擅长语义匹配，BM25 擅长关键词匹配。通过 RRF（Reciprocal Rank Fusion）融合两者结果：

```
score(doc) = w_vector * 1/(k + rank_vector) + w_bm25 * 1/(k + rank_bm25)
```

默认权重：向量 60%，BM25 40%，k=60。

### Query 变换
- **Query Rewrite**：让 LLM 将用户口语化问题改写为更适合检索的查询
- **HyDE**（Hypothetical Document Embeddings）：让 LLM 先生成一个假设性回答，用这个回答的 embedding 去检索，比原始问题更接近目标文档

### Reranker
检索阶段追求召回率（recall），Reranker 追求精确率（precision）。用 Cross-Encoder 模型对 query-document pair 做精细打分，比 embedding 余弦相似度更准确。

### 切分策略对比
| 策略 | 特点 | 适用场景 |
|------|------|----------|
| fixed | 固定长度，简单粗暴 | 基线对比 |
| recursive | 按语义边界递归切分 | 通用场景（推荐） |
| semantic | 基于 embedding 相似度自动切分 | 内容主题变化明显的文档 |

## License

MIT
