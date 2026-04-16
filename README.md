# RAG Demo — LangGraph 版

基于 LangGraph 的 RAG（检索增强生成）全链路 Demo。将 RAG 流程建模为状态图，支持文档相关性评估、幻觉检测、条件路由、循环重试和智能入库。

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| 语言 | Python 3.11 |
| Web 框架 | FastAPI |
| RAG 框架 | LangGraph + LangChain |
| 向量数据库 | Chroma（嵌入式） |
| Embedding | BAAI/bge-large-zh-v1.5（硅基流动 API） |
| Reranker | BAAI/bge-reranker-v2-m3（硅基流动 API） |
| LLM | DeepSeek / Claude（OpenAI 兼容接口） |
| 关键词检索 | BM25（rank-bm25） |
| 可观测性 | Langfuse（可选） |

## 架构

```
START → transform_query → retrieve → grade_documents
                                          ↓
                                ┌─── 有相关文档? ───┐
                                ↓ yes               ↓ no
                              rerank             fallback → END
                                ↓
                             generate
                                ↓
                        check_hallucination
                                ↓
                      ┌─── 有依据? ───────────┐
                      ↓ yes      ↓ no(<2次)   ↓ no(≥2次)
                     END     generate(重试)   fallback → END
```

> 流式查询（`/query/stream`）使用专用的精简图，跳过幻觉检测环节，避免已输出 token 被重试覆盖。

## 特性

- **状态图编排**：LangGraph 状态图，支持条件分支、循环重试、兜底降级
- **文档相关性评估**：LLM 批量判断检索文档是否与问题相关，过滤噪声
- **幻觉检测**：生成回答后验证是否有文档依据，不通过自动重试（最多 2 次）
- **智能入库**：规则分析文档特征，自动选择切分策略，质量验证 + 降级重试
- **多策略切分**：fixed / recursive / semantic 三种策略
- **混合检索**：向量 + BM25，RRF 融合
- **Query 变换**：Query Rewrite / HyDE
- **Reranker**：bge-reranker-v2-m3 二次排序（带自动重试和超时降级）
- **流程可视化**：Mermaid 图自动导出（`/graph`、`/ingest-graph`）
- **执行轨迹**：`graph_steps` 记录每一步执行过程
- **可观测性**：Langfuse 全链路追踪（可选）

## 快速开始

### 1. 前置准备

你只需要准备两个 API Key：

| Key | 用途 | 获取方式 |
|-----|------|----------|
| LLM API Key | 大模型生成回答 | DeepSeek、OpenAI 或其他 OpenAI 兼容接口 |
| 硅基流动 API Key | Embedding + Reranker | 免费申请，见下方 |

**硅基流动 API Key 申请：**
1. 访问 [cloud.siliconflow.cn](https://cloud.siliconflow.cn)
2. 注册账号并登录
3. 进入「账户管理」→「API 密钥」，创建一个 Key
4. 新用户赠送免费额度，本项目使用的 bge-large-zh-v1.5（Embedding）和 bge-reranker-v2-m3（Reranker）均为免费模型

### 2. 环境准备

```bash
git clone https://github.com/daixueyun3377/RAG-demo.git
cd RAG-demo
pip install -r requirements.txt
```

### 3. 配置

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
EMBEDDING_DIM=1024

# Reranker（硅基流动，默认复用 Embedding Key）
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

### 4. 启动

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

访问 Swagger UI：http://localhost:8080/docs

### 5. 使用

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
| POST | `/upload` | 上传文档并入库（手动指定策略） |
| POST | `/smart-upload` | 智能上传（自动选策略 + 质量验证 + 降级重试） |
| POST | `/query` | LangGraph RAG 查询（返回 graph_steps 执行轨迹） |
| POST | `/query/stream` | 流式 RAG 查询（SSE，逐 token 输出） |
| GET | `/graph` | RAG 查询图 Mermaid 可视化 |
| GET | `/ingest-graph` | 智能入库图 Mermaid 可视化 |
| POST | `/compare-chunks` | 对比不同切分策略效果 |
| POST | `/compare-chunks-detail` | 对比切分策略（含完整 chunk 内容） |
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
│   ├── config.py          # 配置管理（环境变量）
│   ├── llm.py             # LLM / Embedding / Langfuse 初始化
│   ├── retriever.py       # 文档加载、切分、向量存储、检索器、Reranker
│   ├── ingest.py          # 智能入库 LangGraph 状态图
│   ├── rag_graph.py       # RAG 查询 LangGraph 状态图
│   └── main.py            # FastAPI 入口，定义 API 接口
├── docs/
│   ├── sample.md          # 示例文档
│   └── test.md            # 测试文档
├── rag-infra/
│   ├── docker-compose.yml         # Docker Compose 部署配置
│   ├── docker-compose-Milvus.yml  # Milvus 向量数据库部署配置
│   └── README.md                  # 基础设施说明
├── tests/
│   ├── __init__.py
│   ├── test_rag_graph.py  # 单元测试
│   └── test_integration.py # 集成测试
├── static/
│   └── chunks.html        # 切分策略对比页面
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
