# 用 LangGraph 重写 RAG 全流程：从线性 Chain 到智能状态图｜附完整代码

### 前言

上一篇文章里，我们用 LangChain 搭了一个完整的 RAG 系统：文档切分、混合检索、Query 变换、Reranker，该有的都有了。

但跑了一段时间后，发现几个问题：

- 检索回来的文档**不一定都相关**，噪声文档会干扰 LLM 生成
- LLM 有时候会**编造信息**（幻觉），即使检索到了正确内容
- 整个流程是**线性的**，出了问题没有重试机制，只能一条路走到黑
- 入库时切分策略是**手动指定**的，不同文档特征适合不同策略

这些问题的本质是：**线性 Chain 缺乏决策和反馈能力**。

LangGraph 正好解决这个问题——它把 RAG 流程建模为**状态图**，每个步骤是一个节点，节点之间可以有条件分支、循环重试、兜底降级。

🔗 项目地址：GitHub 搜索 `daixueyun3377/RAG-demo`，本期对标 `dev-LangGraph` 分支

---

### 一、LangGraph 是什么？为什么用它？

#### 1.1 一句话理解

LangGraph 是 LangChain 团队出的**图编排框架**，用有向图（节点 + 边）来编排 LLM 应用的执行流程。

**LangChain LCEL（上一版）：**

```
prompt | llm | parser
```

线性管道，数据从左到右流过，没有分支、没有循环。

**LangGraph（本版）：**

```
StateGraph + Nodes + Conditional Edges
```

状态图，支持条件路由、循环重试、并行执行。

#### 1.2 核心概念

| 概念 | 说明 |
|------|------|
| **State（状态）** | 一个 TypedDict，贯穿整个图的执行，每个节点都能读写它 |
| **Node（节点）** | 一个 Python 函数，接收 state，返回 state 的部分更新 |
| **Edge（边）** | 节点之间的连接，分为普通边和条件边 |
| **Conditional Edge** | 根据 state 的值动态决定下一步走哪个节点（核心能力） |

#### 1.3 对比：这次升级了什么？

| 能力 | LangChain LCEL 版 | LangGraph 版 |
|------|-------------------|-------------|
| 执行流程 | 线性管道 | 状态图（条件分支 + 循环） |
| 文档质量控制 | ❌ 检索到什么用什么 | ✅ LLM 逐篇评估相关性 |
| 幻觉检测 | ❌ 无 | ✅ 生成后验证是否有依据 |
| 失败处理 | ❌ 直接报错 | ✅ 自动重试 + 兜底回答 |
| 入库策略 | 手动指定 | 规则分析 + 质量验证 + 降级重试 |
| 可观测性 | Langfuse 追踪 | Langfuse + graph_steps 执行轨迹 |
| 流程可视化 | ❌ 无 | ✅ Mermaid 图自动导出 |

---

### 二、项目结构

```
├── app/
│   ├── config.py          # 配置管理（环境变量）
│   ├── main.py            # FastAPI 入口，定义 API 接口
│   └── rag_graph.py       # LangGraph RAG 核心引擎（重点）
├── docs/
│   └── sample.md          # 示例文档
├── rag-infra/
│   └── docker-compose.yml # 基础设施一键部署
├── static/
│   └── chunks.html        # 切分策略对比页面
├── .env.example           # 环境变量模板
└── requirements.txt       # Python 依赖（新增 langgraph）
```

相比上一版，核心变化就是 `rag_engine.py` → `rag_graph.py`，从线性引擎变成了图引擎。

---

### 三、RAG 查询图：7 个节点的完整流程

这是整个项目的核心。先看全局流程图：

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

#### 3.1 State 定义：图的全局状态

```python
class RAGState(TypedDict):
    """RAG 图的全局状态"""
    question: str                # 用户原始问题
    search_query: str            # 变换后的检索 query
    retrieval_mode: str          # vector / bm25 / hybrid
    query_transform: str         # none / rewrite / hyde
    use_reranker: bool
    top_k: int
    retrieved_docs: list         # 检索到的文档
    relevant_docs: list          # 经过相关性评估后的文档
    answer: str                  # 生成的回答
    hallucination_pass: bool     # 幻觉检测是否通过
    retry_count: int             # 重试次数
    sources: list                # 来源信息
    steps: list                  # 执行步骤记录（用于调试）
```

State 就像一个**全局黑板**，每个节点从上面读数据、往上面写结果。`steps` 字段记录了每一步的执行轨迹，方便调试和可视化。

#### 3.2 Node 1：Query 变换（transform_query）

和上一版一样，支持三种模式：直接透传、Query Rewrite、HyDE。

```python
def transform_query(state: RAGState) -> dict:
    question = state["question"]
    mode = state.get("query_transform", "none")
    llm = get_llm()

    if mode == "rewrite":
        prompt = ChatPromptTemplate.from_template(
            "你是一个搜索查询优化助手。请将用户的问题改写为更适合在知识库中检索的查询语句。"
            "只输出改写后的查询，不要解释。\n\n用户问题：{question}"
        )
        search_query = (prompt | llm | StrOutputParser()).invoke({"question": question})
        return {"search_query": search_query, "steps": state.get("steps", []) + [f"query_rewrite → {search_query}"]}

    elif mode == "hyde":
        prompt = ChatPromptTemplate.from_template(
            "请针对以下问题写一段简短的回答（约100字），即使你不确定也请尝试回答。"
            "这段回答将用于检索相关文档。\n\n问题：{question}"
        )
        search_query = (prompt | llm | StrOutputParser()).invoke({"question": question})
        return {"search_query": search_query, "steps": state.get("steps", []) + [f"hyde → {search_query[:80]}..."]}

    else:
        return {"search_query": question, "steps": state.get("steps", []) + ["query_passthrough"]}
```

> **LangGraph 节点的返回值是 state 的部分更新**，不需要返回完整 state。框架会自动合并。


#### 3.3 Node 2：检索（retrieve）

支持三种检索模式，和上一版完全一致：

- **vector**：纯向量检索（语义匹配）
- **bm25**：纯关键词检索（精确匹配）
- **hybrid**：混合检索 + RRF 融合（推荐）

```python
def retrieve(state: RAGState) -> dict:
    query = state["search_query"]
    mode = state.get("retrieval_mode", "hybrid")
    top_k = state.get("top_k", TOP_K)

    if mode == "vector":
        docs = _get_vector_retriever(top_k).invoke(query)
    elif mode == "bm25":
        retriever = _get_bm25_retriever(top_k)
        docs = retriever.invoke(query) if retriever else []
    else:
        docs = _hybrid_retrieve(query, top_k)

    return {
        "retrieved_docs": docs,
        "steps": state.get("steps", []) + [f"retrieve({mode}) → {len(docs)} docs"],
    }
```

混合检索的 RRF 融合权重是 **向量 0.6 + BM25 0.4**，向量检索在大多数场景下效果更好（能理解语义），但关键词检索在精确匹配场景不可替代。

#### 3.4 Node 3：文档相关性评估（grade_documents）⭐ 新增

> 这是 LangGraph 版的**第一个核心升级**。上一版检索到什么就用什么，这一版会让 LLM 逐篇判断文档是否与问题相关，过滤掉噪声文档。

```python
def grade_documents(state: RAGState) -> dict:
    question = state["question"]
    docs = state["retrieved_docs"]
    llm = get_llm()

    grade_prompt = ChatPromptTemplate.from_template(
        "你是一个文档相关性评估专家。判断以下文档是否与用户问题相关。\n"
        "只回答 'yes' 或 'no'，不要解释。\n\n"
        "用户问题：{question}\n\n"
        "文档内容：{document}"
    )
    chain = grade_prompt | llm | StrOutputParser()

    relevant = []
    for doc in docs:
        score = chain.invoke({"question": question, "document": doc.page_content}).strip().lower()
        if score.startswith("yes"):
            relevant.append(doc)

    return {
        "relevant_docs": relevant,
        "steps": state.get("steps", []) + [f"grade_documents → {len(relevant)}/{len(docs)} relevant"],
    }
```

评估完之后，通过**条件路由**决定下一步：

```python
def route_after_grading(state: RAGState) -> str:
    """文档评估后的路由：有相关文档 → rerank，无相关文档 → fallback"""
    if state.get("relevant_docs"):
        return "rerank"
    return "fallback"
```

这就是 LangGraph 的核心能力——**条件边（Conditional Edge）**。线性 Chain 做不到这一点。

#### 3.5 Node 4：Reranker 重排序（rerank）

可选节点。如果开启了 `use_reranker`，会调用硅基流动的 BGE Reranker 对文档做二次排序：

```python
def rerank(state: RAGState) -> dict:
    question = state["question"]
    docs = state["relevant_docs"]
    top_k = state.get("top_k", TOP_K)

    if not state.get("use_reranker", False) or not docs:
        return {"relevant_docs": docs, "steps": state.get("steps", []) + ["rerank_skipped"]}

    try:
        reranked = _rerank_documents(question, docs, top_k)
        return {"relevant_docs": reranked, "steps": state.get("steps", []) + [f"rerank → {len(reranked)} docs"]}
    except Exception as e:
        logger.warning(f"Reranker failed: {e}, using original order")
        return {"relevant_docs": docs, "steps": state.get("steps", []) + [f"rerank_failed: {e}"]}
```

注意这里的容错设计：Reranker 调用失败不会中断流程，而是降级使用原始排序继续执行。

#### 3.6 Node 5：LLM 生成回答（generate）

```python
def generate(state: RAGState) -> dict:
    question = state["question"]
    docs = state["relevant_docs"]
    llm = get_llm()

    # 格式化文档
    context_parts = []
    for doc in docs:
        source = doc.metadata.get("source", "未知")
        context_parts.append(f"[来源: {source}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_template(
        "你是一个知识库问答助手。请基于以下检索到的内容回答用户问题。\n"
        "如果检索内容中没有相关信息，请诚实说明。回答时标注信息来源。\n\n"
        "检索到的内容：\n{context}\n\n"
        "用户问题：{question}\n\n回答："
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question}, config={"callbacks": callbacks})

    sources = [{"text": doc.page_content[:200], "source": doc.metadata.get("source", "未知")} for doc in docs]

    return {"answer": answer, "sources": sources, "steps": state.get("steps", []) + ["generate"]}
```

和上一版的区别：这里用的是经过**相关性评估后的 `relevant_docs`**，而不是原始的 `retrieved_docs`，噪声已经被过滤掉了。

#### 3.7 Node 6：幻觉检测（check_hallucination）⭐ 新增

> 这是 LangGraph 版的**第二个核心升级**。生成回答后，让 LLM 验证回答是否有文档依据。

```python
def check_hallucination(state: RAGState) -> dict:
    answer = state["answer"]
    docs = state["relevant_docs"]
    llm = get_llm()

    doc_contents = "\n\n".join([doc.page_content for doc in docs])

    prompt = ChatPromptTemplate.from_template(
        "你是一个事实核查专家。请判断以下回答是否完全基于提供的参考文档，没有编造信息。\n"
        "只回答 'yes'（有依据）或 'no'（存在编造），不要解释。\n\n"
        "参考文档：\n{documents}\n\n"
        "回答：\n{answer}"
    )
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"documents": doc_contents, "answer": answer}).strip().lower()

    passed = result.startswith("yes")
    retry_count = state.get("retry_count", 0)

    return {
        "hallucination_pass": passed,
        "retry_count": retry_count + (0 if passed else 1),
        "steps": state.get("steps", []) + [f"hallucination_check → {'pass' if passed else 'fail'}"],
    }
```

幻觉检测后的路由逻辑：

```python
def route_after_hallucination(state: RAGState) -> str:
    if state.get("hallucination_pass", False):
        return "finish"        # 通过 → 返回结果
    if state.get("retry_count", 0) < 2:
        return "regenerate"    # 未通过且重试<2次 → 重新生成
    return "fallback"          # 多次失败 → 兜底回答
```

这里形成了一个**循环**：`generate → check_hallucination → generate`，最多重试 2 次。这是线性 Chain 完全做不到的。

#### 3.8 Node 7：兜底回答（fallback）

```python
def fallback(state: RAGState) -> dict:
    return {
        "answer": "抱歉，知识库中没有找到与您问题相关的可靠信息。请尝试换个问法，或上传更多相关文档。",
        "hallucination_pass": True,
        "steps": state.get("steps", []) + ["fallback"],
    }
```

两种情况会触发 fallback：
1. 文档评估后发现**没有相关文档**
2. 幻觉检测**多次失败**（重试 2 次仍然不通过）


#### 3.9 构建状态图：把节点串起来

```python
def build_rag_graph() -> StateGraph:
    graph = StateGraph(RAGState)

    # 添加节点
    graph.add_node("transform_query", transform_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rerank", rerank)
    graph.add_node("generate", generate)
    graph.add_node("check_hallucination", check_hallucination)
    graph.add_node("fallback", fallback)

    # 普通边：固定流向
    graph.add_edge(START, "transform_query")
    graph.add_edge("transform_query", "retrieve")
    graph.add_edge("retrieve", "grade_documents")

    # 条件边：文档评估后
    graph.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {"rerank": "rerank", "fallback": "fallback"},
    )

    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", "check_hallucination")

    # 条件边：幻觉检测后
    graph.add_conditional_edges(
        "check_hallucination",
        route_after_hallucination,
        {"finish": END, "regenerate": "generate", "fallback": "fallback"},
    )

    graph.add_edge("fallback", END)

    return graph.compile()

# 编译图（模块级单例）
rag_graph = build_rag_graph()
```

整个图的构建过程非常直观：
1. `add_node` 注册节点
2. `add_edge` 添加普通边（固定流向）
3. `add_conditional_edges` 添加条件边（动态路由）
4. `compile()` 编译成可执行的图

编译后的 `rag_graph` 可以直接 `.invoke(state)` 执行，也可以 `.get_graph().draw_mermaid()` 导出 Mermaid 流程图。

---

### 四、智能入库图：自动选策略 + 质量验证 + 降级重试

上一版入库时，切分策略是用户手动指定的。但不同文档适合不同策略：

- **结构化文档**（有标题、列表、代码块）→ recursive 效果最好
- **长段落纯文本**（少标记、段落长）→ semantic 语义切分更合适
- **纯文本流**（无段落、无标记）→ fixed 固定切分即可

这一版新增了一个**智能入库图**，自动分析文档特征、选择策略、验证质量、失败降级。

#### 4.1 入库图流程

```
START → load_document → analyze_document → split_document → validate_chunks
                                                                 ↓
                                                       ┌── 质量合格? ──┐
                                                       ↓ yes           ↓ no
                                                 store_document   fallback_strategy
                                                       ↓                ↓
                                                      END        split_document (重试)
```

#### 4.2 文档特征分析（纯规则，零 LLM 调用）

这里有个设计决策：文档分析**不用 LLM**，而是用纯规则启发式。原因是入库是高频操作，每次都调 LLM 太贵也太慢。

```python
def _analyze_doc_features(content: str) -> dict:
    """纯规则启发式分析文档特征，零 LLM 调用，零延迟"""
    total_len = len(content)

    # 特征提取
    heading_count = len(re.findall(r"^#{1,6}\s+", content, re.MULTILINE))  # 标题数
    list_count = len(re.findall(r"^[\s]*[-*]\s+", content, re.MULTILINE))  # 列表项数
    code_block_count = content.count("```") // 2                            # 代码块数
    table_row_count = len(re.findall(r"^\|.+\|", content, re.MULTILINE))   # 表格行数

    # 结构化标记密度（每千字符的标记数）
    structure_marks = heading_count + list_count + code_block_count + table_row_count
    structure_density = structure_marks / (total_len / 1000)

    # 决策逻辑
    if structure_density > 3:
        strategy = "recursive"  # 高结构化 → 递归切分
    elif structure_density < 1 and avg_para_len > 500:
        strategy = "semantic"   # 低结构化 + 长段落 → 语义切分
    elif structure_density < 0.5 and newline_density < 5:
        strategy = "fixed"      # 纯文本流 → 固定切分
    else:
        strategy = "recursive"  # 默认

    return {"strategy": strategy, "chunk_size": chunk_size, "reason": reason}
```

分析维度包括：标题密度、列表密度、代码块数量、段落平均长度、换行密度。根据这些特征组合选择最佳策略和 chunk_size。

#### 4.3 切分质量验证（硬指标）

切分完之后不是直接入库，而是先做质量检查：

```python
_MIN_CHUNKS = 1           # 至少切出 1 个块
_MAX_EMPTY_RATIO = 0.1    # 空块（<10字符）占比不超过 10%
_MIN_AVG_LENGTH = 50      # 平均块长度不低于 50 字符
_MAX_AVG_LENGTH = 3000    # 平均块长度不超过 3000 字符（太长说明没切开）
```

四个硬指标，任何一个不通过就触发降级。

#### 4.4 降级重试

质量不合格时，按 `recursive → fixed → semantic` 的顺序依次尝试：

```python
_FALLBACK_CHAIN = ["recursive", "fixed", "semantic"]

def route_after_validation(state: IngestState) -> str:
    if state.get("quality_pass", False):
        return "store"       # 质量合格 → 入库
    has_untried = any(s not in tried for s in _FALLBACK_CHAIN)
    if has_untried:
        return "fallback"    # 还有策略没试 → 降级
    return "store"           # 全试完了 → 强制入库
```

这又是一个 LangGraph 的循环：`split → validate → fallback → split → validate → ...`，直到质量合格或所有策略都试过。

---

### 五、API 接口

FastAPI 提供了以下接口：

| 接口 | 方法 | 说明 |
|------|------|------|
| `/upload` | POST | 上传文档并入库（手动指定策略） |
| `/smart-upload` | POST | **智能上传**（自动选策略 + 质量验证） |
| `/query` | POST | LangGraph RAG 查询 |
| `/graph` | GET | RAG 查询图的 Mermaid 可视化 |
| `/ingest-graph` | GET | 智能入库图的 Mermaid 可视化 |
| `/compare-chunks` | POST | 切分策略对比 |
| `/health` | GET | 健康检查 |

`/query` 接口的返回值新增了 `graph_steps` 字段，记录了图执行的每一步：

```json
{
  "answer": "RAG 的核心思路是先搜后答...",
  "sources": [...],
  "config": {
    "retrieval_mode": "hybrid",
    "query_transform": "none",
    "use_reranker": false,
    "top_k": 5
  },
  "graph_steps": [
    "query_passthrough",
    "retrieve(hybrid) → 5 docs",
    "grade_documents → 3/5 relevant",
    "rerank_skipped",
    "generate",
    "hallucination_check → pass"
  ]
}
```

通过 `graph_steps` 可以清楚看到：检索了 5 篇文档，经过相关性评估后保留了 3 篇，幻觉检测通过。这对调试和优化非常有用。

---

### 六、快速上手

```bash
# 1. 克隆项目并切换分支
git clone https://github.com/daixueyun3377/RAG-demo.git
cd RAG-demo
git checkout dev-LangGraph

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 DeepSeek 和硅基流动的 API Key

# 4. 启动服务
uvicorn app.main:app --reload --port 8000

# 5. 打开浏览器
# Swagger UI: http://localhost:8000/docs
# RAG 流程图: http://localhost:8000/graph
# 入库流程图: http://localhost:8000/ingest-graph
```

测试流程：

1. 调用 `/smart-upload` 上传 `docs/sample.md`，观察自动选择的切分策略
2. 调用 `/query`，问"RAG 有什么优势"，查看 `graph_steps` 执行轨迹
3. 对比 `use_reranker: true/false` 的效果差异
4. 访问 `/graph` 查看 Mermaid 流程图

---

### 七、我的思考

#### 7.1 LangGraph 的价值

用了 LangGraph 之后，最大的感受是**流程变得可控了**。之前的线性 Chain，数据从头流到尾，中间出了问题只能在最终结果里发现。现在每个节点都有明确的输入输出，条件路由让系统能"做决策"，循环重试让系统能"自我纠错"。

#### 7.2 成本考量

文档评估和幻觉检测都需要额外的 LLM 调用，一次查询可能要调 7-8 次 LLM（5 篇文档评估 + 1 次生成 + 1 次幻觉检测）。如果用 DeepSeek 这种便宜的模型还好，用 GPT-4 就要注意成本了。

可以考虑的优化：
- 文档评估改用小模型或本地模型
- 批量评估（一次调用评估所有文档）
- 只在高置信度场景跳过幻觉检测

#### 7.3 后续计划

- **多轮对话**：在 State 里加入 `chat_history`，实现上下文记忆
- **自适应检索**：根据问题类型自动选择检索模式
- **流式输出**：LangGraph 支持 `astream_events`，可以实现逐 token 输出
- **评估体系**：接入 RAGAS，量化每个节点的效果

---

### 总结

从 LangChain LCEL 到 LangGraph，核心变化是从**线性管道**到**状态图**：

- **文档评估**过滤噪声，让 LLM 只看相关内容
- **幻觉检测**验证回答，发现编造自动重试
- **条件路由**让系统能做决策，不再一条路走到黑
- **智能入库**自动选策略，质量不合格自动降级

RAG 的每个环节都有优化空间，LangGraph 提供了一个很好的框架来组织这些优化——每个优化点是一个节点，节点之间的关系用边来表达，清晰、可控、可扩展。

如果你也在做 RAG，建议把项目跑起来，看看 `graph_steps` 的输出，比光看代码理解深刻得多。

📌 本文为个人学习笔记，如有错误欢迎指正。项目代码持续更新中，欢迎 Star ⭐

