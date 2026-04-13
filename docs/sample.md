# RAG 技术简介

## 什么是 RAG

RAG（Retrieval-Augmented Generation）即检索增强生成，是一种将信息检索与大语言模型结合的技术方案。它通过从外部知识库中检索相关信息，为大语言模型提供上下文，从而生成更准确、更有依据的回答。

## RAG 的核心流程

RAG 的工作流程主要分为两个阶段：

### 索引阶段
1. 文档解析：将 PDF、Word、Markdown 等格式的文档转换为纯文本
2. 文本切分：将长文档切分为较小的文本块（chunk），通常 256-1024 个 token
3. 向量化：使用 Embedding 模型将文本块转换为向量表示
4. 存储：将向量和原始文本存入向量数据库

### 查询阶段
1. 用户提出问题
2. 将问题转换为向量
3. 在向量数据库中检索最相似的文本块
4. 将检索到的文本块作为上下文，连同问题一起发送给 LLM
5. LLM 基于上下文生成回答

## RAG 的优势

- 减少幻觉：LLM 基于真实数据回答，而非凭空编造
- 知识更新：只需更新知识库，无需重新训练模型
- 可溯源：回答可以标注信息来源，增强可信度
- 数据安全：私有数据不需要发送给模型训练

## 常用技术栈

- Embedding 模型：bge-large-zh、text-embedding-3-small
- 向量数据库：Milvus、Chroma、Qdrant、pgvector
- LLM：DeepSeek、GPT-4o、Qwen
- 框架：LangChain、LlamaIndex
