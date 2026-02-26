# llm-rag

基于分层架构的 RAG（Retrieval-Augmented Generation）示例，已接入千问大模型 API。

## 架构梳理

- `domain`：定义核心实体（`Document`、`RAGResult`）与抽象接口（`EmbeddingModel`、`VectorStore`、`LLMClient`）。
- `application`：`RAGPipeline` 负责编排 ingest / retrieve / generate 流程。
- `infrastructure`：提供可替换实现（内存向量库、Bag-of-Words embedding、`QwenLLM` API 客户端）。
- `main`：组装依赖并提供 demo 入口。

## 千问 API 配置（只需 API Key）

```bash
export DASHSCOPE_API_KEY="你的apikey"
# 或者 export QWEN_API_KEY="你的apikey"
```

## 快速开始

```bash
PYTHONPATH=src python -m llm_rag.main
pytest
```

## 目录结构

```text
src/llm_rag/
  domain/
  application/
  infrastructure/
  main.py
tests/
```
