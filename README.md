# llm-rag

基于分层架构的 RAG（Retrieval-Augmented Generation）最小实现，便于在项目中快速落地。

## 架构梳理

- `domain`：定义核心实体（`Document`、`RAGResult`）与抽象接口（`EmbeddingModel`、`VectorStore`、`LLMClient`）。
- `application`：`RAGPipeline` 负责编排 ingest / retrieve / generate 流程。
- `infrastructure`：提供可替换实现（本项目内置内存向量库、规则生成器、Bag-of-Words embedding）。
- `main`：组装依赖并提供 demo 入口。

## 快速开始

```bash
python -m llm_rag.main
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
