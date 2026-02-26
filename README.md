# 项目全生命周期知识库（RAG）方案与实现

本仓库提供一个**可落地的最小可用版本（MVP）**，用于将软件项目全生命周期文档（售前、研发、测试、交付、运维）进行数据资产化，并通过大模型提供问答与 Copilot 能力。

## 1. 目标与场景

### 1.1 目标
- 建立统一知识底座，沉淀项目过程资产。
- 支持自然语言问答（政策、流程、规范、案例）。
- 支持 Copilot（方案撰写、代码规范建议、交付文档生成）。
- 提供可扩展架构，后续可接企业权限、审计、评估体系。

### 1.2 核心场景
- **售前**：快速检索行业方案、标书模板、历史案例。
- **研发**：基于架构规范、编码规范、接口文档做 Copilot 辅助。
- **交付**：自动引用实施手册、SOP、验收标准回答问题。
- **管理**：知识复用率、回答命中率、资产覆盖度统计。

---

## 2. 总体架构设计

```text
            ┌─────────────────────────────────────────────┐
            │                 用户入口层                  │
            │ Web/企业IM/IDE插件/API Gateway             │
            └─────────────────────────────────────────────┘
                               │
                               ▼
            ┌─────────────────────────────────────────────┐
            │              智能应用编排层                  │
            │ 会话管理 | Prompt模板 | 工具调用 | 审计日志   │
            └─────────────────────────────────────────────┘
                               │
                               ▼
            ┌─────────────────────────────────────────────┐
            │                 RAG服务层                    │
            │ Query改写 | 检索 | 重排 | 上下文构建 | 回答生成 │
            └─────────────────────────────────────────────┘
                    │                           │
                    ▼                           ▼
      ┌──────────────────────────┐   ┌──────────────────────────┐
      │      知识数据处理层       │   │      模型与推理层         │
      │ 清洗/切分/Embedding/索引  │   │ 企业模型/云模型/本地模型   │
      └──────────────────────────┘   └──────────────────────────┘
                    │
                    ▼
      ┌────────────────────────────────────────────────────┐
      │               数据与治理层                         │
      │ 文档库(对象存储) | 元数据DB | 向量库 | 权限体系 | 评估 │
      └────────────────────────────────────────────────────┘
```

### 2.1 关键模块说明
1. **文档接入**：支持 `.md/.txt`（MVP），后续可扩展 PDF、Word、Wiki、Git、Jira、Confluence。
2. **数据处理**：清洗（去噪、脱敏）、切块（chunk）、元数据标注（部门、阶段、项目）。
3. **向量化与索引**：将 chunk 转 embedding，写入向量索引并持久化。
4. **检索增强生成（RAG）**：
   - Query 改写（可选）
   - TopK 向量检索
   - 上下文拼接
   - LLM 生成答案并附引用来源
5. **Copilot**：将检索内容注入 Prompt，支持“方案生成/评审建议/交付文档草稿”。
6. **治理与安全**：文档分级、权限过滤、审计日志、模型输出评估。

---

## 3. 数据资产化建模建议

### 3.1 文档分类模型
- `phase`: presale / design / dev / test / delivery / ops
- `domain`: finance / gov / retail / manufacturing ...
- `artifact_type`: proposal / architecture / api / runbook / report
- `project_code`: 项目标识
- `owner`: 责任人
- `version`: 版本号
- `security_level`: public / internal / confidential

### 3.2 元数据策略
- 检索前基于用户身份与项目上下文做 metadata filter。
- 回答必须带出处（chunk_id + source + line_range）。
- 对关键文档（SOP、合同条款）设置高权重或白名单检索。

---

## 4. 参考技术选型

- API 框架：FastAPI
- 向量存储：MVP 使用本地 JSON（便于演示）；生产建议 Milvus / pgvector / OpenSearch
- Embedding：MVP 使用 HashEmbedding（无需外部模型）；生产建议 bge/m3e 或商业 embedding
- LLM：支持 OpenAI 兼容接口（可替换企业私有模型）

---

## 5. 代码实现说明（本仓库）

### 5.1 目录结构

```text
src/
  main.py                 # FastAPI入口
  rag/
    config.py             # 配置加载
    schemas.py            # 请求响应模型
    document_loader.py    # 文档读取与切块
    embedding.py          # Hash Embedding实现
    vector_store.py       # 本地向量库持久化
    retriever.py          # 检索逻辑
    llm_client.py         # 模型调用封装
    service.py            # RAG编排服务
scripts/
  build_kb.py             # 构建知识库脚本
sample_data/
  presale_solution.md
  dev_guideline.md
requirements.txt
```

### 5.2 运行步骤

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 构建知识库：
   ```bash
   python scripts/build_kb.py --input_dir sample_data --db_path data/vector_store.json
   ```
3. 启动服务：
   ```bash
   uvicorn src.main:app --reload --port 8000
   ```
4. 访问接口：
   - 健康检查：`GET /health`
   - 问答：`POST /qa`
   - Copilot：`POST /copilot`

### 5.3 环境变量

| 变量 | 说明 | 默认值 |
|---|---|---|
| `VECTOR_DB_PATH` | 向量库存储文件 | `data/vector_store.json` |
| `EMBED_DIM` | embedding维度 | `256` |
| `TOP_K` | 检索返回文档数 | `4` |
| `OPENAI_BASE_URL` | OpenAI兼容接口地址 | 空 |
| `OPENAI_API_KEY` | 模型调用密钥 | 空 |
| `OPENAI_MODEL` | 模型名 | `gpt-4o-mini` |

> 未配置 OpenAI 时，系统会返回“基于检索上下文的模板答案”，便于离线演示。

---

## 6. 生产落地路线图（建议）

### 阶段1：MVP（2~4周）
- 接入核心文档源（售前方案库 + 开发规范）
- 建立问答接口与来源引用
- 小范围试点（售前+架构组）

### 阶段2：可运营（4~8周）
- 增加权限过滤与审计
- 建立评估集（准确率、召回率、幻觉率）
- 接入企业 IM 和 IDE 插件

### 阶段3：规模化（8周+）
- 多索引分层（公共知识、项目知识、个人知识）
- 引入重排模型与查询路由
- 建立知识运营机制（过期、合并、评分）

---

## 7. API 示例

### 7.1 QA

```bash
curl -X POST http://localhost:8000/qa \
  -H "Content-Type: application/json" \
  -d '{
    "question": "投标技术方案一般包含哪些章节？",
    "filters": {"phase": "presale"}
  }'
```

### 7.2 Copilot

```bash
curl -X POST http://localhost:8000/copilot \
  -H "Content-Type: application/json" \
  -d '{
    "task": "根据规范，生成一个微服务项目的代码评审清单",
    "style": "checklist"
  }'
```

---

## 8. 注意事项

- 生产环境请务必接入权限系统，避免跨项目文档泄漏。
- 对外输出需加“来源引用 + 置信度提示”。
- 对合同条款、报价等高风险内容引入人工复核流程。

