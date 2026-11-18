# Issue 列表

> 一次只做一张 issue，每张都对应 SPEC 里的验收标准。
> 完成的 issue 在标题前打 `[x]`。

## 已完成

- [x] **ISSUE-001 项目骨架与健康检查**
  - 建 uv 项目、FastAPI 入口、`/health`、`/chat`（占位）、pytest、CI
  - 验收：`uv run pytest -v` 全绿；`/health` 返回 200 且不回显密钥

## 待办

- [ ] **ISSUE-002 文档解析与切分**
  - 支持 PDF / Word / Markdown，按标题 + 语义切块，保留元数据
  - 依赖：`unstructured` 或 `langchain-community` 分块器
  - 验收：喂入样例文档，能输出带元数据的分块

- [ ] **ISSUE-003 向量化与入库**
  - 用 `bge-m3` 生成向量，写入 pgvector；建索引
  - 依赖：`langchain-openai`（嵌入走兼容接口）或本地 `sentence-transformers`、`pgvector`
  - 验收：按语义检索能命中相关内容

- [ ] **ISSUE-004 混合检索 + 重排**
  - 向量召回 + BM25 召回 → 融合 → 重排（`bge-reranker`）
  - 验收：给定查询，Top-K 结果与人工判断一致，且有出处

- [ ] **ISSUE-005 生成 + 引用溯源 + 流式**
  - 用 DeepSeek 生成带引用编号的回答；`/chat` 支持 SSE 流式
  - 依赖：`langchain`、`langchain-deepseek`
  - 验收：回答可逐字流式返回，附来源列表；无匹配时不编造

- [ ] **ISSUE-006 RBAC + PII 脱敏**
  - 文档级权限过滤；用 `presidio-analyzer` / `presidio-anonymizer` 做 PII 脱敏
  - 验收：越权文档不出现在结果；手机号等被 `***` 替换

- [ ] **ISSUE-007 评估 + 可观测**
  - 接 Langfuse 追踪；准备黄金集，跑 RAGAS 出量化报告
  - 依赖：`langfuse`、`ragas`
  - 验收：能输出召回率 / 忠实度 / 相关性，并有 Docker Compose 部署

## 建议依赖清单（按 issue 逐步添加）

```powershell
# ISSUE-002
uv add unstructured
# ISSUE-003
uv add pgvector sentence-transformers
# ISSUE-004
uv add elasticsearch        # BM25（或先用纯内存 BM25：rank-bm25）
# ISSUE-005
uv add langchain langchain-deepseek
# ISSUE-006
uv add presidio-analyzer presidio-anonymizer
# ISSUE-007
uv add langfuse ragas
```
