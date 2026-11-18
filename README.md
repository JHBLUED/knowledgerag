# KnowledgeRAG

企业知识库问答服务 —— 旗舰项目一。

面向企业内部文档（制度 / 手册 / FAQ）的精准问答：回答**带引用出处**、**按权限过滤**、**敏感信息脱敏**、**可评估可追踪**。

> 目标岗位：AI 应用开发 / AI 全栈 / Agent 开发。

## 技术栈

- 后端：Python · FastAPI（异步）· LangGraph（后续 issue 接入）
- 模型：DeepSeek（OpenAI 兼容协议）
- 检索：混合检索（向量 + BM25）+ 重排（后续 issue 接入）
- 存储：PostgreSQL + pgvector（后续 issue 接入）
- 可观测 / 评估：Langfuse + RAGAS（后续 issue 接入）
- 依赖与运行：uv

## 快速开始

```powershell
cd D:\deepseek工作区\knowledgerag

# 1) 安装依赖（自动创建 .venv）
uv sync

# 2) 配置密钥（复制后填入真实 Key；.env 不会入库）
Copy-Item .env.example .env

# 3) 启动服务
uv run uvicorn knowledgerag.main:app --reload

# 4) 跑测试
uv run pytest -v
```

启动后访问：

- 健康检查：http://127.0.0.1:8000/health
- 接口文档（Swagger）：http://127.0.0.1:8000/docs

## 目录结构

```
knowledgerag/
├── pyproject.toml          # uv 项目配置与依赖
├── .env.example            # 环境变量模板（密钥）
├── knowledgerag/           # 应用包
│   ├── main.py             # FastAPI 入口
│   ├── config.py           # 集中配置
│   └── api/                # 路由与数据模型
├── tests/                  # 测试（pytest）
├── spec/SPEC.md            # 规格文档（先定验收再实现）
├── issues/ISSUES.md        # 拆解后的任务单
└── .github/workflows/ci.yml # CI（GitHub Actions）
```

## 开发流程（企业式，不是 vibe coding）

本项目按“规格驱动 + 测试门禁 + CI + 评审”推进：

1. 先读 [`spec/SPEC.md`](spec/SPEC.md)，明确“做到什么算对”
2. 从 [`issues/ISSUES.md`](issues/ISSUES.md) 挑一张 issue，一次只做一张
3. 写代码 + 写测试，本地 `uv run pytest -v` 通过
4. 提交 → 开 PR → 走 review → 合并
5. 合并后 CI 自动跑测试门禁

## 当前状态

- ✅ 骨架：FastAPI + `/health` + `/chat`（占位）+ 测试 + CI
- ⬜ 文档解析与切分（ISSUE-002）
- ⬜ 向量化与入库（ISSUE-003）
- ⬜ 混合检索 + 重排（ISSUE-004）
- ⬜ 生成 + 引用溯源 + 流式（ISSUE-005）
- ⬜ RBAC + PII 脱敏（ISSUE-006）
- ⬜ 评估 + 可观测（ISSUE-007）
