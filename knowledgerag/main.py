"""FastAPI 应用入口。

运行：uv run uvicorn knowledgerag.main:app --reload
文档：http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI

from knowledgerag.api.routes import router

app = FastAPI(
    title="KnowledgeRAG",
    description="企业知识库问答服务（旗舰项目一）",
    version="0.1.0",
)

app.include_router(router)
