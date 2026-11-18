"""请求 / 响应的数据模型（Pydantic）。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """对话请求。"""

    question: str = Field(..., description="用户问题")


class Source(BaseModel):
    """回答的引用来源。"""

    document_id: str
    title: str
    snippet: str = ""
    score: float = 0.0


class ChatResponse(BaseModel):
    """对话响应。"""

    answer: str
    sources: list[Source] = Field(default_factory=list, description="引用来源列表")
