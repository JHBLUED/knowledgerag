"""请求 / 响应的数据模型（Pydantic）。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """对话请求。"""

    question: str = Field(..., min_length=1, description="用户问题")
    top_k: int = Field(5, ge=1, le=20, description="检索条数上限")


class Source(BaseModel):
    """回答引用的来源。"""

    index: int = 0
    source: str = ""
    section: str = ""
    snippet: str = ""


class ChatResponse(BaseModel):
    """对话响应。"""

    answer: str
    sources: list[Source] = Field(default_factory=list, description="引用来源列表")


class IngestRequest(BaseModel):
    """录入一段文本（本地演示用；生产应改为文件上传 + 队列）。"""

    content: str = Field(..., min_length=1)
    source: str = Field("upload.md", description="文件名，作为来源标识")
    metadata: dict = Field(default_factory=dict)


class IngestResponse(BaseModel):
    """录入结果。"""

    inserted: int
    total_chunks: int
