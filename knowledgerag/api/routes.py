"""HTTP 路由。"""
from __future__ import annotations

from fastapi import APIRouter

from knowledgerag.api.schemas import ChatRequest, ChatResponse
from knowledgerag.config import settings

router = APIRouter()


@router.get("/health", tags=["system"])
def health() -> dict:
    """健康检查：返回服务与模型配置状态（不回显密钥）。"""
    return {
        "status": "ok",
        "app": settings.app_name,
        "model": settings.model,
        "api_key_configured": bool(settings.deepseek_api_key),
    }


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(req: ChatRequest) -> ChatResponse:
    """占位对话接口。

    当前仅回显问题；检索（混合检索 + 重排）与生成（引用溯源 + 流式）
    将在后续 issue 中实现。见 issues/ISSUES.md。
    """
    return ChatResponse(
        answer=f"[占位] 收到问题：{req.question}（检索与生成将在后续 issue 实现）",
        sources=[],
    )
