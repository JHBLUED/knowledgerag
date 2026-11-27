"""HTTP 路由（ISSUE-005：接入生成 + 引用溯源 + 流式）。"""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from knowledgerag import service as service_module
from knowledgerag.answer import RAGAnswerer
from knowledgerag.api.schemas import (
    ChatRequest,
    ChatResponse,
    IngestRequest,
    IngestResponse,
    Source,
)
from knowledgerag.config import settings
from knowledgerag.service import AnswerService

router = APIRouter()


def get_service() -> AnswerService:
    """FastAPI 依赖：测试可用 app.dependency_overrides 替换掉。"""
    return service_module.get_default_service()


def get_answerer(svc: AnswerService = Depends(get_service)) -> RAGAnswerer:
    return svc.answerer


# --------------------------------------------------------------------------- #
# 系统
# --------------------------------------------------------------------------- #
@router.get("/health", tags=["system"])
def health() -> dict:
    """健康检查：返回服务与模型配置状态（不回显密钥）。"""
    return {
        "status": "ok",
        "app": settings.app_name,
        "model": settings.model,
        "api_key_configured": bool(settings.deepseek_api_key),
    }


# --------------------------------------------------------------------------- #
# 录入
# --------------------------------------------------------------------------- #
@router.post("/documents", response_model=IngestResponse, tags=["ingest"])
def add_document(req: IngestRequest, svc: AnswerService = Depends(get_service)) -> IngestResponse:
    """录入一段文本，切块后写入向量库与 BM25 索引。"""
    inserted = svc.add_text(req.content, source=req.source, metadata=req.metadata)
    return IngestResponse(inserted=inserted, total_chunks=len(svc.chunks))


# --------------------------------------------------------------------------- #
# 问答
# --------------------------------------------------------------------------- #
@router.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(req: ChatRequest, answerer: RAGAnswerer = Depends(get_answerer)) -> ChatResponse:
    """一次性问答：检索 → 生成 → 返回答案与引用。"""
    result = answerer.answer(req.question, top_k=req.top_k)
    return ChatResponse(
        answer=result.text,
        sources=[
            Source(index=c.index, source=c.source, section=c.section, snippet=c.snippet)
            for c in result.citations
        ],
    )


@router.post("/chat/stream", tags=["chat"])
def chat_stream(req: ChatRequest, answerer: RAGAnswerer = Depends(get_answerer)) -> StreamingResponse:
    """流式问答（SSE）。

    事件格式：
        data: {"type": "delta", "text": "增量文本"}
        data: {"type": "done", "sources": [...]}
    """

    def event_stream():
        for event in answerer.stream_answer(req.question, top_k=req.top_k):
            if event.done:
                payload = {
                    "type": "done",
                    "sources": [
                        {"index": c.index, "source": c.source, "section": c.section, "snippet": c.snippet}
                        for c in event.citations
                    ],
                }
            else:
                payload = {"type": "delta", "text": event.delta}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
