"""对话接口的测试（用 dependency_overrides 注入假的问答器，不联网）。"""
from __future__ import annotations

from fastapi.testclient import TestClient

from knowledgerag.answer import NO_ANSWER, RAGAnswerer
from knowledgerag.api.routes import get_answerer
from knowledgerag.main import app
from knowledgerag.vectorstore import SearchHit

client = TestClient(app)


class FakeChatModel:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def complete(self, messages):
        return self.reply

    def stream(self, messages):
        yield from self.reply


class FakeRetriever:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits

    def search(self, query: str, top_k: int = 5, *, where: dict | None = None):
        return self.hits


def _make_answerer(reply: str, *, with_hits: bool = True) -> RAGAnswerer:
    hits = (
        [
            SearchHit(
                text="员工请假需提前一天在系统提交申请。",
                score=0.9,
                id="制度.md#0",
                source="制度.md",
                section="考勤 > 请假",
            )
        ]
        if with_hits
        else []
    )
    return RAGAnswerer(FakeRetriever(hits), FakeChatModel(reply))


def test_chat_returns_answer_and_sources():
    app.dependency_overrides[get_answerer] = lambda: _make_answerer("请假需提前一天申请[1]。")
    try:
        resp = client.post("/chat", json={"question": "怎么请假"})
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    data = resp.json()
    assert "请假" in data["answer"]
    assert len(data["sources"]) == 1
    assert data["sources"][0]["index"] == 1
    assert data["sources"][0]["source"] == "制度.md"


def test_chat_returns_no_answer_when_nothing_retrieved():
    app.dependency_overrides[get_answerer] = lambda: _make_answerer("不该被调用", with_hits=False)
    try:
        resp = client.post("/chat", json={"question": "量子纠缠"})
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert resp.json()["answer"] == NO_ANSWER
    assert resp.json()["sources"] == []


def test_chat_requires_question():
    """缺少 question 字段时，应返回 422 校验错误。"""
    resp = client.post("/chat", json={})
    assert resp.status_code == 422


def test_chat_rejects_too_large_top_k():
    resp = client.post("/chat", json={"question": "x", "top_k": 999})
    assert resp.status_code == 422


def test_chat_stream_emits_sse_events():
    app.dependency_overrides[get_answerer] = lambda: _make_answerer("请假[1]要提前")
    try:
        resp = client.post("/chat/stream", json={"question": "怎么请假"})
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    body = resp.text
    assert '"type": "delta"' in body
    assert '"type": "done"' in body
    assert "请假" in body
    assert "制度.md" in body
