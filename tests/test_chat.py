"""对话接口（占位）的测试。"""
from fastapi.testclient import TestClient

from knowledgerag.main import app

client = TestClient(app)


def test_chat_placeholder():
    resp = client.post("/chat", json={"question": "你好"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"]  # 占位回答非空
    assert data["sources"] == []


def test_chat_requires_question():
    """缺少 question 字段时，应返回 422 校验错误。"""
    resp = client.post("/chat", json={})
    assert resp.status_code == 422
