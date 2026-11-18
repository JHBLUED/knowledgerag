"""健康检查接口的测试。"""
from fastapi.testclient import TestClient

from knowledgerag.main import app

client = TestClient(app)


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["app"] == "KnowledgeRAG"
    assert isinstance(data["model"], str)


def test_health_does_not_leak_key():
    """健康检查绝不能回显 API Key。"""
    data = client.get("/health").json()
    assert "api_key" not in data
