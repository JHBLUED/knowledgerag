"""AnswerService 的测试（写临时目录，验证录入 + 重启重建 + 问答链路）。

注意：chromadb 的 EphemeralClient 在同进程内是共享的，
所以每个测试用**独立的集合名**，避免数据串到别的测试里。
"""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from knowledgerag.service import AnswerService

SAMPLE = """# 考勤制度

## 请假

员工请假需提前一天在系统提交申请，由直属主管审批。

## 报销

报销需提供发票原件，财务部三个工作日内处理。
"""


class FakeChatModel:
    def __init__(self, reply: str = "请假需提前一天提交[1]。") -> None:
        self.reply = reply

    def complete(self, messages):
        return self.reply

    def stream(self, messages):
        yield from self.reply


def _service(
    tmp_path: Path,
    reply: str = "请假需提前一天提交[1]。",
    collection_name: str | None = None,
) -> AnswerService:
    return AnswerService(
        data_dir=tmp_path / "data",
        llm=FakeChatModel(reply),
        collection_name=collection_name or f"test_{uuid4().hex}",
    )


def test_add_text_and_ask(tmp_path: Path):
    svc = _service(tmp_path)

    inserted = svc.add_text(SAMPLE, source="kaoqin.md")

    assert inserted > 0
    answer = svc.ask("怎么请假", top_k=3)
    assert "请假" in answer.text
    assert answer.citations
    assert answer.retrieved >= 1


def test_chunks_persist_across_restart(tmp_path: Path):
    # 同一个集合名：模拟"同一份数据、进程重启"
    name = f"test_{uuid4().hex}"
    svc = _service(tmp_path, collection_name=name)
    svc.add_text(SAMPLE, source="kaoqin.md")
    expected = len(svc.chunks)

    reloaded = _service(tmp_path, collection_name=name)

    # 真源（chunks.json）被读回
    assert len(reloaded.chunks) == expected
    assert reloaded.chunks[0].source == "kaoqin.md"
    # 向量索引在启动时用真源重建过，所以仍能检索到
    answer = reloaded.ask("怎么请假", top_k=3)
    assert answer.retrieved >= 1


def test_ask_without_documents_returns_no_answer(tmp_path: Path):
    svc = _service(tmp_path)

    answer = svc.ask("怎么请假")

    assert "无法回答" in answer.text
    assert answer.citations == []


def test_add_text_twice_accumulates(tmp_path: Path):
    svc = _service(tmp_path)
    svc.add_text(SAMPLE, source="a.md")
    first = len(svc.chunks)

    svc.add_text("# 第二条\n\n设备领用请到行政部登记。\n", source="b.md")

    assert len(svc.chunks) > first


def test_stream_returns_events(tmp_path: Path):
    svc = _service(tmp_path)
    svc.add_text(SAMPLE, source="kaoqin.md")

    events = list(svc.stream("怎么请假"))

    assert events
    assert events[-1].done is True
    assert "请假" in "".join(e.delta for e in events)
