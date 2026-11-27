"""ISSUE-005 生成与引用溯源的测试（纯逻辑，不碰网络和文件系统）。"""
from __future__ import annotations

import pytest

from knowledgerag.answer import (
    NO_ANSWER,
    RAGAnswerer,
    StreamEvent,
    build_context,
    build_messages,
    extract_citations,
)
from knowledgerag.vectorstore import SearchHit


class FakeChatModel:
    """假的模型：不发请求，直接返回预置文本。"""

    def __init__(self, reply: str = "请假需提前一天申请[1]") -> None:
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages):
        self.calls.append(messages)
        return self.reply

    def stream(self, messages):
        self.calls.append(messages)
        yield from self.reply


class FakeRetriever:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits
        self.queries: list[tuple] = []

    def search(self, query: str, top_k: int = 5, *, where: dict | None = None):
        self.queries.append((query, top_k, where))
        return self.hits


def _hits() -> list[SearchHit]:
    return [
        SearchHit(
            text="员工请假需提前一天在系统提交申请。",
            score=0.9,
            id="制度.md#0",
            source="制度.md",
            section="考勤 > 请假",
        ),
        SearchHit(
            text="报销需提供发票原件。",
            score=0.8,
            id="制度.md#1",
            source="制度.md",
            section="财务 > 报销",
        ),
    ]


# --------------------------------------------------------------------------- #
# prompt 构造
# --------------------------------------------------------------------------- #
def test_build_context_has_index_and_source():
    context = build_context(_hits())

    assert "[1]" in context
    assert "[2]" in context
    assert "制度.md" in context
    assert "考勤 > 请假" in context


def test_build_messages_contains_rules_and_question():
    messages = build_messages("怎么请假", _hits())

    assert messages[0]["role"] == "system"
    assert "不得" in messages[0]["content"]  # 包含"不许编造"的约束
    assert messages[1]["role"] == "user"
    assert "怎么请假" in messages[1]["content"]
    assert "员工请假需提前一天" in messages[1]["content"]


# --------------------------------------------------------------------------- #
# 引用解析
# --------------------------------------------------------------------------- #
def test_extract_citations_in_order_and_dedup():
    text = "先说这个[2]，再说那个[1]，又提到[2]。"

    citations = extract_citations(text, _hits())

    assert [c.index for c in citations] == [2, 1]
    assert citations[0].section == "财务 > 报销"


def test_extract_citations_ignores_out_of_range():
    citations = extract_citations("引用[1]和[9]", _hits())

    assert [c.index for c in citations] == [1]


def test_extract_citations_empty_when_no_marks():
    assert extract_citations("没有任何引用的回答", _hits()) == []


# --------------------------------------------------------------------------- #
# 问答
# --------------------------------------------------------------------------- #
def test_answer_returns_text_and_citations():
    retriever = FakeRetriever(_hits())
    llm = FakeChatModel("请假要提前一天提交[1]。")
    answerer = RAGAnswerer(retriever, llm)

    result = answerer.answer("怎么请假", top_k=3)

    assert "提前一天" in result.text
    assert [c.index for c in result.citations] == [1]
    assert result.retrieved == 2
    assert retriever.queries == [("怎么请假", 3, None)]
    assert len(llm.calls) == 1


def test_answer_without_hits_skips_llm():
    """检索不到时必须直接回固定话术，不能调模型（防止编造）。"""
    llm = FakeChatModel("不该被调用")
    answerer = RAGAnswerer(FakeRetriever([]), llm)

    result = answerer.answer("完全不相关的问题")

    assert result.text == NO_ANSWER
    assert result.citations == []
    assert result.retrieved == 0
    assert llm.calls == []


def test_answer_passes_where_filter():
    retriever = FakeRetriever(_hits())
    answerer = RAGAnswerer(retriever, FakeChatModel("x[1]"))

    answerer.answer("请假", where={"dept": "HR"})

    assert retriever.queries[0][2] == {"dept": "HR"}


# --------------------------------------------------------------------------- #
# 流式
# --------------------------------------------------------------------------- #
def test_stream_answer_yields_deltas_then_done():
    answerer = RAGAnswerer(FakeRetriever(_hits()), FakeChatModel("请假[1]要提前"))

    events = list(answerer.stream_answer("怎么请假"))

    assert all(isinstance(e, StreamEvent) for e in events)
    text = "".join(e.delta for e in events)
    assert text == "请假[1]要提前"
    assert events[-1].done is True
    assert [c.index for c in events[-1].citations] == [1]
    assert all(not e.done for e in events[:-1])


def test_stream_answer_without_hits_yields_no_answer():
    events = list(RAGAnswerer(FakeRetriever([]), FakeChatModel()).stream_answer("x"))

    assert len(events) == 1
    assert events[0].done is True
    assert events[0].delta == NO_ANSWER
    assert events[0].citations == []
