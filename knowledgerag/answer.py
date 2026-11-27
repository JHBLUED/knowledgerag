"""RAG 生成与引用溯源（ISSUE-005）。

三件事：
1. 把检索到的分块编号成 [1] [2] ...，拼成 prompt（资料优先，不许编造）
2. 让模型回答并标注来源编号
3. 从回答里把 [n] 解析回来，映射成可展示的引用

「检索不到就明说」是硬要求——企业最怕的是编造答案。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator, Protocol

from knowledgerag.llm import ChatModel
from knowledgerag.vectorstore import SearchHit

NO_ANSWER = "根据现有资料无法回答这个问题。"

SYSTEM_PROMPT = (
    "你是企业内部知识库助手。\n"
    "规则：\n"
    "1. 只能依据【资料】回答，不得使用资料以外的知识，也不要编造。\n"
    "2. 每条结论后用方括号标注来源编号，例如 [1] 或 [1][2]。\n"
    f"3. 如果资料不足以回答，直接回复：{NO_ANSWER}\n"
    "4. 用中文回答，简洁准确。"
)

_CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Citation:
    """一条引用。"""

    index: int
    source: str = ""
    section: str = ""
    snippet: str = ""


@dataclass
class Answer:
    """一次问答的完整结果。"""

    text: str
    citations: list[Citation] = field(default_factory=list)
    retrieved: int = 0


@dataclass
class StreamEvent:
    """流式事件：delta 是增量文本，done=True 时携带最终引用。"""

    delta: str = ""
    done: bool = False
    citations: list[Citation] = field(default_factory=list)


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 5, *, where: dict | None = None) -> list[SearchHit]:
        ...


# --------------------------------------------------------------------------- #
# prompt 构造
# --------------------------------------------------------------------------- #
def build_context(hits: list[SearchHit]) -> str:
    """把检索结果拼成带编号的资料块。"""
    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        origin = " / ".join(part for part in (hit.source, hit.section) if part)
        blocks.append(f"[{index}]（来源：{origin}）\n{hit.text}")
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list[SearchHit]) -> list[dict[str, str]]:
    """构造发给模型的消息列表。"""
    user_content = f"【资料】\n{build_context(hits)}\n\n【问题】\n{question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# --------------------------------------------------------------------------- #
# 引用解析
# --------------------------------------------------------------------------- #
def extract_citations(text: str, hits: list[SearchHit]) -> list[Citation]:
    """从回答文本里提取 [n]，按出现顺序去重并映射回检索结果。"""
    order: list[int] = []
    for match in _CITATION_RE.finditer(text):
        index = int(match.group(1))
        if 1 <= index <= len(hits) and index not in order:
            order.append(index)

    citations: list[Citation] = []
    for index in order:
        hit = hits[index - 1]
        citations.append(
            Citation(
                index=index,
                source=hit.source,
                section=hit.section,
                snippet=hit.text[:120],
            )
        )
    return citations


# --------------------------------------------------------------------------- #
# 问答器
# --------------------------------------------------------------------------- #
class RAGAnswerer:
    """检索 → 生成 → 引用溯源。"""

    def __init__(self, retriever: Retriever, llm: ChatModel) -> None:
        self._retriever = retriever
        self._llm = llm

    def retrieve(self, question: str, *, top_k: int = 5, where: dict | None = None) -> list[SearchHit]:
        return self._retriever.search(question, top_k=top_k, where=where)

    def answer(self, question: str, *, top_k: int = 5, where: dict | None = None) -> Answer:
        """一次性问答。检索不到时**不调用模型**，直接返回固定话术。"""
        hits = self.retrieve(question, top_k=top_k, where=where)
        if not hits:
            return Answer(text=NO_ANSWER, citations=[], retrieved=0)

        text = self._llm.complete(build_messages(question, hits))
        return Answer(
            text=text,
            citations=extract_citations(text, hits),
            retrieved=len(hits),
        )

    def stream_answer(
        self,
        question: str,
        *,
        top_k: int = 5,
        where: dict | None = None,
    ) -> Iterator[StreamEvent]:
        """流式问答：先逐块吐文本，最后一帧带引用。"""
        hits = self.retrieve(question, top_k=top_k, where=where)
        if not hits:
            yield StreamEvent(delta=NO_ANSWER, done=True, citations=[])
            return

        parts: list[str] = []
        for delta in self._llm.stream(build_messages(question, hits)):
            parts.append(delta)
            yield StreamEvent(delta=delta)

        yield StreamEvent(done=True, citations=extract_citations("".join(parts), hits))
