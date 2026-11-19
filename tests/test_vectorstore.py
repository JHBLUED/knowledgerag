"""ISSUE-003 向量化与入库的测试。

说明：测试用 HashingEmbedder（离线、确定），用内存版 Chroma（EphemeralClient），
不联网、不下载模型，跑得快。
"""
from __future__ import annotations

from uuid import uuid4

import pytest

from knowledgerag.embeddings import HashingEmbedder
from knowledgerag.ingest import Chunk
from knowledgerag.vectorstore import ChromaStore, SearchHit


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            text="员工请假需提前一天在系统提交申请。",
            source="制度.md",
            title="考勤制度",
            section="考勤制度 > 请假",
            metadata={"dept": "HR"},
            index=0,
        ),
        Chunk(
            text="报销需要提供发票和审批单。",
            source="制度.md",
            title="财务制度",
            section="财务制度 > 报销",
            metadata={"dept": "FIN"},
            index=1,
        ),
        Chunk(
            text="设备领用请找行政部登记。",
            source="制度.md",
            title="行政制度",
            section="行政制度 > 设备",
            metadata={"dept": "ADMIN"},
            index=2,
        ),
    ]


@pytest.fixture()
def store() -> ChromaStore:
    # 每个测试用独立集合名，避免互相污染
    return ChromaStore(
        embedder=HashingEmbedder(),
        collection_name=f"test_{uuid4().hex}",
    )


def test_add_and_count(store: ChromaStore):
    assert store.count() == 0
    inserted = store.add(_chunks())
    assert inserted == 3
    assert store.count() == 3


def test_add_empty_returns_zero(store: ChromaStore):
    assert store.add([]) == 0
    assert store.count() == 0


def test_search_finds_relevant_chunk(store: ChromaStore):
    store.add(_chunks())

    hits = store.search("请假怎么提交", top_k=3)

    assert hits, "应该有检索结果"
    assert isinstance(hits[0], SearchHit)
    assert "请假" in hits[0].text
    # 分数应随时间递减（相似度越高越靠前）
    assert hits[0].score >= hits[-1].score


def test_search_keeps_metadata_and_source(store: ChromaStore):
    store.add(_chunks())

    hit = store.search("请假", top_k=1)[0]

    assert hit.source == "制度.md"
    assert hit.section.startswith("考勤制度")
    assert hit.metadata["dept"] == "HR"


def test_search_respects_where_filter(store: ChromaStore):
    store.add(_chunks())

    hits = store.search("制度", top_k=3, where={"dept": "FIN"})

    assert hits, "FIN 部门应该有数据"
    assert all(h.metadata["dept"] == "FIN" for h in hits)


def test_search_on_empty_store_returns_empty(store: ChromaStore):
    assert store.search("任何问题") == []


def test_top_k_larger_than_data(store: ChromaStore):
    store.add(_chunks())

    hits = store.search("制度", top_k=100)

    assert len(hits) == 3


def test_delete_all(store: ChromaStore):
    store.add(_chunks())
    store.delete_all()
    assert store.count() == 0
