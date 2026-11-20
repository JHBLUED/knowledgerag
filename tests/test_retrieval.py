"""ISSUE-004 混合检索与重排的测试。

BM25 部分只依赖 rank-bm25 + jieba（纯 Python，不联网）。
向量部分用 HashingEmbedder + 内存版 Chroma。
"""
from __future__ import annotations

from uuid import uuid4

import pytest

from knowledgerag.embeddings import HashingEmbedder
from knowledgerag.ingest import Chunk
from knowledgerag.retrieval import (
    BM25Index,
    HybridRetriever,
    NoopReranker,
    reciprocal_rank_fusion,
)
from knowledgerag.vectorstore import ChromaStore, SearchHit


def _chunks() -> list[Chunk]:
    data = [
        ("员工请假需提前一天在系统提交申请，由直属主管审批。", "HR"),
        ("报销需提供发票原件和审批单，财务部三个工作日内处理。", "FIN"),
        ("工单 TK-8831-ZQ 已处理完成，等待客户确认。", "IT"),
        ("年假按工龄计算，满一年可享受五天带薪年假。", "HR"),
        ("设备领用请到行政部登记，归还时需检查完好。", "ADMIN"),
    ]
    return [
        Chunk(
            text=text,
            source="制度.md",
            title="员工手册",
            section=f"第{i + 1}条",
            metadata={"dept": dept},
            index=i,
        )
        for i, (text, dept) in enumerate(data)
    ]


@pytest.fixture()
def store() -> ChromaStore:
    return ChromaStore(
        embedder=HashingEmbedder(),
        collection_name=f"test_{uuid4().hex}",
    )


@pytest.fixture()
def retriever(store: ChromaStore) -> HybridRetriever:
    chunks = _chunks()
    store.add(chunks)
    return HybridRetriever(store, chunks)


# --------------------------------------------------------------------------- #
# BM25
# --------------------------------------------------------------------------- #
def test_bm25_finds_exact_keyword():
    """向量检索对精确编号很弱，BM25 必须能一击命中。"""
    index = BM25Index(_chunks())

    hits = index.search("TK-8831-ZQ", top_k=3)

    assert hits, "应命中包含工单号的片段"
    assert "TK-8831-ZQ" in hits[0].text


def test_bm25_handles_chinese_query():
    index = BM25Index(_chunks())

    hits = index.search("请假流程", top_k=3)

    assert hits
    assert "请假" in hits[0].text


def test_bm25_respects_where_filter():
    index = BM25Index(_chunks())

    hits = index.search("报销", top_k=5, where={"dept": "FIN"})

    assert hits
    assert all(hit.metadata["dept"] == "FIN" for hit in hits)


def test_bm25_returns_empty_for_unknown_term():
    index = BM25Index(_chunks())

    assert index.search("完全不存在的词汇组合") == []


def test_bm25_keeps_source_and_section():
    index = BM25Index(_chunks())

    hit = index.search("请假", top_k=1)[0]

    assert hit.source == "制度.md"
    assert hit.section.startswith("第")
    assert hit.id.startswith("制度.md#")


# --------------------------------------------------------------------------- #
# RRF 融合
# --------------------------------------------------------------------------- #
def _hit(hit_id: str, score: float = 0.0) -> SearchHit:
    return SearchHit(text=hit_id, score=score, id=hit_id)


def test_rrf_prefers_document_ranked_high_everywhere():
    both = [_hit("A"), _hit("B"), _hit("C")]
    other = [_hit("A"), _hit("C"), _hit("B")]

    fused = reciprocal_rank_fusion([both, other])

    assert fused[0].id == "A"


def test_rrf_only_uses_ranks_not_scores():
    """两路分数量纲完全不同，RRF 应给出相同顺序。"""
    ranking1 = [_hit("A", 0.9), _hit("B", 0.89), _hit("C", 0.88)]
    ranking2 = [_hit("A", 0.001), _hit("B", 0.0009), _hit("C", 0.0008)]

    fused = reciprocal_rank_fusion([ranking1, ranking2])

    assert [hit.id for hit in fused] == ["A", "B", "C"]


def test_rrf_top_k_truncates():
    fused = reciprocal_rank_fusion([[_hit("A"), _hit("B"), _hit("C")]], top_k=2)

    assert len(fused) == 2


def test_rrf_empty_input():
    assert reciprocal_rank_fusion([]) == []


# --------------------------------------------------------------------------- #
# 重排
# --------------------------------------------------------------------------- #
def test_noop_reranker_truncates():
    hits = [_hit("A"), _hit("B"), _hit("C")]

    assert [h.id for h in NoopReranker().rerank("q", hits, top_k=2)] == ["A", "B"]


# --------------------------------------------------------------------------- #
# HybridRetriever
# --------------------------------------------------------------------------- #
def test_hybrid_hits_exact_keyword(retriever: HybridRetriever):
    """混合检索里，精确编号必须出现在结果中。"""
    hits = retriever.search("TK-8831-ZQ", top_k=3)

    assert hits
    assert any("TK-8831-ZQ" in hit.text for hit in hits)


def test_hybrid_hits_semantic_paraphrase(retriever: HybridRetriever):
    """语义相近的问法也应命中（向量路的贡献）。"""
    hits = retriever.search("我要怎么申请休假", top_k=3)

    assert any("请假" in hit.text or "年假" in hit.text for hit in hits)


def test_vector_mode_and_bm25_mode_both_work(retriever: HybridRetriever):
    assert retriever.search("报销", top_k=3, mode="vector")
    assert retriever.search("报销", top_k=3, mode="bm25")


def test_where_filter_applies_in_hybrid(retriever: HybridRetriever):
    hits = retriever.search("制度", top_k=5, where={"dept": "HR"})

    assert hits
    assert all(hit.metadata["dept"] == "HR" for hit in hits)


def test_invalid_mode_raises(retriever: HybridRetriever):
    with pytest.raises(ValueError):
        retriever.search("x", mode="bogus")


def test_empty_result_when_nothing_matches(retriever: HybridRetriever):
    assert retriever.search("量子纠缠拓扑绝缘体", top_k=3, mode="bm25") == []
