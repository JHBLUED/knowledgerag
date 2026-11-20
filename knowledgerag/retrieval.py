"""混合检索与重排（ISSUE-004）。

企业 RAG 的标准做法是「两阶段检索」：

    召回（recall）  向量路 + BM25 路，各自取一批候选
    精排（rerank）  对候选重新打分，取最终 top-k

为什么不能只用向量？
- 向量擅长「语义相近」：问"怎么休假"能命中"请假流程"
- 但对「精确关键词」很弱：工单号 TK-8831-ZQ、错误码 E1024、专有名词
- BM25 正好反过来：精确匹配强、同义词弱

所以两路互补，再用 RRF 融合：
    score(d) = Σ 1 / (k + rank_i(d))

RRF 只用「排名」不用「分数」，天然规避了两种检索器分数量纲不同的问题
（向量是 0~1 余弦相似度，BM25 是无界值）。
"""
from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from knowledgerag.ingest import Chunk
from knowledgerag.vectorstore import ChromaStore, SearchHit


# --------------------------------------------------------------------------- #
# 分词
# --------------------------------------------------------------------------- #
def tokenize(text: str) -> list[str]:
    """把文本切成 token。

    中文必须分词：不过滤就会退化成"按字匹配"，BM25 效果大打折扣。
    """
    import jieba  # 懒加载：jieba 首次加载词典约 1 秒

    tokens = [token.strip().lower() for token in jieba.lcut(text)]
    return [token for token in tokens if token]


# --------------------------------------------------------------------------- #
# BM25 关键词检索
# --------------------------------------------------------------------------- #
def _matches(metadata: dict, where: dict) -> bool:
    """判断元数据是否满足过滤条件（全部相等）。"""
    return all(metadata.get(key) == value for key, value in where.items())


def _to_hit(chunk: Chunk, score: float) -> SearchHit:
    return SearchHit(
        text=chunk.text,
        score=score,
        id=f"{chunk.source}#{chunk.index}",
        source=chunk.source,
        title=chunk.title,
        section=chunk.section,
        metadata=dict(chunk.metadata),
    )


class BM25Index:
    """基于 BM25 的关键词索引（内存版）。"""

    def __init__(self, chunks: list[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        from rank_bm25 import BM25Okapi

        self._chunks: list[Chunk] = list(chunks)
        corpus = [tokenize(chunk.text) for chunk in self._chunks]
        # 空语料会让 BM25 除零，这里兜底
        self._bm25 = BM25Okapi(corpus, k1=k1, b=b) if any(corpus) else None

    def search(self, query: str, top_k: int = 5, where: dict | None = None) -> list[SearchHit]:
        """按关键词打分检索。"""
        if self._bm25 is None:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        scored: list[tuple[float, Chunk]] = []
        for chunk, raw in zip(self._chunks, scores):
            value = float(raw)
            if value <= 0:
                continue
            if where and not _matches(chunk.metadata, where):
                continue
            scored.append((value, chunk))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [_to_hit(chunk, value) for value, chunk in scored[:top_k]]


# --------------------------------------------------------------------------- #
# RRF 融合
# --------------------------------------------------------------------------- #
def reciprocal_rank_fusion(
    rankings: list[list[SearchHit]],
    *,
    k: int = 60,
    top_k: int | None = None,
) -> list[SearchHit]:
    """Reciprocal Rank Fusion：只用排名融合多路结果。

    Args:
        rankings: 多路检索结果，每路已按相关性降序排列。
        k: 平滑常数，常用 60。k 越大，排名差异的影响越平缓。
        top_k: 截断长度。

    Returns:
        融合后的结果，score 是 RRF 分数（越大越靠前，不是概率）。
    """
    if not rankings:
        return []

    scores: dict[str, float] = {}
    representatives: dict[str, SearchHit] = {}

    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            key = hit.id or f"{hit.source}|{hit.section}|{hit.text}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            representatives.setdefault(key, hit)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    fused: list[SearchHit] = []
    for key, score in ordered:
        hit = replace(representatives[key])  # 复制一份，避免改到原始结果
        hit.score = score
        fused.append(hit)

    return fused[:top_k] if top_k else fused


# --------------------------------------------------------------------------- #
# 重排
# --------------------------------------------------------------------------- #
class Reranker(Protocol):
    """重排器协议：拿到候选后重新排序。"""

    def rerank(
        self,
        query: str,
        hits: list[SearchHit],
        *,
        top_k: int | None = None,
    ) -> list[SearchHit]:
        ...


class NoopReranker:
    """不重排，仅截断（默认实现）。

    好处：测试和无模型环境都能跑；真正上 CrossEncoder / LLM 重排时
    换一个实现即可，HybridRetriever 不用改。
    """

    def rerank(
        self,
        query: str,
        hits: list[SearchHit],
        *,
        top_k: int | None = None,
    ) -> list[SearchHit]:
        return list(hits[:top_k] if top_k else hits)


# --------------------------------------------------------------------------- #
# 混合检索器
# --------------------------------------------------------------------------- #
class HybridRetriever:
    """组合「向量召回 + BM25 召回 + RRF 融合 + 重排」。"""

    _MODES = ("hybrid", "vector", "bm25")

    def __init__(
        self,
        store: ChromaStore,
        chunks: list[Chunk],
        *,
        reranker: Reranker | None = None,
        rrf_k: int = 60,
    ) -> None:
        self._store = store
        self._bm25 = BM25Index(chunks)
        self._reranker: Reranker = reranker or NoopReranker()
        self._rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        where: dict | None = None,
        mode: str = "hybrid",
        recall_k: int | None = None,
    ) -> list[SearchHit]:
        """检索。

        Args:
            query: 查询文本。
            top_k: 最终返回条数。
            where: 元数据过滤（权限过滤、部门过滤）。
            mode: `hybrid` / `vector` / `bm25`，便于做效果对比。
            recall_k: 每路召回的候选数，默认 top_k*3（至少 10）。
        """
        if mode not in self._MODES:
            raise ValueError(f"mode 必须是 {self._MODES} 之一，收到：{mode!r}")

        recall = recall_k or max(top_k * 3, 10)

        rankings: list[list[SearchHit]] = []
        if mode in ("hybrid", "vector"):
            rankings.append(self._store.search(query, top_k=recall, where=where))
        if mode in ("hybrid", "bm25"):
            rankings.append(self._bm25.search(query, top_k=recall, where=where))

        rankings = [ranking for ranking in rankings if ranking]
        if not rankings:
            return []

        candidates = (
            rankings[0]
            if len(rankings) == 1
            else reciprocal_rank_fusion(rankings, k=self._rrf_k, top_k=recall)
        )
        return self._reranker.rerank(query, candidates, top_k=top_k)
