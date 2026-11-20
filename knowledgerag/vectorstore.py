"""向量库：把 Chunk 存进去，按语义检索（ISSUE-003）。

当前实现：**Chroma**（内嵌式，零部署，pip 装上就能跑）。

为什么抽一层接口
----------------
`ChromaStore` 只暴露 `add / search / count / delete_all` 四个方法。
后续要迁到 pgvector 时，只写一个 `PgVectorStore` 实现同样的方法，
业务代码一行都不用改。这类「可替换」的设计在面试里也很好讲。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from knowledgerag.embeddings import Embedder, HashingEmbedder
from knowledgerag.ingest import Chunk

if TYPE_CHECKING:  # 只在类型检查时导入，运行时不依赖 chromadb
    from chromadb.base_types import Metadata


@dataclass
class SearchHit:
    """一条检索结果。

    id 是分块的唯一标识（`文件名#序号`），混合检索融合时用来识别"同一条结果"。
    """

    text: str
    score: float
    id: str = ""
    source: str = ""
    title: str = ""
    section: str = ""
    metadata: dict = field(default_factory=dict)


class ChromaStore:
    """基于 Chroma 的向量库。

    Args:
        embedder: 向量化器；不传则用 HashingEmbedder（离线可用）。
        persist_dir: 传目录则持久化到磁盘；不传则用内存库（适合测试）。
        collection_name: 集合名，不同语料建议分开。
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        *,
        persist_dir: str | Path | None = None,
        collection_name: str = "knowledgerag",
    ) -> None:
        import chromadb  # 懒加载：没装 chromadb 时，仅 import 本模块不会报错

        self.embedder = embedder or HashingEmbedder()
        self._client = (
            chromadb.PersistentClient(path=str(persist_dir))
            if persist_dir
            else chromadb.EphemeralClient()
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _embed_matrix(self, texts: list[str]):
        """把文本批量向量化成 float32 矩阵。

        Chroma 的 Embedding 类型声明是 numpy NDArray（不是 list[float]），
        列表虽然运行时也能用，但类型上不匹配；统一转成 float32 数组，
        既满足类型检查，也符合 Chroma 偏好的输入格式。
        向量化器本身仍只需要返回 list[list[float]]，保持与框架解耦。
        """
        import numpy as np

        return np.asarray(self.embedder.embed(texts), dtype=np.float32)

    # ------------------------------------------------------------------ #
    # 写入
    # ------------------------------------------------------------------ #
    def add(self, chunks: list[Chunk]) -> int:
        """批量写入分块，返回写入条数。"""
        if not chunks:
            return 0

        ids: list[str] = []
        documents: list[str] = []
        # Chroma 要求 metadatas 是 list[Metadata]（Metadata 是 Mapping 别名）。
        # 注意 list 是"不变型"，所以这里必须用 Chroma 的 Metadata 类型标注，
        # 不能写成 list[dict[...]]，否则类型检查器会判定不兼容。
        metadatas: list[Metadata] = []

        for chunk in chunks:
            ids.append(f"{chunk.source}#{chunk.index}")
            documents.append(chunk.text)

            # Chroma 的 metadata 只支持标量，这里先放固定字段，再合并用户元数据
            meta: dict[str, str | int | float | bool] = {
                "source": chunk.source,
                "title": chunk.title,
                "section": chunk.section,
            }
            for key, value in chunk.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    meta[str(key)] = value
            metadatas.append(meta)

        embeddings = self._embed_matrix(documents)
        self._collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(ids)

    # ------------------------------------------------------------------ #
    # 检索
    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str,
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[SearchHit]:
        """按语义检索。

        Args:
            query: 查询文本。
            top_k: 返回条数上限。
            where: 元数据过滤条件，如 {"dept": "HR"}。权限过滤就靠它。
        """
        total = self._collection.count()
        if total == 0:
            return []

        n_results = max(1, min(top_k, total))
        result = self._collection.query(
            query_embeddings=self._embed_matrix([query]),
            n_results=n_results,
            # chromadb 把 where 声明成 TypedDict(Where)，普通 dict 运行时完全可用，
            # 静态检查器会挑刺，这里定向忽略
            where=where or None,  # type: ignore[arg-type]
        )

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[SearchHit] = []
        for i, _doc_id in enumerate(ids):
            meta = dict(metadatas[i] or {})
            # cosine 距离 → 相似度分数（越大越像）
            distance = float(distances[i]) if i < len(distances) else 1.0
            hits.append(
                SearchHit(
                    text=documents[i] or "",
                    score=1.0 - distance,
                    id=ids[i] if i < len(ids) else "",
                    source=str(meta.pop("source", "") or ""),
                    title=str(meta.pop("title", "") or ""),
                    section=str(meta.pop("section", "") or ""),
                    metadata=meta,
                )
            )
        return hits

    def count(self) -> int:
        """当前集合里的分块数量。"""
        return self._collection.count()

    def delete_all(self) -> None:
        """清空集合（测试或重建索引用）。"""
        existing = self._collection.get()
        if existing and existing.get("ids"):
            self._collection.delete(ids=existing["ids"])
