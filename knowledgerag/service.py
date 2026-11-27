"""服务装配：把向量库、检索器、LLM 组装成一个可用的问答服务（ISSUE-005）。

职责：
- 管理分块数据的持久化（data/chunks.json），重启后 BM25 索引还能重建
- 提供 add_text / ask / stream 三个入口给 API 层
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from knowledgerag.answer import Answer, RAGAnswerer, StreamEvent
from knowledgerag.embeddings import Embedder, HashingEmbedder
from knowledgerag.ingest import Chunk, chunk_document
from knowledgerag.llm import ChatModel, DeepSeekChatModel
from knowledgerag.retrieval import HybridRetriever
from knowledgerag.vectorstore import ChromaStore


def _chunk_to_dict(chunk: Chunk) -> dict:
    """显式序列化。

    不用 dataclasses.asdict：它对跨模块 dataclass 的类型推断不稳；
    而且显式写字段，将来改数据结构时一眼看得见影响范围（也便于做版本迁移）。
    """
    return {
        "text": chunk.text,
        "source": chunk.source,
        "title": chunk.title,
        "section": chunk.section,
        "metadata": dict(chunk.metadata),
        "index": chunk.index,
    }


def _dict_to_chunk(item: dict) -> Chunk:
    """显式反序列化，对缺失字段做兜底。"""
    return Chunk(
        text=item["text"],
        source=item["source"],
        title=item.get("title", ""),
        section=item.get("section", ""),
        metadata=dict(item.get("metadata") or {}),
        index=int(item.get("index", 0)),
    )


class AnswerService:
    """问答服务门面。"""

    def __init__(
        self,
        *,
        data_dir: str | Path = "data",
        llm: ChatModel | None = None,
        embedder: Embedder | None = None,
        collection_name: str = "knowledgerag",
    ) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._chunks_file = self._data_dir / "chunks.json"

        self._embedder: Embedder = embedder or HashingEmbedder()

        # 设计说明：向量库用**内存索引**，`data/chunks.json` 才是真源。
        #
        # 为什么不直接持久化向量？
        # chromadb 1.5 的 Rust 持久化内核（PersistentClient）在部分 Windows 环境
        # 会报 `(code: 14) unable to open database file`，无法稳定使用。
        # 而「真源 + 可重建的派生索引」本身就是更稳的架构：
        #   - 分块原文持久化在 JSON，是唯一事实来源
        #   - 向量索引属于派生数据，进程启动时用真源重建即可
        # 代价是启动要重新向量化；分块规模大了以后应换成 pgvector 这类
        # 真正持久化的向量库（后续 issue）。
        self._store = ChromaStore(self._embedder, collection_name=collection_name)
        self._llm: ChatModel = llm or DeepSeekChatModel()

        self._chunks: list[Chunk] = self._load_chunks()
        if self._chunks:
            # 用真源重建向量索引
            self._store.add(self._chunks)
        self._rebuild()

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #
    def _rebuild(self) -> None:
        """重建检索器（BM25 索引依赖全量分块）。"""
        self._retriever = HybridRetriever(self._store, self._chunks)
        self._answerer = RAGAnswerer(self._retriever, self._llm)

    def _load_chunks(self) -> list[Chunk]:
        if not self._chunks_file.exists():
            return []
        try:
            raw = json.loads(self._chunks_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        return [_dict_to_chunk(item) for item in raw]

    def _save_chunks(self) -> None:
        payload = [_chunk_to_dict(chunk) for chunk in self._chunks]
        self._chunks_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ #
    # 写入
    # ------------------------------------------------------------------ #
    def add_text(
        self,
        text: str,
        *,
        source: str = "upload.md",
        metadata: dict | None = None,
    ) -> int:
        """直接录入一段文本（按 Markdown 规则切块）。"""
        temp_name = self._data_dir / source
        temp_name.parent.mkdir(parents=True, exist_ok=True)
        temp_name.write_text(text, encoding="utf-8")
        try:
            return self.add_file(temp_name, metadata=metadata)
        finally:
            temp_name.unlink(missing_ok=True)

    def add_file(self, path: str | Path, *, metadata: dict | None = None) -> int:
        """录入一个文件（md / txt / pdf / docx）。"""
        chunks = chunk_document(path, metadata=metadata)
        if not chunks:
            return 0
        inserted = self._store.add(chunks)
        self._chunks.extend(chunks)
        self._save_chunks()
        self._rebuild()
        return inserted

    # ------------------------------------------------------------------ #
    # 读取
    # ------------------------------------------------------------------ #
    def ask(self, question: str, *, top_k: int = 5, where: dict | None = None) -> Answer:
        return self._answerer.answer(question, top_k=top_k, where=where)

    def stream(self, question: str, *, top_k: int = 5, where: dict | None = None) -> Iterator[StreamEvent]:
        return self._answerer.stream_answer(question, top_k=top_k, where=where)

    @property
    def answerer(self) -> RAGAnswerer:
        return self._answerer

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)


_service: AnswerService | None = None


def get_default_service() -> AnswerService:
    """进程内单例（API 层用）。"""
    global _service
    if _service is None:
        _service = AnswerService()
    return _service
