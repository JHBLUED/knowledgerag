"""向量化：把文本转成向量（ISSUE-003）。

设计原则
--------
1. `Embedder` 只是一个协议（Protocol）：只要能 `embed(texts) -> list[list[float]]` 就是合法实现。
   这样后面换模型（bge-m3、OpenAI embeddings、本地 Ollama）不用改业务代码。
2. `HashingEmbedder`：纯标准库、确定性、离线可用。
   用途是测试和本地开发——不下载模型、不联网、结果稳定。
3. 生产环境可换成真正的语义模型（见 ISSUE 备注），接口不变。
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """向量化器协议。"""

    dimension: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """把一批文本转成向量，返回顺序与输入一致。"""
        ...


# 英文/数字按词切，中文按单字切（中文没有空格，先粗粒度切开）
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def _tokens(text: str) -> list[str]:
    """切出 token，并补上相邻二元组以保留一点语序信息。"""
    raw = _TOKEN_RE.findall(text.lower())
    grams = list(raw)
    grams.extend(a + b for a, b in zip(raw, raw[1:]))
    return grams


class HashingEmbedder:
    """基于哈希的朴素向量化（feature hashing）。

    优点：零依赖、确定、离线、快。
    缺点：只反映「字面重合」，不理解语义（"请假" 和 "休假" 不算相似）。
    所以它只用于测试和打通链路，真正的语义检索要换真模型。
    """

    def __init__(self, dimension: int = 512) -> None:
        if dimension <= 0:
            raise ValueError("dimension 必须是正整数")
        self.dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in _tokens(text):
            # 用 md5 而不是内置 hash()：内置 hash 每个进程随机化，会导致结果不稳定
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        # L2 归一化，让余弦相似度只反映方向
        norm = math.sqrt(sum(v * v for v in vector))
        if norm:
            vector = [v / norm for v in vector]
        return vector
