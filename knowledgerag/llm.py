"""LLM 客户端封装（ISSUE-005）。

为什么再抽一层协议
------------------
`ChatModel` 只有两个方法：`complete`（一次性返回）和 `stream`（逐块产出）。
这样测试可以注入 `FakeChatModel`——**不联网、不花钱、离线可跑**；
生产换 DeepSeek / OpenAI / Ollama 都不用改上层代码。
"""
from __future__ import annotations

from typing import Iterator, Protocol

from knowledgerag.config import settings


class ChatModel(Protocol):
    """对话模型协议。messages 形如 [{"role": "user", "content": "..."}]。"""

    def complete(self, messages: list[dict[str, str]]) -> str:
        ...

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        ...


class DeepSeekChatModel:
    """通过 OpenAI 兼容协议调用 DeepSeek。"""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._api_key = api_key or settings.deepseek_api_key
        self._base_url = base_url or settings.deepseek_base_url
        self._model = model or settings.model
        self._temperature = settings.temperature if temperature is None else temperature
        self._timeout = timeout

    def _client(self):
        if not self._api_key:
            raise RuntimeError(
                "未配置 DEEPSEEK_API_KEY。请在项目根目录的 .env 里填写后重启服务。"
            )
        from openai import OpenAI  # 懒加载

        return OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout,
        )

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = self._client().chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self._temperature,
        )
        if not response.choices:
            return ""
        return response.choices[0].message.content or ""

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        stream = self._client().chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self._temperature,
            stream=True,
        )
        for chunk in stream:
            # 注意：最后可能有一帧只带 usage，choices 为空，直接跳过
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
