"""应用配置：从环境变量 / .env 读取。

原则：所有可配置项集中在这一处，业务代码不直接读 os.environ。
密钥只读进内存，永远不写日志、不返回给前端。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# 加载项目根目录的 .env（若存在）；已存在的环境变量优先，不会被覆盖。
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """不可变配置对象。"""

    # 应用
    app_name: str = "KnowledgeRAG"

    # 模型（DeepSeek，OpenAI 兼容协议）
    deepseek_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    deepseek_base_url: str = field(default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    model: str = field(default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    temperature: float = field(default_factory=lambda: float(os.getenv("DEEPSEEK_TEMPERATURE", "0")))


settings = Settings()
