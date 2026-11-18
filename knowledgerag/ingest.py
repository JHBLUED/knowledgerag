"""文档解析与切分（ISSUE-002）。

目标：
- 把 Markdown / 纯文本 / PDF / Word 文档切成带「标题路径 + 元数据」的块
- 切分优先按语义边界（标题 > 段落），段落过长才退化为定长 + 重叠

设计原则：
- 核心切分只用标准库（测试不依赖重型包）
- PDF / Word 的第三方依赖按需懒加载，没装时给出明确提示
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf", ".docx"}

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    """一个文本块。

    section 例："制度 > 1.2 考勤"，让检索结果能显示"出自哪一节"。
    metadata 用来携带部门、密级、来源、日期等权限/过滤字段。
    """

    text: str
    source: str
    title: str = ""
    section: str = ""
    metadata: dict = field(default_factory=dict)
    index: int = 0

    @property
    def chars(self) -> int:
        return len(self.text)


# --------------------------------------------------------------------------- #
# 读取
# --------------------------------------------------------------------------- #
def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:  # pragma: no cover - 依赖缺失分支
        raise ModuleNotFoundError("解析 PDF 需要安装 pypdf：uv add pypdf") from exc
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_docx(path: Path) -> str:
    try:
        import docx  # python-docx
    except ModuleNotFoundError as exc:  # pragma: no cover - 依赖缺失分支
        raise ModuleNotFoundError("解析 Word 需要安装 python-docx：uv add python-docx") from exc
    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)


def read_text(path: str | Path) -> str:
    """读取文档为纯文本。"""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown", ".txt"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix == ".docx":
        return _read_docx(path)
    raise ValueError(f"不支持的文档类型：{suffix or '(无扩展名)'}")


# --------------------------------------------------------------------------- #
# 切分
# --------------------------------------------------------------------------- #
def _split_markdown_sections(text: str) -> list[tuple[str, str]]:
    """按标题切分为 [(标题路径, 正文), ...]。"""
    sections: list[tuple[str, str]] = []
    path: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append((" > ".join(path), body))
        buffer.clear()

    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            path[:] = path[: level - 1]
            path.append(match.group(2).strip())
        else:
            buffer.append(line)
    flush()
    return sections


def _split_paragraphs(text: str) -> list[str]:
    """按空行切段落。"""
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def _hard_split(text: str, max_chars: int, overlap: int) -> list[str]:
    """段落自身超长时，定长硬切（带重叠，避免句子被切断）。"""
    step = max(1, max_chars - overlap)
    return [text[i : i + max_chars] for i in range(0, len(text), step)]


def _pack_paragraphs(paragraphs: list[str], max_chars: int, overlap: int) -> list[str]:
    """把段落打包成不超过 max_chars 的块。"""
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        pieces = [para] if len(para) <= max_chars else _hard_split(para, max_chars, overlap)
        for piece in pieces:
            if not current:
                current = piece
            elif len(current) + 1 + len(piece) <= max_chars:
                current = f"{current}\n{piece}"
            else:
                chunks.append(current)
                tail = current[-overlap:] if overlap and len(current) > overlap else ""
                candidate = f"{tail}\n{piece}" if tail else piece
                # 带上重叠后若超长，就放弃重叠，保证不超限
                current = candidate if len(candidate) <= max_chars else piece
    if current:
        chunks.append(current)
    return chunks


# --------------------------------------------------------------------------- #
# 对外入口
# --------------------------------------------------------------------------- #
def chunk_document(
    path: str | Path,
    *,
    max_chars: int = 800,
    overlap: int = 100,
    metadata: dict | None = None,
) -> list[Chunk]:
    """把一份文档切成若干 Chunk。

    Args:
        path: 文档路径。
        max_chars: 单块最大字符数。
        overlap: 相邻块重叠字符数（仅用于硬切与跨块衔接）。
        metadata: 附加元数据（部门、密级、来源、日期等），每个块独立拷贝。
    """
    path = Path(path)
    text = read_text(path)
    extra = dict(metadata or {})

    if path.suffix.lower() in {".md", ".markdown"}:
        sections = _split_markdown_sections(text)
        title = next((s.split(" > ")[0] for s, _ in sections if s), path.stem)
        pairs: list[tuple[str, str]] = []
        for section, body in sections:
            for piece in _pack_paragraphs(_split_paragraphs(body), max_chars, overlap):
                pairs.append((section, piece))
    else:
        title = path.stem
        pairs = [("", p) for p in _pack_paragraphs(_split_paragraphs(text), max_chars, overlap)]

    return [
        Chunk(
            text=piece,
            source=path.name,
            title=title,
            section=section,
            metadata=dict(extra),
            index=i,
        )
        for i, (section, piece) in enumerate(pairs)
    ]
