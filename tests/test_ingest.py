"""ISSUE-002 文档解析与切分的测试。"""
from __future__ import annotations

import pytest

from knowledgerag.ingest import chunk_document

SAMPLE_MD = """# 考勤制度

## 上班时间

上午 9:00 上班，下午 6:00 下班。

## 请假

请假需提前一天在系统提交申请。

### 事假

事假按天扣除相应工资。
"""


def test_markdown_sections_and_title(tmp_path):
    doc = tmp_path / "考勤.md"
    doc.write_text(SAMPLE_MD, encoding="utf-8")

    chunks = chunk_document(doc)

    assert chunks, "应至少产出一个块"
    assert chunks[0].title == "考勤制度"
    sections = {c.section for c in chunks}
    assert any("上班时间" in s for s in sections)
    assert any("事假" in s for s in sections)
    # 标题路径应包含父级
    assert any("考勤制度 > 请假 > 事假" == s for s in sections)


def test_metadata_is_preserved_and_isolated(tmp_path):
    doc = tmp_path / "制度.md"
    doc.write_text("# 第一章\n\n内容内容。\n", encoding="utf-8")

    chunks = chunk_document(doc, metadata={"dept": "HR", "classification": "internal"})

    assert chunks[0].metadata["dept"] == "HR"
    assert chunks[0].metadata["classification"] == "internal"

    # 每个块的 metadata 是独立副本
    chunks[0].metadata["dept"] = "ADMIN"
    assert chunks[0].metadata["dept"] != "HR"


def test_max_chars_is_respected(tmp_path):
    doc = tmp_path / "long.md"
    doc.write_text("# 长文\n\n" + ("这是一句话。" * 300), encoding="utf-8")

    chunks = chunk_document(doc, max_chars=100, overlap=10)

    assert chunks
    assert all(c.chars <= 100 for c in chunks)


def test_plain_text_single_chunk(tmp_path):
    doc = tmp_path / "notes.txt"
    doc.write_text("第一段。\n\n第二段。\n", encoding="utf-8")

    chunks = chunk_document(doc, max_chars=200)

    assert len(chunks) == 1
    assert "第一段" in chunks[0].text
    assert "第二段" in chunks[0].text


def test_empty_file_yields_no_chunks(tmp_path):
    doc = tmp_path / "empty.md"
    doc.write_text("", encoding="utf-8")

    assert chunk_document(doc) == []


def test_unsupported_suffix_raises(tmp_path):
    doc = tmp_path / "a.xyz"
    doc.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError):
        chunk_document(doc)
