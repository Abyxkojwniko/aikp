# -*- coding: utf-8 -*-
"""AIKP Document Chunker — Split long TRPG module texts into LLM-friendly chunks."""

from __future__ import annotations

import re
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class Chunk:
    index: int
    title: str                    # e.g. "事件① 诀别" or "Chunk 3"
    text: str                     # Full chunk text
    start_line: int               # Line number in source
    end_line: int                 # Line number in source
    char_count: int
    section_type: str = ""        # "event" | "dialogue" | "rule" | "prose" | "metadata"


# ── Structure Detection Patterns ──────────────────────────────

SECTION_PATTERNS = [
    (re.compile(r"^【?(事件|Event)\s*[①②③④⑤⑥⑦⑧⑨⑩⑪⑫\d]+"), "event"),
    (re.compile(r"^■+(真相|登場人物|开始|对话|简介|规则|参考文献|あらすじ)"), "metadata"),
    (re.compile(r"^★+\s*"), "rule"),
    (re.compile(r"^◆+"), "dialogue"),
    (re.compile(r"^(END[：:]|高潮阶段|Climax)"), "climax"),
    (re.compile(r"^第[一二三四五六七八九十\d]+章"), "chapter"),
    (re.compile(r"^#+\s+"), "header"),
]


def detect_structure(lines: list[str]) -> list[dict]:
    """Detect document structure: section boundaries and types."""
    sections = []
    for i, line in enumerate(lines):
        for pattern, stype in SECTION_PATTERNS:
            if pattern.match(line.strip()):
                sections.append({"line": i, "title": line.strip(), "type": stype})
                break
    return sections


# ── Chunking Strategies ───────────────────────────────────────

def chunk_by_structure(lines: list[str], target_size: int = 6000, overlap: int = 300) -> list[Chunk]:
    """Chunk using detected section boundaries. Preferred strategy."""
    sections = detect_structure(lines)
    if len(sections) < 2:
        return []  # Not enough structure, fall back

    chunks = []
    for idx in range(len(sections)):
        sec = sections[idx]
        start = sec["line"]

        # End at next section boundary or end of doc
        if idx + 1 < len(sections):
            end = sections[idx + 1]["line"]
        else:
            end = len(lines)

        # Collect lines for this section
        section_lines = lines[start:end]
        raw_text = "\n".join(section_lines).strip()
        if not raw_text:
            continue

        # If section is too large, sub-chunk it
        if len(raw_text) > target_size * 1.5:
            sub_chunks = _split_long_section(
                section_lines, start, target_size, overlap,
                title=sec["title"], stype=sec["type"]
            )
            chunks.extend(sub_chunks)
        else:
            chunks.append(Chunk(
                index=len(chunks),
                title=sec["title"],
                text=raw_text,
                start_line=start,
                end_line=end,
                char_count=len(raw_text),
                section_type=sec["type"],
            ))
    return chunks


def _split_long_section(
    lines: list[str], base_line: int, target_size: int, overlap: int,
    title: str, stype: str
) -> list[Chunk]:
    """Split a section that's too large into sub-chunks."""
    chunks = []
    current = []
    current_len = 0
    sub_idx = 0

    for i, line in enumerate(lines):
        line_len = len(line) + 1  # +1 for newline
        if current_len + line_len > target_size and current:
            raw_text = "\n".join(current).strip()
            chunks.append(Chunk(
                index=len(chunks),
                title=f"{title} (part {sub_idx + 1})",
                text=raw_text,
                start_line=base_line + i - len(current),
                end_line=base_line + i,
                char_count=len(raw_text),
                section_type=stype,
            ))
            # Keep overlap
            overlap_lines = max(1, int(len(current) * overlap / target_size))
            current = current[-overlap_lines:] if overlap_lines < len(current) else []
            current_len = sum(len(l) + 1 for l in current)
            sub_idx += 1

        current.append(line)
        current_len += line_len

    # Final chunk
    if current:
        raw_text = "\n".join(current).strip()
        chunks.append(Chunk(
            index=len(chunks),
            title=f"{title} (part {sub_idx + 1})" if sub_idx > 0 else title,
            text=raw_text,
            start_line=base_line + len(lines) - len(current),
            end_line=base_line + len(lines),
            char_count=len(raw_text),
            section_type=stype,
        ))

    return chunks


def chunk_by_paragraph(lines: list[str], target_size: int = 6000, overlap: int = 300) -> list[Chunk]:
    """Fallback: chunk by paragraphs when no clear structure detected.
    Handles very long paragraphs by splitting line-by-line."""
    chunks = []
    chunk_idx = 0
    current_lines = []
    current_len = 0
    line_num = 0

    for line in lines:
        line_len = len(line) + 1

        # Split very long lines character-by-character
        if line_len > target_size * 2:
            sub_chunks = _split_long_line(line, line_num, chunk_idx, target_size, overlap)
            chunks.extend(sub_chunks)
            chunk_idx += len(sub_chunks)
            current_lines = []
            current_len = 0
            line_num += 1
            continue

        if current_len + line_len > target_size and current_lines:
            text = "\n".join(current_lines)
            chunks.append(Chunk(
                index=chunk_idx,
                title=f"Chunk {chunk_idx + 1}",
                text=text,
                start_line=line_num - len(current_lines),
                end_line=line_num,
                char_count=len(text),
                section_type="prose",
            ))
            chunk_idx += 1

            # Keep overlap
            overlap_chars = 0
            overlap_lines = []
            for ol in reversed(current_lines):
                if overlap_chars + len(ol) > overlap:
                    break
                overlap_lines.insert(0, ol)
                overlap_chars += len(ol) + 1
            current_lines = overlap_lines
            current_len = overlap_chars

        current_lines.append(line)
        current_len += line_len
        line_num += 1

    if current_lines:
        text = "\n".join(current_lines)
        chunks.append(Chunk(
            index=chunk_idx,
            title=f"Chunk {chunk_idx + 1}",
            text=text,
            start_line=line_num - len(current_lines),
            end_line=line_num,
            char_count=len(text),
            section_type="prose",
        ))

    return chunks


def _split_long_line(line: str, line_num: int, base_chunk_idx: int, target_size: int, overlap: int) -> list[Chunk]:
    """Split a single very long line into character-based chunks."""
    chunks = []
    for i in range(0, len(line), target_size - overlap):
        sub = line[i:i + target_size]
        chunks.append(Chunk(
            index=base_chunk_idx + len(chunks),
            title=f"Chunk {base_chunk_idx + len(chunks) + 1}",
            text=sub,
            start_line=line_num,
            end_line=line_num,
            char_count=len(sub),
            section_type="prose",
        ))
    return chunks


# ── Main API ──────────────────────────────────────────────────

def chunk_document(text: str, target_size: int = 6000, overlap: int = 300) -> list[Chunk]:
    """Split document text into chunks for LLM processing.

    Args:
        text: Full document text.
        target_size: Target characters per chunk (~1.5K tokens for Chinese).
        overlap: Character overlap between chunks.

    Returns:
        List of Chunk objects.
    """
    lines = text.split("\n")

    # Try structure-based chunking first
    chunks = chunk_by_structure(lines, target_size, overlap)
    if chunks:
        return chunks

    # Fall back to paragraph-based chunking
    return chunk_by_paragraph(lines, target_size, overlap)


def chunks_to_dicts(chunks: list[Chunk]) -> list[dict]:
    """Convert chunks to dicts for JSON serialization."""
    return [
        {
            "index": c.index,
            "title": c.title,
            "text": c.text,
            "start_line": c.start_line,
            "end_line": c.end_line,
            "char_count": c.char_count,
            "section_type": c.section_type,
        }
        for c in chunks
    ]


# ── Test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python chunker.py <file.txt>")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Input: {len(text)} chars, {text.count(chr(10))} lines")
    print(f"Structure detected: {len(detect_structure(text.split(chr(10))))} sections")
    print()

    chunks = chunk_document(text)
    print(f"Chunks: {len(chunks)}")
    for c in chunks:
        print(f"  [{c.section_type}] {c.title}: {c.char_count} chars (L{c.start_line}-{c.end_line})")
