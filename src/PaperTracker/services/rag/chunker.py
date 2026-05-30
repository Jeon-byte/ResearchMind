"""Section-aware chunking adapted from MultiRAG-Doc."""

from __future__ import annotations

import bisect
import re
from typing import Sequence

from PaperTracker.storage.research import ParsedChunk


def chunk_page_texts(
    page_texts: Sequence[tuple[int, str]],
    *,
    chunk_size: int = 2000,
    overlap_sentences: int = 2,
    min_chunk_size: int = 200,
) -> list[ParsedChunk]:
    """Build section-aware chunks from page-level text."""
    full_text, page_offsets = _concat_pages(page_texts)
    section_map = _extract_section_map(full_text)
    chunks = []
    for idx, item in enumerate(_section_aware_chunk(full_text, chunk_size, overlap_sentences, min_chunk_size)):
        start_page = _lookup_page(page_offsets, item["start"])
        end_page = _lookup_page(page_offsets, max(item["start"], item["end"] - 1))
        chunks.append(
            ParsedChunk(
                chunk_index=idx,
                content=item["text"].strip(),
                page_start=start_page,
                page_end=end_page,
                section_title=_lookup_section(section_map, item["start"]) or None,
                token_count=len(item["text"].split()),
            )
        )
    return chunks


def _concat_pages(page_texts: Sequence[tuple[int, str]]) -> tuple[str, list[tuple[int, int]]]:
    parts = []
    page_offsets = []
    offset = 0
    for page_no, text in page_texts:
        if not text:
            continue
        page_offsets.append((offset, int(page_no)))
        parts.append(text)
        offset += len(text) + 2
    return "\n\n".join(parts), page_offsets


def _section_aware_chunk(
    text: str,
    chunk_size: int,
    overlap_sentences: int,
    min_chunk_size: int,
) -> list[dict]:
    sections = _merge_short_sections(_parse_sections(text), text, min_chunk_size)
    chunks = []
    for section in sections:
        body = text[section["body_start"]:section["body_end"]]
        if not body.strip():
            continue
        if len(body) <= chunk_size:
            chunks.append({"text": body, "start": section["body_start"], "end": section["body_end"]})
        else:
            for sub in _sentence_boundary_chunk(body, chunk_size, overlap_sentences):
                chunks.append(
                    {
                        "text": sub["text"],
                        "start": section["body_start"] + sub["start"],
                        "end": section["body_start"] + sub["end"],
                    }
                )
    return chunks


def _sentence_boundary_chunk(text: str, chunk_size: int, overlap_sentences: int) -> list[dict]:
    sentences = _split_sentences(text)
    chunks = []
    current = []
    current_len = 0
    for sentence, start, end in sentences:
        if len(sentence) > chunk_size:
            if current:
                chunks.append({"text": "".join(part[0] for part in current), "start": current[0][1], "end": current[-1][2]})
                current = []
                current_len = 0
            for sub_start in range(0, len(sentence), chunk_size):
                sub_end = min(sub_start + chunk_size, len(sentence))
                chunks.append({"text": sentence[sub_start:sub_end], "start": start + sub_start, "end": start + sub_end})
            continue
        if current and current_len + len(sentence) > chunk_size:
            chunks.append({"text": "".join(part[0] for part in current), "start": current[0][1], "end": current[-1][2]})
            current = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current_len = sum(len(part[0]) for part in current)
        current.append((sentence, start, end))
        current_len += len(sentence)
    if current:
        chunks.append({"text": "".join(part[0] for part in current), "start": current[0][1], "end": current[-1][2]})
    return chunks


def _parse_sections(text: str) -> list[dict]:
    matches = list(re.finditer(r"^#{1,3}\s+.+$", text, re.MULTILINE))
    sections = []
    first_header = matches[0].start() if matches else len(text)
    if text[:first_header].strip():
        sections.append({"header": "", "body_start": 0, "body_end": first_header})
    for idx, match in enumerate(matches):
        body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        sections.append({"header": match.group(0).strip(), "body_start": match.end(), "body_end": body_end})
    return sections or [{"header": "", "body_start": 0, "body_end": len(text)}]


def _merge_short_sections(sections: list[dict], text: str, min_chunk_size: int) -> list[dict]:
    if not sections:
        return []
    result = [dict(sections[0])]
    for section in sections[1:]:
        body = text[section["body_start"]:section["body_end"]]
        if len(body.strip()) < min_chunk_size:
            result[-1] = dict(result[-1])
            result[-1]["body_end"] = section["body_end"]
        else:
            result.append(dict(section))
    return result


def _extract_section_map(text: str) -> list[tuple[int, str]]:
    return [(match.start(), match.group(1).strip()) for match in re.finditer(r"^#{1,3}\s+(.+)$", text, re.MULTILINE)]


def _lookup_section(section_map: list[tuple[int, str]], offset: int) -> str:
    if not section_map:
        return ""
    offsets = [item[0] for item in section_map]
    idx = bisect.bisect_right(offsets, offset) - 1
    return section_map[idx][1] if idx >= 0 else ""


def _lookup_page(page_offsets: list[tuple[int, int]], offset: int) -> int:
    if not page_offsets:
        return 1
    offsets = [item[0] for item in page_offsets]
    idx = bisect.bisect_right(offsets, offset) - 1
    return page_offsets[max(idx, 0)][1]


def _split_sentences(text: str) -> list[tuple[str, int, int]]:
    result = []
    prev = 0
    for match in re.finditer(r"[.!?]+\s+|\n{2,}", text):
        end = match.end()
        fragment = text[prev:end]
        if fragment.strip():
            result.append((fragment, prev, end))
        prev = end
    tail = text[prev:]
    if tail.strip():
        result.append((tail, prev, len(text)))
    return result
