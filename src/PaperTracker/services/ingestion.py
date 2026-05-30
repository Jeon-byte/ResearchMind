"""PDF ingestion service.

Downloads paper PDFs, extracts text, chunks content, and persists ingestion status.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from PaperTracker.llm.client import LLMApiClient
from PaperTracker.services.rag.chunker import chunk_page_texts
from PaperTracker.services.rag.service import CollectionRAGService
from PaperTracker.services.rag.vision_caption import VisionCaptionGenerator, merge_captions
from PaperTracker.storage.research import ParsedChunk, ResearchStore, ensure_paper_directory
from PaperTracker.utils.log import log

try:
    import fitz
except ImportError:  # pragma: no cover - import availability depends on environment
    fitz = None


@dataclass(slots=True)
class IngestionService:
    """Coordinates full-text ingestion for one paper."""

    store: ResearchStore
    papers_dir: Path
    rag_service: CollectionRAGService | None = None
    llm_client: LLMApiClient | None = None
    llm_model: str | None = None
    request_timeout: int = 60
    vision_captioner: VisionCaptionGenerator | None = None

    def ingest_paper(self, collection_id: int, source: str, source_id: str) -> None:
        """Run the PDF ingestion pipeline for one collection paper.

        Args:
            collection_id: Collection primary key.
            source: Source identifier.
            source_id: Source-level paper identifier.
        """
        record = self.store.get_latest_paper_record(source, source_id)
        pdf_url = record.get("pdf_url") if record else None
        self.store.update_collection_paper_status(collection_id, source, source_id, "downloading")
        self.store.upsert_asset_state(
            source,
            source_id,
            pdf_url=pdf_url,
            download_status="running",
            download_error=None,
            parse_status="queued",
            parse_error=None,
            index_status="queued",
            index_error=None,
        )
        if not pdf_url:
            message = "Missing PDF URL in stored paper metadata"
            self.store.update_collection_paper_status(collection_id, source, source_id, "failed")
            self.store.upsert_asset_state(
                source,
                source_id,
                download_status="failed",
                download_error=message,
                parse_status="failed",
                parse_error=message,
                index_status="failed",
                index_error=message,
            )
            return

        try:
            pdf_path = self._download_pdf(source, source_id, pdf_url)
            self.store.upsert_asset_state(
                source,
                source_id,
                pdf_url=pdf_url,
                local_path=str(pdf_path),
                download_status="completed",
                download_error=None,
                parse_status="running",
                parse_error=None,
                index_status="queued",
                index_error=None,
            )
            self.store.update_collection_paper_status(collection_id, source, source_id, "parsing")
            chunks = self._extract_chunks(pdf_path)
            if not chunks:
                raise RuntimeError("PDF parser returned no text chunks")
            chunks = self._prepend_summary_chunk(record or {}, chunks)
            chunks.extend(self._extract_figure_chunks(pdf_path, collection_id, source, source_id, start_index=len(chunks)))
            self.store.replace_paper_chunks(source, source_id, chunks)
            if self.rag_service is not None:
                self.rag_service.index_paper(
                    collection_id=collection_id,
                    source=source,
                    source_id=source_id,
                    chunk_rows=self.store.list_paper_chunks_for_rag(source, source_id),
                )
            self.store.upsert_asset_state(
                source,
                source_id,
                parse_status="completed",
                parse_error=None,
                index_status="completed",
                index_error=None,
            )
            self.store.update_collection_paper_status(collection_id, source, source_id, "indexed")
        except Exception as error:  # noqa: BLE001 - ingestion should surface status, not crash request
            message = str(error)
            log.warning("Paper ingestion failed: source=%s id=%s error=%s", source, source_id, message)
            self.store.update_collection_paper_status(collection_id, source, source_id, "failed")
            self.store.upsert_asset_state(
                source,
                source_id,
                download_status="failed" if "download" in message.lower() else None,
                download_error=message if "download" in message.lower() else None,
                parse_status="failed",
                parse_error=message,
                index_status="failed",
                index_error=message,
            )

    def _download_pdf(self, source: str, source_id: str, pdf_url: str) -> Path:
        """Download one paper PDF to local storage.

        Args:
            source: Source identifier.
            source_id: Source-level paper identifier.
            pdf_url: Remote PDF URL.

        Returns:
            Local PDF path.
        """
        source_dir = ensure_paper_directory(self.papers_dir, source)
        file_name = _safe_file_name(source_id)
        pdf_path = source_dir / f"{file_name}.pdf"
        response = requests.get(pdf_url, timeout=self.request_timeout)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if "pdf" not in content_type and not pdf_url.lower().endswith(".pdf"):
            log.warning("Downloading non-explicit PDF response: url=%s content_type=%s", pdf_url, content_type)
        pdf_path.write_bytes(response.content)
        return pdf_path

    def _extract_chunks(self, pdf_path: Path) -> list[ParsedChunk]:
        """Parse a PDF file into text chunks.

        Args:
            pdf_path: Local PDF file path.

        Returns:
            Parsed chunks with page metadata.
        """
        if fitz is None:
            raise RuntimeError("PyMuPDF is not installed")

        document = fitz.open(pdf_path)
        page_texts: list[tuple[int, str]] = []
        try:
            for page_index in range(document.page_count):
                text = document.load_page(page_index).get_text("text").strip()
                if text:
                    page_texts.append((page_index + 1, _clean_text(text)))
        finally:
            document.close()

        return chunk_page_texts(page_texts)

    def _extract_figure_chunks(
        self,
        pdf_path: Path,
        collection_id: int,
        source: str,
        source_id: str,
        *,
        start_index: int,
    ) -> list[ParsedChunk]:
        """Extract PDF image assets and caption-like text as figure chunks.

        This lightweight path gives ResearchMind figure-aware retrieval without
        requiring the heavier Docling/Qwen3-VL stack. Full visual embeddings can
        be layered onto the same chunks later.
        """
        if fitz is None or self.rag_service is None:
            return []

        figures_dir = self.rag_service.workspace(collection_id).collection_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        chunks: list[ParsedChunk] = []
        document = fitz.open(pdf_path)
        try:
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                page_no = page_index + 1
                captions = _caption_candidates(page.get_text("text"))
                images = page.get_images(full=True)
                for image_index, image_info in enumerate(images):
                    xref = image_info[0]
                    try:
                        extracted = document.extract_image(xref)
                    except Exception:
                        continue
                    image_bytes = extracted.get("image")
                    if not image_bytes:
                        continue
                    ext = (extracted.get("ext") or "png").lower()
                    figure_id = f"{_safe_file_name(source)}_{_safe_file_name(source_id)}_fig_{page_no}_{image_index}"
                    image_path = figures_dir / f"{figure_id}.{ext}"
                    image_path.write_bytes(image_bytes)
                    caption = captions[min(image_index, len(captions) - 1)] if captions else ""
                    generated_caption = self._generate_figure_caption(image_bytes, caption, len(chunks))
                    content = (
                        merge_captions(caption, generated_caption)
                        or f"Figure extracted from page {page_no} of {source}:{source_id}."
                    )
                    chunks.append(
                        ParsedChunk(
                            chunk_index=start_index + len(chunks),
                            content=content,
                            page_start=page_no,
                            page_end=page_no,
                            section_title=f"Figure {image_index + 1}",
                            token_count=len(content.split()),
                            modality="figure",
                            image_path=str(image_path),
                        )
                    )
        finally:
            document.close()
            if self.vision_captioner is not None:
                self.vision_captioner.release()
        return chunks

    def _generate_figure_caption(self, image_bytes: bytes, original_caption: str, figure_count: int) -> str:
        """Generate optional VLM caption text for one extracted figure."""
        if self.rag_service is None or self.rag_service.config is None:
            return ""
        config = self.rag_service.config
        if not config.multimodal_enabled or not config.vl_caption_enabled:
            return ""
        max_figures = config.vl_caption_max_figures_per_paper
        if max_figures != -1 and figure_count >= max_figures:
            return ""
        captioner = self._vision_captioner()
        if captioner is None:
            return ""
        return captioner.generate_from_bytes(image_bytes, original_caption)

    def _vision_captioner(self) -> VisionCaptionGenerator | None:
        """Return a shared VLM caption generator when configured."""
        if self.rag_service is None or self.rag_service.config is None:
            return None
        if self.vision_captioner is None:
            self.vision_captioner = VisionCaptionGenerator.from_config(self.rag_service.config)
        return self.vision_captioner

    def _prepend_summary_chunk(self, record: dict[str, Any], chunks: list[ParsedChunk]) -> list[ParsedChunk]:
        """Create a paper-level summary chunk and keep chunk indices stable."""
        summary = self._generate_paper_summary(record, chunks)
        if not summary:
            return chunks
        summary_chunk = ParsedChunk(
            chunk_index=0,
            content=summary,
            page_start=None,
            page_end=None,
            section_title="Paper Summary",
            token_count=len(summary.split()),
            modality="summary",
        )
        return [
            summary_chunk,
            *[
                ParsedChunk(
                    chunk_index=index,
                    content=chunk.content,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section_title=chunk.section_title,
                    token_count=chunk.token_count,
                    modality=chunk.modality,
                    image_path=chunk.image_path,
                )
                for index, chunk in enumerate(chunks, start=1)
            ],
        ]

    def _generate_paper_summary(self, record: dict[str, Any], chunks: list[ParsedChunk]) -> str:
        """Generate one retrieval-oriented paper summary."""
        title = str(record.get("title") or "").strip()
        abstract = str(record.get("abstract") or "").strip()
        evidence = _select_summary_evidence(chunks)
        if self.llm_client is not None and self.llm_model:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You create compact retrieval-oriented summaries for academic papers. "
                        "Use only the provided abstract and excerpts. "
                        "Write in Simplified Chinese. "
                        "If a field is not supported by evidence, write '未在证据中明确说明'."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Title:\n{title}\n\n"
                        f"Abstract:\n{abstract}\n\n"
                        f"Excerpts:\n{evidence}\n\n"
                        "请生成一个 paper_summary chunk，严格包含以下小节：\n"
                        "1. 主要贡献\n"
                        "2. 方法模块\n"
                        "3. 实验数据集与设置\n"
                        "4. 主要实验结论\n"
                        "5. 适合回答的问题\n"
                        "要求简洁，但保留关键术语、数据集名、指标名、模块名。"
                    ),
                },
            ]
            try:
                answer = self.llm_client.chat_completion(
                    messages=messages,
                    model=self.llm_model,
                    temperature=0.0,
                    max_tokens=900,
                ).strip()
                if answer:
                    return f"Paper Summary\nTitle: {title}\n\n{answer}"
            except Exception as error:  # noqa: BLE001 - fallback keeps ingestion usable
                log.warning("Paper summary generation failed; using extractive fallback: %s", error)
        return _build_extractive_paper_summary(title, abstract, evidence)


def _clean_text(text: str) -> str:
    """Normalize extracted PDF text for chunking."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _caption_candidates(page_text: str) -> list[str]:
    """Return caption-like lines from one PDF page."""
    lines = [" ".join(line.split()) for line in page_text.splitlines()]
    lines = [line for line in lines if line]
    captions = []
    pattern = re.compile(r"^(fig\.?|figure|table)\s*\d*[:.\s-]", re.IGNORECASE)
    for index, line in enumerate(lines):
        if not pattern.search(line):
            continue
        tail = lines[index + 1] if index + 1 < len(lines) and len(line) < 80 else ""
        caption = f"{line} {tail}".strip()
        captions.append(caption[:900])
    return captions


def _select_summary_evidence(chunks: list[ParsedChunk], *, limit: int = 9000) -> str:
    """Pick broad paper-level evidence without sending the full PDF to the LLM."""
    scored = []
    for chunk in chunks:
        if chunk.modality != "text":
            continue
        text = " ".join(chunk.content.split())
        lower = text.lower()
        score = 0
        if chunk.page_start and chunk.page_start <= 2:
            score += 3
        if any(term in lower for term in ("abstract", "introduction", "contribution", "we propose", "we introduce")):
            score += 3
        if any(term in lower for term in ("method", "framework", "pipeline", "architecture", "module", "approach")):
            score += 4
        if any(term in lower for term in ("experiment", "evaluation", "dataset", "benchmark", "result", "comparison", "ablation", "metric", "table")):
            score += 4
        if any(term in lower for term in ("conclusion", "in this paper", "we present")):
            score += 2
        scored.append((score, chunk.page_start or 9999, text))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = []
    total = 0
    for score, page, text in scored:
        if score <= 0 and selected:
            continue
        prefix = f"[page {page}] "
        fragment = prefix + text[:1800]
        if total + len(fragment) > limit:
            break
        selected.append(fragment)
        total += len(fragment)
    return "\n\n".join(selected)


def _build_extractive_paper_summary(title: str, abstract: str, evidence: str) -> str:
    """Deterministic summary chunk when LLM generation is unavailable."""
    lines = [
        "Paper Summary",
        f"Title: {title or 'Untitled'}",
        "",
        "1. 主要贡献",
        abstract[:1200] if abstract else "未在证据中明确说明",
        "",
        "2. 方法模块",
        _extract_lines_for_summary(evidence, ("method", "framework", "pipeline", "module", "approach", "we propose", "we introduce")),
        "",
        "3. 实验数据集与设置",
        _extract_lines_for_summary(evidence, ("experiment", "evaluation", "dataset", "benchmark", "metric", "table", "comparison")),
        "",
        "4. 主要实验结论",
        _extract_lines_for_summary(evidence, ("result", "outperform", "improve", "ablation", "comparison", "achieve")),
        "",
        "5. 适合回答的问题",
        "该 summary chunk 适合回答论文贡献、方法模块、实验设置、数据集、实验结论和全文概览类问题。",
    ]
    return "\n".join(lines)


def _extract_lines_for_summary(text: str, keywords: tuple[str, ...], *, limit: int = 900) -> str:
    sentences = re.split(r"(?<=[.!?。！？])\s+", " ".join(text.split()))
    picked = [sentence for sentence in sentences if any(keyword in sentence.lower() for keyword in keywords)]
    if not picked:
        return "未在证据中明确说明"
    result = " ".join(picked[:4]).strip()
    return result[:limit] if result else "未在证据中明确说明"


def _safe_file_name(source_id: str) -> str:
    """Convert source id into a filesystem-safe name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", source_id).strip("_") or "paper"
