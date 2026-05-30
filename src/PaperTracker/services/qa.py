"""Collection question answering service.

Retrieves evidence chunks from indexed papers and builds citation-backed answers.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from PaperTracker.llm.client import LLMApiClient
from PaperTracker.services.rag.service import CollectionRAGService
from PaperTracker.storage.research import ResearchStore


@dataclass(frozen=True, slots=True)
class QAResult:
    """Question answering result payload."""

    conversation_id: int
    answer_message_id: int
    answer: str
    citations: list[dict]
    retrieval_debug: dict


@dataclass(slots=True)
class CollectionQAService:
    """Runs retrieval and answer generation for one collection."""

    store: ResearchStore
    llm_client: LLMApiClient | None = None
    llm_model: str | None = None
    rag_service: CollectionRAGService | None = None
    max_chunks: int = 4

    def ask(
        self,
        collection_id: int,
        question: str,
        *,
        conversation_id: int | None = None,
        mode: str = "standard",
    ) -> QAResult:
        """Answer one question against a collection knowledge base.

        Args:
            collection_id: Collection primary key.
            question: User question.
            conversation_id: Optional existing conversation id.

        Returns:
            Structured QA result with citations.
        """
        if conversation_id is None:
            conversation_id = self.store.create_conversation(collection_id, question[:80])

        user_message_id = self.store.add_message(conversation_id, "user", question.strip())
        del user_message_id

        rag_hits = []
        if self.rag_service is not None:
            rag_hits = self.rag_service.retrieve(collection_id, question, top_k=self.max_chunks, mode=mode)
        if rag_hits:
            answer = self.rag_service.generate_answer(
                question=question,
                hits=rag_hits,
                llm_client=self.llm_client,
                llm_model=self.llm_model,
            )
            answer_message_id = self.store.add_message(conversation_id, "assistant", answer)
            citations = [
                {
                    "chunk_id": hit.chunk.db_chunk_id,
                    "paper_title": hit.chunk.paper_title,
                    "page_start": hit.chunk.page_start,
                    "page_end": hit.chunk.page_end,
                    "section_title": hit.chunk.section_title,
                    "quote_text": _quote_text(hit.chunk.content),
                    "score": hit.score,
                    "modality": hit.chunk.modality,
                    "image_path": hit.chunk.image_path,
                    "image_url": _figure_image_url(collection_id, hit.chunk.image_path),
                }
                for hit in rag_hits
            ]
            self.store.add_citations(answer_message_id, citations)
            return QAResult(
                conversation_id=conversation_id,
                answer_message_id=answer_message_id,
                answer=answer,
                citations=citations,
                retrieval_debug={
                    "mode": mode,
                    "source": "rag",
                    "hits": [hit.to_dict() for hit in rag_hits],
                    "status": self.rag_service.debug_status(),
                },
            )

        chunks = self.store.search_collection_chunks(collection_id, question, limit=self.max_chunks)
        if not chunks:
            answer = (
                "I could not find indexed full-text evidence in this collection yet. "
                "Please make sure the papers were downloaded and parsed successfully."
            )
            answer_message_id = self.store.add_message(conversation_id, "assistant", answer)
            return QAResult(
                conversation_id=conversation_id,
                answer_message_id=answer_message_id,
                answer=answer,
                citations=[],
                retrieval_debug={"mode": mode, "source": "none", "hits": []},
            )

        answer = self._generate_answer(question, chunks)
        answer_message_id = self.store.add_message(conversation_id, "assistant", answer)
        citations = [
            {
                "chunk_id": chunk["chunk_id"],
                "paper_title": chunk["paper_title"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "section_title": chunk["section_title"],
                "quote_text": _quote_text(chunk["content"]),
                "score": chunk["score"],
                "modality": chunk.get("modality", "text"),
                "image_path": chunk.get("image_path"),
                "image_url": _figure_image_url(collection_id, chunk.get("image_path")),
            }
            for chunk in chunks
        ]
        self.store.add_citations(answer_message_id, citations)
        return QAResult(
            conversation_id=conversation_id,
            answer_message_id=answer_message_id,
            answer=answer,
            citations=citations,
            retrieval_debug={"mode": mode, "source": "fts", "hits": chunks},
        )

    def ask_stream(
        self,
        collection_id: int,
        question: str,
        *,
        conversation_id: int | None = None,
        mode: str = "standard",
    ) -> Iterator[dict]:
        """Stream one answer while preserving the same DB side effects as ask()."""
        if conversation_id is None:
            conversation_id = self.store.create_conversation(collection_id, question[:80])

        user_message_id = self.store.add_message(conversation_id, "user", question.strip())
        del user_message_id
        yield {"type": "meta", "conversation_id": conversation_id}

        rag_hits = []
        if self.rag_service is not None:
            rag_hits = self.rag_service.retrieve(collection_id, question, top_k=self.max_chunks, mode=mode)
        if rag_hits:
            citations = [
                {
                    "chunk_id": hit.chunk.db_chunk_id,
                    "paper_title": hit.chunk.paper_title,
                    "page_start": hit.chunk.page_start,
                    "page_end": hit.chunk.page_end,
                    "section_title": hit.chunk.section_title,
                    "quote_text": _quote_text(hit.chunk.content),
                    "score": hit.score,
                    "modality": hit.chunk.modality,
                    "image_path": hit.chunk.image_path,
                    "image_url": _figure_image_url(collection_id, hit.chunk.image_path),
                }
                for hit in rag_hits
            ]
            answer_parts = []
            for token in self.rag_service.generate_answer_stream(
                question=question,
                hits=rag_hits,
                llm_client=self.llm_client,
                llm_model=self.llm_model,
            ):
                answer_parts.append(token)
                yield {"type": "token", "content": token}
            answer = "".join(answer_parts).strip()
            answer_message_id = self.store.add_message(conversation_id, "assistant", answer)
            self.store.add_citations(answer_message_id, citations)
            yield {
                "type": "done",
                "message_id": answer_message_id,
                "answer": answer,
                "citations": citations,
                "retrieval_debug": {
                    "mode": mode,
                    "source": "rag",
                    "hits": [hit.to_dict() for hit in rag_hits],
                    "status": self.rag_service.debug_status(),
                },
            }
            return

        chunks = self.store.search_collection_chunks(collection_id, question, limit=self.max_chunks)
        if not chunks:
            answer = (
                "I could not find indexed full-text evidence in this collection yet. "
                "Please make sure the papers were downloaded and parsed successfully."
            )
            yield {"type": "token", "content": answer}
            answer_message_id = self.store.add_message(conversation_id, "assistant", answer)
            yield {
                "type": "done",
                "message_id": answer_message_id,
                "answer": answer,
                "citations": [],
                "retrieval_debug": {"mode": mode, "source": "none", "hits": []},
            }
            return

        answer = self._generate_answer(question, chunks)
        yield {"type": "token", "content": answer}
        answer_message_id = self.store.add_message(conversation_id, "assistant", answer)
        citations = [
            {
                "chunk_id": chunk["chunk_id"],
                "paper_title": chunk["paper_title"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "section_title": chunk["section_title"],
                "quote_text": _quote_text(chunk["content"]),
                "score": chunk["score"],
                "modality": chunk.get("modality", "text"),
                "image_path": chunk.get("image_path"),
                "image_url": _figure_image_url(collection_id, chunk.get("image_path")),
            }
            for chunk in chunks
        ]
        self.store.add_citations(answer_message_id, citations)
        yield {
            "type": "done",
            "message_id": answer_message_id,
            "answer": answer,
            "citations": citations,
            "retrieval_debug": {"mode": mode, "source": "fts", "hits": chunks},
        }

    def _generate_answer(self, question: str, chunks: list[dict]) -> str:
        """Generate answer text from retrieved evidence."""
        if self.llm_client is None or not self.llm_model:
            return _build_extractive_answer(question, chunks)

        evidence_lines = []
        for idx, chunk in enumerate(chunks, start=1):
            evidence_lines.append(
                f"[{idx}] Title: {chunk['paper_title']}\n"
                f"Pages: {chunk['page_start']}-{chunk['page_end']}\n"
                f"Content: {chunk['content']}"
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise research assistant. "
                    "Answer only from the provided evidence. "
                    "If the evidence is insufficient, say so explicitly. "
                    "Cite sources inline using bracket numbers like [1]."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Evidence:\n\n{'\n\n'.join(evidence_lines)}"
                ),
            },
        ]
        return self.llm_client.chat_completion(
            messages=messages,
            model=self.llm_model,
            temperature=0.1,
            max_tokens=900,
        ).strip()


def _build_extractive_answer(question: str, chunks: list[dict]) -> str:
    """Build a deterministic fallback answer from evidence chunks."""
    del question
    lines = [
        "LLM answering is not configured, so this response is a citation-backed evidence digest:",
    ]
    for idx, chunk in enumerate(chunks, start=1):
        page_label = "unknown page"
        if chunk["page_start"] and chunk["page_end"]:
            page_label = f"pages {chunk['page_start']}-{chunk['page_end']}"
        lines.append(
            f"[{idx}] {chunk['paper_title']} ({page_label}): {_quote_text(chunk['content'])}"
        )
    return "\n\n".join(lines)


def _quote_text(text: str, limit: int = 360) -> str:
    """Return a compact evidence quote for UI display."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _figure_image_url(collection_id: int, image_path: str | None) -> str | None:
    if not image_path:
        return None
    from pathlib import Path

    return f"/api/collections/{collection_id}/figures/{Path(image_path).name}"
