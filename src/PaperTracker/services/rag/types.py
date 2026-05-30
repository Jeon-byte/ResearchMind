"""RAG data types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RAGChunk:
    """One chunk stored in a Knowledge Base RAG workspace."""

    vector_id: int
    chunk_id: str
    db_chunk_id: int
    collection_id: int
    source: str
    source_id: str
    paper_title: str
    content: str
    page_start: int | None = None
    page_end: int | None = None
    section_title: str | None = None
    token_count: int = 0
    modality: str = "text"
    image_path: str | None = None

    def to_dict(self) -> dict:
        """Serialize chunk metadata to JSON."""
        return {
            "vector_id": self.vector_id,
            "chunk_id": self.chunk_id,
            "db_chunk_id": self.db_chunk_id,
            "collection_id": self.collection_id,
            "source": self.source,
            "source_id": self.source_id,
            "paper_title": self.paper_title,
            "content": self.content,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section_title": self.section_title,
            "token_count": self.token_count,
            "modality": self.modality,
            "image_path": self.image_path,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "RAGChunk":
        """Load chunk metadata from JSON."""
        return cls(
            vector_id=int(payload["vector_id"]),
            chunk_id=str(payload["chunk_id"]),
            db_chunk_id=int(payload["db_chunk_id"]),
            collection_id=int(payload["collection_id"]),
            source=str(payload["source"]),
            source_id=str(payload["source_id"]),
            paper_title=str(payload.get("paper_title") or payload["source_id"]),
            content=str(payload.get("content") or ""),
            page_start=payload.get("page_start"),
            page_end=payload.get("page_end"),
            section_title=payload.get("section_title"),
            token_count=int(payload.get("token_count") or 0),
            modality=str(payload.get("modality") or "text"),
            image_path=payload.get("image_path"),
        )


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    """Retrieved evidence with score and source metadata."""

    chunk: RAGChunk
    score: float
    rank: int
    retrieval_path: str = "vector"
    vector_score: float | None = None
    fts_score: float | None = None
    image_score: float | None = None
    rerank_score: float | None = None

    def to_dict(self) -> dict:
        """Serialize hit for answer generation and API output."""
        return {
            **self.chunk.to_dict(),
            "score": self.score,
            "rank": self.rank,
            "retrieval_path": self.retrieval_path,
            "vector_score": self.vector_score,
            "fts_score": self.fts_score,
            "image_score": self.image_score,
            "rerank_score": self.rerank_score,
        }
