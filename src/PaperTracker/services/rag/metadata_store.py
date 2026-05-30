"""JSON chunk metadata store for Knowledge Base RAG workspaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from PaperTracker.services.rag.types import RAGChunk


class RAGMetadataStore:
    """Stores chunk metadata in an `all_chunks.json` snapshot."""

    def __init__(self, chunks: Iterable[RAGChunk] = ()) -> None:
        self._chunks = list(chunks)

    def __len__(self) -> int:
        return len(self._chunks)

    def all_chunks(self) -> list[RAGChunk]:
        """Return all chunks ordered by vector id."""
        return sorted(self._chunks, key=lambda chunk: chunk.vector_id)

    def by_vector_id(self) -> dict[int, RAGChunk]:
        """Return vector-id lookup."""
        return {chunk.vector_id: chunk for chunk in self._chunks}

    def paper_chunks(self, source: str, source_id: str) -> list[RAGChunk]:
        """Return chunks for one paper."""
        return [
            chunk
            for chunk in self.all_chunks()
            if chunk.source == source and chunk.source_id == source_id
        ]

    def append_paper_chunks(
        self,
        *,
        collection_id: int,
        source: str,
        source_id: str,
        rows: list[dict],
    ) -> list[RAGChunk]:
        """Append chunks for a paper that is not yet present in this workspace."""
        if self.paper_chunks(source, source_id):
            raise ValueError("Paper already exists in RAG metadata; use replace_paper_chunks")
        start = max((chunk.vector_id for chunk in self._chunks), default=-1) + 1
        added = []
        for offset, row in enumerate(rows):
            chunk_index = int(row["chunk_index"])
            added.append(
                RAGChunk(
                    vector_id=start + offset,
                    chunk_id=f"{source}:{source_id}:chunk:{chunk_index}",
                    db_chunk_id=int(row["id"]),
                    collection_id=collection_id,
                    source=source,
                    source_id=source_id,
                    paper_title=str(row.get("paper_title") or source_id),
                    content=str(row.get("content") or ""),
                    page_start=row.get("page_start"),
                    page_end=row.get("page_end"),
                    section_title=row.get("section_title"),
                    token_count=int(row.get("token_count") or 0),
                    modality=str(row.get("modality") or "text"),
                    image_path=row.get("image_path"),
                )
            )
        self._chunks.extend(added)
        return added

    def replace_paper_chunks(
        self,
        *,
        collection_id: int,
        source: str,
        source_id: str,
        rows: list[dict],
    ) -> None:
        """Replace one paper's chunks and compact vector ids."""
        kept = [
            chunk
            for chunk in self._chunks
            if not (chunk.source == source and chunk.source_id == source_id)
        ]
        start = len(kept)
        added = []
        for offset, row in enumerate(rows):
            vector_id = start + offset
            chunk_index = int(row["chunk_index"])
            added.append(
                RAGChunk(
                    vector_id=vector_id,
                    chunk_id=f"{source}:{source_id}:chunk:{chunk_index}",
                    db_chunk_id=int(row["id"]),
                    collection_id=collection_id,
                    source=source,
                    source_id=source_id,
                    paper_title=str(row.get("paper_title") or source_id),
                    content=str(row.get("content") or ""),
                    page_start=row.get("page_start"),
                    page_end=row.get("page_end"),
                    section_title=row.get("section_title"),
                    token_count=int(row.get("token_count") or 0),
                    modality=str(row.get("modality") or "text"),
                    image_path=row.get("image_path"),
                )
            )

        compacted = []
        for vector_id, chunk in enumerate([*kept, *added]):
            compacted.append(
                RAGChunk(
                    vector_id=vector_id,
                    chunk_id=chunk.chunk_id,
                    db_chunk_id=chunk.db_chunk_id,
                    collection_id=chunk.collection_id,
                    source=chunk.source,
                    source_id=chunk.source_id,
                    paper_title=chunk.paper_title,
                    content=chunk.content,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section_title=chunk.section_title,
                    token_count=chunk.token_count,
                    modality=chunk.modality,
                    image_path=chunk.image_path,
                )
            )
        self._chunks = compacted

    def save(self, path: Path) -> None:
        """Save metadata snapshot."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"chunks": [chunk.to_dict() for chunk in self.all_chunks()]}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RAGMetadataStore":
        """Load metadata snapshot. Missing files return an empty store."""
        if not path.exists():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(RAGChunk.from_dict(item) for item in payload.get("chunks", []))
