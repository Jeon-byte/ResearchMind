"""Knowledge Base RAG workspace paths."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RAGWorkspace:
    """Filesystem workspace for one Knowledge Base."""

    root: Path
    collection_id: int

    @property
    def collection_dir(self) -> Path:
        return self.root / "collections" / f"kb_{self.collection_id}"

    @property
    def chunks_dir(self) -> Path:
        return self.collection_dir / "chunks"

    @property
    def index_dir(self) -> Path:
        return self.collection_dir / "index"

    @property
    def manifest_path(self) -> Path:
        return self.collection_dir / "manifest.json"

    @property
    def chunks_path(self) -> Path:
        return self.chunks_dir / "all_chunks.json"

    @property
    def faiss_index_path(self) -> Path:
        return self.index_dir / "text_index.faiss"

    @property
    def fallback_index_path(self) -> Path:
        return self.index_dir / "text_index.npz"

    @property
    def image_faiss_index_path(self) -> Path:
        return self.index_dir / "image_index.faiss"

    @property
    def image_fallback_index_path(self) -> Path:
        return self.index_dir / "image_index.npz"

    def ensure(self) -> None:
        """Create workspace directories."""
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def write_manifest(self, *, embedding_backend: str, dim: int, chunk_count: int) -> None:
        """Persist lightweight workspace metadata."""
        self.ensure()
        payload = {
            "collection_id": self.collection_id,
            "embedding_backend": embedding_backend,
            "embedding_dim": dim,
            "chunk_count": chunk_count,
            "updated_at": int(time.time()),
        }
        self.manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
