"""Vector index storage for Knowledge Base RAG workspaces."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class VectorStore:
    """FAISS-backed vector store with numpy fallback."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def save(self, embeddings: np.ndarray, ids: list[int], *, faiss_path: Path, fallback_path: Path) -> None:
        """Persist vectors with FAISS if available, otherwise numpy."""
        faiss_path.parent.mkdir(parents=True, exist_ok=True)
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import faiss  # type: ignore

            index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))
            index.add_with_ids(embeddings.astype("float32"), np.asarray(ids, dtype=np.int64))
            faiss.write_index(index, str(faiss_path))
            if fallback_path.exists():
                fallback_path.unlink()
        except Exception:
            if faiss_path.exists():
                faiss_path.unlink()
            np.savez_compressed(fallback_path, embeddings=embeddings.astype("float32"), ids=np.asarray(ids, dtype=np.int64))

    def append(self, embeddings: np.ndarray, ids: list[int], *, faiss_path: Path, fallback_path: Path) -> None:
        """Append vectors to an existing index without rebuilding old vectors."""
        if len(ids) == 0:
            return
        faiss_path.parent.mkdir(parents=True, exist_ok=True)
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import faiss  # type: ignore

            if faiss_path.exists():
                index = faiss.read_index(str(faiss_path))
            else:
                index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))
            index.add_with_ids(embeddings.astype("float32"), np.asarray(ids, dtype=np.int64))
            faiss.write_index(index, str(faiss_path))
            if fallback_path.exists():
                fallback_path.unlink()
            return
        except Exception:
            pass

        existing_embeddings = np.empty((0, self.dim), dtype="float32")
        existing_ids = np.empty((0,), dtype=np.int64)
        if fallback_path.exists():
            data = np.load(fallback_path)
            existing_embeddings = data["embeddings"].astype("float32")
            existing_ids = data["ids"].astype("int64")
        np.savez_compressed(
            fallback_path,
            embeddings=np.vstack([existing_embeddings, embeddings.astype("float32")]),
            ids=np.concatenate([existing_ids, np.asarray(ids, dtype=np.int64)]),
        )

    def search(
        self,
        query_vec: np.ndarray,
        *,
        top_k: int,
        faiss_path: Path,
        fallback_path: Path,
    ) -> list[tuple[int, float]]:
        """Search persisted vectors and return `(vector_id, score)` pairs."""
        if top_k <= 0:
            return []
        if faiss_path.exists():
            try:
                import faiss  # type: ignore

                index = faiss.read_index(str(faiss_path))
                scores, indices = index.search(query_vec.astype("float32"), min(top_k, index.ntotal))
                return [
                    (int(idx), float(score))
                    for idx, score in zip(indices[0], scores[0])
                    if int(idx) >= 0
                ]
            except Exception:
                pass

        if not fallback_path.exists():
            return []
        data = np.load(fallback_path)
        embeddings = data["embeddings"].astype("float32")
        ids = data["ids"].astype("int64")
        if len(ids) == 0:
            return []
        scores = embeddings @ query_vec[0].astype("float32")
        order = np.argsort(scores)[::-1][:top_k]
        return [(int(ids[idx]), float(scores[idx])) for idx in order]
