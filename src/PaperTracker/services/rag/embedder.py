"""Text embedding backends for ResearchMind RAG."""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import numpy as np

from PaperTracker.services.rag.downloader import resolve_model_source


class TextEmbedder:
    """BGE-M3 embedder with deterministic fallback.

    When FlagEmbedding is installed, this class uses BGE-M3 dense embeddings.
    Otherwise it falls back to a stable hashing embedder so the RAG pipeline can
    run in lightweight development and CI environments.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        *,
        models_dir: Path = Path("models"),
        dim: int = 1024,
        batch_size: int = 32,
        allow_remote_model_loading: bool = False,
    ) -> None:
        self.model_name = model_name
        self._dim = dim
        self._batch_size = batch_size
        self._model = None
        self._backend = "hashing"
        self._lock = threading.Lock()
        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            source = resolve_model_source(
                models_dir,
                model_name,
                allow_remote=allow_remote_model_loading,
            )
            if source is not None:
                self._model = BGEM3FlagModel(source, use_fp16=torch.cuda.is_available(), devices=[device])
                self._backend = "bge-m3"
        except Exception:
            self._model = None

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into L2-normalized float32 vectors."""
        if self._model is not None:
            with self._lock:
                output = self._model.encode(
                    texts,
                    batch_size=self._batch_size,
                    max_length=512,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )
            embeddings = np.array(output["dense_vecs"], dtype="float32")
        else:
            embeddings = np.vstack([_hash_embed(text, self._dim) for text in texts]).astype("float32")

        return _normalize(embeddings)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode one query."""
        return self.encode([query])


def _hash_embed(text: str, dim: int) -> np.ndarray:
    """Build a deterministic lexical vector for fallback retrieval."""
    vector = np.zeros(dim, dtype="float32")
    tokens = [token for token in _tokenize(text) if token]
    if not tokens:
        return vector
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    return vector


def _tokenize(text: str) -> list[str]:
    cleaned = []
    current = []
    for char in text.lower():
        if char.isalnum() or char in {"_", "-"}:
            current.append(char)
        else:
            if current:
                cleaned.append("".join(current))
                current = []
    if current:
        cleaned.append("".join(current))
    return cleaned


def _normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (embeddings / norms).astype("float32")
