"""Standard RAG service for ResearchMind Knowledge Bases."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
import re
from typing import TYPE_CHECKING

import numpy as np

from PaperTracker.config.rag import RAGConfig
from PaperTracker.llm.client import LLMApiClient
from PaperTracker.services.rag.image_embedder import ImageEmbedder
from PaperTracker.services.rag.embedder import TextEmbedder
from PaperTracker.services.rag.metadata_store import RAGMetadataStore
from PaperTracker.services.rag.reranker import RAGReranker
from PaperTracker.services.rag.downloader import resolve_model_source
from PaperTracker.services.rag.types import RAGChunk, RetrievalHit
from PaperTracker.services.rag.vector_store import VectorStore
from PaperTracker.services.rag.workspace import RAGWorkspace

if TYPE_CHECKING:
    from PaperTracker.storage.research import ResearchStore


@dataclass(slots=True)
class CollectionRAGService:
    """Owns Knowledge Base RAG workspaces and Standard retrieval."""

    root: Path
    config: RAGConfig | None = None
    store: "ResearchStore | None" = None
    embedder: TextEmbedder | None = None
    image_embedder: ImageEmbedder | None = None
    reranker: RAGReranker | None = None

    def _embedder(self) -> TextEmbedder:
        if self.embedder is None:
            self.embedder = TextEmbedder(
                model_name=self._embedding_model,
                models_dir=self._models_dir,
                dim=self._embedding_dim,
                batch_size=self._embedding_batch_size,
                allow_remote_model_loading=self._allow_remote_model_loading,
            )
        return self.embedder

    def _reranker(self) -> RAGReranker | None:
        if not self._reranker_enabled:
            return None
        if self.reranker is None:
            self.reranker = RAGReranker(
                self._reranker_model,
                models_dir=self._models_dir,
                allow_remote_model_loading=self._allow_remote_model_loading,
            )
        return self.reranker

    def _image_embedder(self) -> ImageEmbedder | None:
        if not self._image_vector_enabled:
            return None
        if self.image_embedder is None:
            self.image_embedder = ImageEmbedder(
                self._image_embedding_model,
                model_dir=self._image_embedding_model_dir,
                dim=self._image_embedding_dim,
                batch_size=self._image_embedding_batch_size,
                device_map=self._image_embedding_device_map,
                max_memory=self._image_embedding_max_memory,
            )
        return self.image_embedder if self.image_embedder.available else None

    @property
    def _models_dir(self) -> Path:
        return self.config.models_dir if self.config is not None else Path("models")

    @property
    def _embedding_model(self) -> str:
        return self.config.embedding_model if self.config is not None else "BAAI/bge-m3"

    @property
    def _embedding_dim(self) -> int:
        return self.config.embedding_dim if self.config is not None else 1024

    @property
    def _embedding_batch_size(self) -> int:
        return self.config.embedding_batch_size if self.config is not None else 32

    @property
    def _allow_remote_model_loading(self) -> bool:
        return self.config.allow_remote_model_loading if self.config is not None else False

    @property
    def _candidate_k(self) -> int:
        return self.config.candidate_k if self.config is not None else 24

    @property
    def _hybrid_enabled(self) -> bool:
        return self.config.hybrid_enabled if self.config is not None else True

    @property
    def _vector_weight(self) -> float:
        return self.config.vector_weight if self.config is not None else 0.65

    @property
    def _fts_weight(self) -> float:
        return self.config.fts_weight if self.config is not None else 0.35

    @property
    def _reranker_enabled(self) -> bool:
        return self.config.reranker_enabled if self.config is not None else True

    @property
    def _reranker_model(self) -> str:
        return self.config.reranker_model if self.config is not None else "BAAI/bge-reranker-base"

    @property
    def _image_vector_enabled(self) -> bool:
        return bool(self.config and self.config.image_vector_enabled)

    @property
    def _image_embedding_model(self) -> str:
        return self.config.image_embedding_model if self.config is not None else "Qwen/Qwen3-VL-Embedding-8B"

    @property
    def _image_embedding_model_dir(self) -> str:
        return self.config.image_embedding_model_dir if self.config is not None else ""

    @property
    def _image_embedding_dim(self) -> int:
        return self.config.image_embedding_dim if self.config is not None else 4096

    @property
    def _image_embedding_batch_size(self) -> int:
        return self.config.image_embedding_batch_size if self.config is not None else 2

    @property
    def _image_embedding_device_map(self) -> str:
        return self.config.image_embedding_device_map if self.config is not None else "auto"

    @property
    def _image_embedding_max_memory(self) -> str:
        return self.config.image_embedding_max_memory if self.config is not None else ""

    @property
    def _image_vector_weight(self) -> float:
        return self.config.image_vector_weight if self.config is not None else 0.35

    def workspace(self, collection_id: int) -> RAGWorkspace:
        """Return the workspace for one Knowledge Base."""
        return RAGWorkspace(root=self.root, collection_id=collection_id)

    def index_paper(
        self,
        *,
        collection_id: int,
        source: str,
        source_id: str,
        chunk_rows: list[dict],
    ) -> None:
        """Index one paper in a KB workspace.

        New papers are appended incrementally. Existing papers are skipped when
        their persisted chunks are unchanged, and rebuilt only when their chunk
        ids/content changed after re-parsing.
        """
        workspace = self.workspace(collection_id)
        workspace.ensure()
        metadata = RAGMetadataStore.load(workspace.chunks_path)
        embedder = self._embedder()
        vector_store = VectorStore(dim=embedder.dim)
        existing = metadata.paper_chunks(source, source_id)
        if existing and _same_indexed_chunks(existing, chunk_rows):
            if self._image_vector_enabled and not (
                workspace.image_faiss_index_path.exists() or workspace.image_fallback_index_path.exists()
            ):
                self._rebuild_image_index(workspace, metadata.all_chunks())
            workspace.write_manifest(
                embedding_backend=embedder.backend,
                dim=embedder.dim,
                chunk_count=len(metadata.all_chunks()),
            )
            return

        if not existing:
            added = metadata.append_paper_chunks(
                collection_id=collection_id,
                source=source,
                source_id=source_id,
                rows=chunk_rows,
            )
            metadata.save(workspace.chunks_path)
            if added:
                embeddings = embedder.encode([chunk.content for chunk in added])
                vector_store.append(
                    embeddings,
                    [chunk.vector_id for chunk in added],
                    faiss_path=workspace.faiss_index_path,
                    fallback_path=workspace.fallback_index_path,
                )
                self._append_image_index(workspace, added)
        else:
            metadata.replace_paper_chunks(
                collection_id=collection_id,
                source=source,
                source_id=source_id,
                rows=chunk_rows,
            )
            chunks = metadata.all_chunks()
            metadata.save(workspace.chunks_path)
            if chunks:
                embeddings = embedder.encode([chunk.content for chunk in chunks])
                ids = [chunk.vector_id for chunk in chunks]
            else:
                embeddings = np.empty((0, embedder.dim), dtype="float32")
                ids = []
            vector_store.save(
                embeddings,
                ids,
                faiss_path=workspace.faiss_index_path,
                fallback_path=workspace.fallback_index_path,
            )
            self._rebuild_image_index(workspace, chunks)
        workspace.write_manifest(
            embedding_backend=embedder.backend,
            dim=embedder.dim,
            chunk_count=len(metadata.all_chunks()),
        )

    def retrieve(
        self,
        collection_id: int,
        question: str,
        *,
        top_k: int = 6,
        mode: str = "standard",
    ) -> list[RetrievalHit]:
        """Retrieve top evidence chunks from one KB workspace."""
        if mode == "decompose":
            return self.retrieve_decompose(collection_id, question, top_k=top_k)
        if mode == "agent":
            return self.retrieve_agent(collection_id, question, top_k=top_k)
        return self.retrieve_standard(collection_id, question, top_k=top_k)

    def retrieve_standard(self, collection_id: int, question: str, *, top_k: int = 6) -> list[RetrievalHit]:
        """Standard dense/hybrid retrieval with optional rerank."""
        retrieval_question = _expand_retrieval_query(question)
        vector_hits = self._retrieve_vector(collection_id, retrieval_question, top_k=self._candidate_k)
        if not self._hybrid_enabled or self.store is None:
            merged = vector_hits
        else:
            fts_hits = self._retrieve_fts(collection_id, retrieval_question, top_k=self._candidate_k)
            merged = self._merge_hybrid(vector_hits, fts_hits)
        image_hits = self._retrieve_image_vector(collection_id, retrieval_question, top_k=self._candidate_k)
        if image_hits:
            merged = self._merge_image_hits(merged, image_hits)
        merged = _apply_query_context_prior(question, merged)

        reranker = self._reranker()
        if reranker is not None:
            return reranker.rerank(question, merged, top_k=top_k)
        return [
            replace(hit, rank=rank)
            for rank, hit in enumerate(sorted(merged, key=lambda item: item.score, reverse=True)[:top_k], start=1)
        ]

    def retrieve_decompose(self, collection_id: int, question: str, *, top_k: int = 6) -> list[RetrievalHit]:
        """Minimal query decomposition: retrieve sub-queries and fuse by reciprocal rank."""
        sub_queries = _decompose_question(question)
        if len(sub_queries) <= 1:
            return self.retrieve_standard(collection_id, question, top_k=top_k)
        candidates = []
        scores_by_db_id: dict[int, float] = {}
        best_by_db_id: dict[int, RetrievalHit] = {}
        for sub_query in sub_queries:
            hits = self.retrieve_standard(collection_id, sub_query, top_k=max(top_k, 4))
            for rank, hit in enumerate(hits, start=1):
                chunk_id = hit.chunk.db_chunk_id
                scores_by_db_id[chunk_id] = scores_by_db_id.get(chunk_id, 0.0) + 1.0 / (60 + rank)
                existing = best_by_db_id.get(chunk_id)
                if existing is None or hit.score > existing.score:
                    best_by_db_id[chunk_id] = replace(hit, retrieval_path=f"decompose:{hit.retrieval_path}")
        for chunk_id, score in scores_by_db_id.items():
            candidates.append(replace(best_by_db_id[chunk_id], score=score))
        reranker = self._reranker()
        candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
        if reranker is not None:
            return reranker.rerank(question, candidates, top_k=top_k)
        return [replace(hit, rank=rank) for rank, hit in enumerate(candidates[:top_k], start=1)]

    def retrieve_agent(self, collection_id: int, question: str, *, top_k: int = 6) -> list[RetrievalHit]:
        """Agent-lite retrieval: standard first, then decompose if evidence is sparse."""
        hits = self.retrieve_standard(collection_id, question, top_k=top_k)
        if len(hits) >= min(3, top_k):
            return [replace(hit, retrieval_path=f"agent:{hit.retrieval_path}") for hit in hits]
        return [
            replace(hit, retrieval_path=f"agent:{hit.retrieval_path}")
            for hit in self.retrieve_decompose(collection_id, question, top_k=top_k)
        ]

    def _retrieve_vector(self, collection_id: int, question: str, *, top_k: int) -> list[RetrievalHit]:
        """Retrieve vector hits from the KB workspace."""
        workspace = self.workspace(collection_id)
        metadata = RAGMetadataStore.load(workspace.chunks_path)
        chunks_by_id = metadata.by_vector_id()
        if not chunks_by_id:
            return []
        embedder = self._embedder()
        query_vec = embedder.encode_query(question)
        raw_hits = VectorStore(dim=embedder.dim).search(
            query_vec,
            top_k=top_k,
            faiss_path=workspace.faiss_index_path,
            fallback_path=workspace.fallback_index_path,
        )
        hits = []
        for rank, (vector_id, score) in enumerate(raw_hits, start=1):
            chunk = chunks_by_id.get(vector_id)
            if chunk is None:
                continue
            hits.append(
                RetrievalHit(
                    chunk=chunk,
                    score=score,
                    rank=rank,
                    retrieval_path="vector",
                    vector_score=score,
                )
            )
        return hits

    def _retrieve_fts(self, collection_id: int, question: str, *, top_k: int) -> list[RetrievalHit]:
        """Retrieve keyword/BM25 hits from SQLite FTS."""
        if self.store is None:
            return []
        rows = self.store.search_collection_chunks(collection_id, question, limit=top_k)
        hits = []
        for rank, row in enumerate(rows, start=1):
            rag_chunk = RAGChunk(
                vector_id=-1,
                chunk_id=f"sqlite:{row['chunk_id']}",
                db_chunk_id=int(row["chunk_id"]),
                collection_id=collection_id,
                source=str(row["source"]),
                source_id=str(row["source_id"]),
                paper_title=str(row.get("paper_title") or row["source_id"]),
                content=str(row.get("content") or ""),
                page_start=row.get("page_start"),
                page_end=row.get("page_end"),
                section_title=row.get("section_title"),
                token_count=len(str(row.get("content") or "").split()),
                modality=str(row.get("modality") or "text"),
                image_path=row.get("image_path"),
            )
            hits.append(
                RetrievalHit(
                    chunk=rag_chunk,
                    score=float(row.get("score") or 0.0),
                    rank=rank,
                    retrieval_path="fts",
                    fts_score=float(row.get("score") or 0.0),
                )
            )
        return hits

    def _retrieve_image_vector(self, collection_id: int, question: str, *, top_k: int) -> list[RetrievalHit]:
        """Retrieve figure chunks from the Qwen3-VL image-vector index."""
        image_embedder = self._image_embedder()
        if image_embedder is None:
            return []
        workspace = self.workspace(collection_id)
        metadata = RAGMetadataStore.load(workspace.chunks_path)
        chunks_by_db_id = {
            chunk.db_chunk_id: chunk
            for chunk in metadata.all_chunks()
            if chunk.modality == "figure" and chunk.image_path
        }
        if not chunks_by_db_id:
            return []
        query_vec = image_embedder.encode_text_query(question)
        raw_hits = VectorStore(dim=image_embedder.dim).search(
            query_vec,
            top_k=top_k,
            faiss_path=workspace.image_faiss_index_path,
            fallback_path=workspace.image_fallback_index_path,
        )
        hits = []
        for rank, (db_chunk_id, score) in enumerate(raw_hits, start=1):
            chunk = chunks_by_db_id.get(db_chunk_id)
            if chunk is None:
                continue
            hits.append(
                RetrievalHit(
                    chunk=chunk,
                    score=score,
                    rank=rank,
                    retrieval_path="image-vector",
                    image_score=score,
                )
            )
        return hits

    def _merge_hybrid(self, vector_hits: list[RetrievalHit], fts_hits: list[RetrievalHit]) -> list[RetrievalHit]:
        """Merge vector and FTS hits with normalized weighted scores."""
        vector_norm = _normalize_scores({hit.chunk.db_chunk_id: hit.score for hit in vector_hits})
        fts_norm = _normalize_scores({hit.chunk.db_chunk_id: hit.score for hit in fts_hits})
        best: dict[int, RetrievalHit] = {}
        for hit in [*vector_hits, *fts_hits]:
            chunk_id = hit.chunk.db_chunk_id
            existing = best.get(chunk_id)
            if existing is None or hit.score > existing.score:
                best[chunk_id] = hit

        merged = []
        for chunk_id, hit in best.items():
            vector_score = vector_norm.get(chunk_id)
            fts_score = fts_norm.get(chunk_id)
            score = self._vector_weight * (vector_score or 0.0) + self._fts_weight * (fts_score or 0.0)
            paths = []
            if vector_score is not None:
                paths.append("vector")
            if fts_score is not None:
                paths.append("fts")
            merged.append(
                replace(
                    hit,
                    score=score,
                    retrieval_path="+".join(paths),
                    vector_score=vector_score,
                    fts_score=fts_score,
                )
            )
        ranked = sorted(merged, key=lambda item: item.score, reverse=True)
        filtered = []
        image_only_count = 0
        for hit in ranked:
            image_only = hit.image_score is not None and hit.vector_score is None and hit.fts_score is None
            if image_only:
                image_only_count += 1
                if image_only_count > 2:
                    continue
            filtered.append(hit)
        return filtered

    def _merge_image_hits(self, text_hits: list[RetrievalHit], image_hits: list[RetrievalHit]) -> list[RetrievalHit]:
        """Merge text/hybrid hits with image-vector hits by DB chunk id."""
        text_norm = _normalize_scores({hit.chunk.db_chunk_id: hit.score for hit in text_hits})
        image_norm = _normalize_scores({hit.chunk.db_chunk_id: hit.score for hit in image_hits})
        best: dict[int, RetrievalHit] = {}
        for hit in [*text_hits, *image_hits]:
            chunk_id = hit.chunk.db_chunk_id
            existing = best.get(chunk_id)
            if existing is None or hit.score > existing.score:
                best[chunk_id] = hit

        image_weight = min(max(self._image_vector_weight, 0.0), 1.0)
        text_weight = 1.0 - image_weight
        merged = []
        for chunk_id, hit in best.items():
            text_score = text_norm.get(chunk_id)
            image_score = image_norm.get(chunk_id)
            score = text_weight * (text_score or 0.0) + image_weight * (image_score or 0.0)
            paths = []
            existing_path = hit.retrieval_path
            if text_score is not None:
                paths.append(existing_path if existing_path != "image-vector" else "text")
            if image_score is not None:
                paths.append("image-vector")
            merged.append(
                replace(
                    hit,
                    score=score,
                    retrieval_path="+".join(dict.fromkeys(paths)),
                    image_score=image_score,
                )
            )
        return sorted(merged, key=lambda item: item.score, reverse=True)

    def _append_image_index(self, workspace: RAGWorkspace, chunks: list[RAGChunk]) -> None:
        """Append figure image vectors for newly added chunks."""
        image_embedder = self._image_embedder()
        if image_embedder is None:
            return
        figures = _load_figure_images(chunks)
        if not figures:
            return
        images, captions, ids = figures
        embeddings = image_embedder.encode_images(images, captions=captions)
        VectorStore(dim=image_embedder.dim).append(
            embeddings,
            ids,
            faiss_path=workspace.image_faiss_index_path,
            fallback_path=workspace.image_fallback_index_path,
        )

    def _rebuild_image_index(self, workspace: RAGWorkspace, chunks: list[RAGChunk]) -> None:
        """Rebuild all figure image vectors after metadata compaction."""
        image_embedder = self._image_embedder()
        if image_embedder is None:
            return
        figures = _load_figure_images(chunks)
        if not figures:
            if workspace.image_faiss_index_path.exists():
                workspace.image_faiss_index_path.unlink()
            if workspace.image_fallback_index_path.exists():
                workspace.image_fallback_index_path.unlink()
            return
        images, captions, ids = figures
        embeddings = image_embedder.encode_images(images, captions=captions)
        VectorStore(dim=image_embedder.dim).save(
            embeddings,
            ids,
            faiss_path=workspace.image_faiss_index_path,
            fallback_path=workspace.image_fallback_index_path,
        )

    def debug_status(self) -> dict:
        """Return configured model/backend status for the UI."""
        embedder = self._embedder()
        reranker = self._reranker()
        return {
            "embedding_backend": embedder.backend,
            "embedding_model": self._embedding_model,
            "embedding_model_source": resolve_model_source(
                self._models_dir,
                self._embedding_model,
                allow_remote=self._allow_remote_model_loading,
            ) or "fallback:hashing",
            "reranker_enabled": self._reranker_enabled,
            "reranker_available": bool(reranker and reranker.available),
            "reranker_model": self._reranker_model,
            "hybrid_enabled": self._hybrid_enabled,
            "multimodal_enabled": bool(self.config and self.config.multimodal_enabled),
            "vl_caption_enabled": bool(self.config and self.config.vl_caption_enabled),
            "vl_caption_model": self.config.vl_caption_model if self.config else "",
            "vl_caption_adapter_path": self.config.vl_caption_adapter_path if self.config else "",
            "image_embedding_model": self.config.image_embedding_model if self.config else "",
            "image_embedding_model_dir": self.config.image_embedding_model_dir if self.config else "",
            "image_vector_enabled": self._image_vector_enabled,
        }

    def generate_answer(
        self,
        *,
        question: str,
        hits: list[RetrievalHit],
        llm_client: LLMApiClient | None,
        llm_model: str | None,
    ) -> str:
        """Generate a citation-backed answer from retrieved evidence."""
        if not hits:
            return (
                "I could not find indexed full-text evidence in this Knowledge Base yet. "
                "Please make sure the selected papers were downloaded, parsed, and indexed."
            )
        if llm_client is None or not llm_model:
            return _build_evidence_digest(hits)

        messages = _build_answer_messages(question, hits)
        return llm_client.chat_completion(
            messages=messages,
            model=llm_model,
            temperature=0.0,
            max_tokens=1000,
        ).strip()

    def generate_answer_stream(
        self,
        *,
        question: str,
        hits: list[RetrievalHit],
        llm_client: LLMApiClient | None,
        llm_model: str | None,
    ) -> Iterator[str]:
        """Stream a citation-backed answer from retrieved evidence."""
        if not hits:
            yield (
                "I could not find indexed full-text evidence in this Knowledge Base yet. "
                "Please make sure the selected papers were downloaded, parsed, and indexed."
            )
            return
        if llm_client is None or not llm_model:
            yield _build_evidence_digest(hits)
            return
        yield from llm_client.stream_chat_completion(
            messages=_build_answer_messages(question, hits),
            model=llm_model,
            temperature=0.0,
            max_tokens=1000,
        )


def _build_evidence_digest(hits: list[RetrievalHit]) -> str:
    lines = ["LLM answering is not configured, so this is a retrieval-backed evidence digest:"]
    for idx, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        lines.append(
            f"[{idx}] {chunk.paper_title} ({_page_label(chunk.page_start, chunk.page_end)}): "
            f"{_quote(chunk.content)}"
        )
    return "\n\n".join(lines)


def _build_answer_messages(question: str, hits: list[RetrievalHit]) -> list[dict[str, str]]:
    evidence = []
    for idx, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        page = _page_label(chunk.page_start, chunk.page_end)
        evidence.append(
            f"[{idx}] Title: {chunk.paper_title}\n"
            f"Modality: {chunk.modality}\n"
            f"Chunk ID: {chunk.chunk_id}\n"
            f"Pages: {page}\n"
            f"Section: {chunk.section_title or 'unknown'}\n"
            f"Evidence: {chunk.content}"
        )
    return [
        {
            "role": "system",
            "content": (
                "You are ResearchMind, a precise research assistant. "
                "Answer only from the provided evidence. "
                "If the evidence is insufficient, say so explicitly. "
                "When a Paper Summary evidence item is available, use it as the overview anchor, "
                "then verify or enrich it with nearby detailed evidence. "
                "For questions about a proposed method, architecture, or technical framework, "
                "prioritize evidence that describes the method design, components, pipeline, "
                "training, or algorithm. Do not treat illustrative result/example figure captions "
                "as sufficient evidence for the framework unless they explicitly describe it. "
                "Cite evidence inline using bracket numbers like [1]."
            ),
        },
        {
            "role": "user",
            "content": f"Question:\n{question}\n\nEvidence:\n\n{'\n\n'.join(evidence)}",
        },
    ]


def _quote(text: str, limit: int = 420) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _page_label(page_start: int | None, page_end: int | None) -> str:
    if page_start and page_end:
        return f"pages {page_start}-{page_end}"
    return "unknown page"


def _normalize_scores(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    low = min(values)
    high = max(values)
    if high == low:
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def _apply_query_context_prior(question: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
    """Bias retrieval toward the paper and evidence type the user explicitly asked for."""
    if not hits:
        return []

    title_matches = _paper_title_matches(question, hits)
    if title_matches:
        scoped = [hit for hit in hits if hit.chunk.paper_title in title_matches]
        if len(scoped) >= 2:
            hits = scoped

    experiment_intent = _is_experiment_question(question)
    method_intent = _is_method_or_experiment_question(question)
    adjusted = []
    seen_figure_text: set[str] = set()
    for hit in hits:
        score = hit.score
        content = hit.chunk.content.lower()
        if hit.chunk.modality == "summary":
            score += 0.22 if (experiment_intent or method_intent) else 0.08
        if experiment_intent:
            if _is_experiment_evidence(hit):
                score += 0.28
            if hit.chunk.modality != "figure":
                score += 0.18
        elif method_intent and hit.chunk.modality != "figure":
            score += 0.12

        if hit.chunk.modality == "figure":
            normalized = " ".join(content.split())
            if normalized in seen_figure_text:
                score -= 0.35
            seen_figure_text.add(normalized)
            if method_intent and _is_informative_visual(hit):
                score += 0.12
            if experiment_intent and not _is_experiment_visual(hit):
                score -= 0.45

        adjusted.append(replace(hit, score=score))
    return sorted(adjusted, key=lambda item: item.score, reverse=True)


def _expand_retrieval_query(question: str) -> str:
    """Add retrieval hints for common Chinese research questions."""
    leading_terms = _significant_tokens(question)
    hints = []
    if _is_experiment_question(question):
        hints.extend(
            [
                "dataset",
                "evaluation",
                "experiment",
                "comparison",
                "quantitative",
                "qualitative",
                "metric",
                "table",
                "benchmark",
                "results",
            ]
        )
    if _is_method_or_experiment_question(question):
        hints.extend(["method", "framework", "pipeline", "architecture", "module", "component"])
    if not hints:
        return question
    existing = {token.lower() for token in leading_terms}
    additions = [hint for hint in hints if hint.lower() not in existing]
    return " ".join([*leading_terms, *additions, question])


def _paper_title_matches(question: str, hits: list[RetrievalHit]) -> set[str]:
    q = question.lower()
    q_tokens = set(_significant_tokens(question))
    matches = set()
    for hit in hits:
        title = hit.chunk.paper_title
        title_lower = title.lower()
        title_tokens = set(_significant_tokens(title))
        if title_lower and title_lower in q:
            matches.add(title)
            continue
        if q_tokens & title_tokens:
            matches.add(title)
            continue
        for token in q_tokens:
            if len(token) >= 4 and token in title_lower:
                matches.add(title)
                break
    return matches


def _significant_tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text)
        if len(token) >= 4
    ]


def _is_experiment_question(question: str) -> bool:
    q = question.lower()
    return any(
        term in q
        for term in (
            "dataset",
            "datasets",
            "experiment",
            "experimental",
            "result",
            "results",
            "benchmark",
            "metric",
            "数据集",
            "实验",
            "结果",
            "指标",
            "验证",
        )
    )


def _is_experiment_evidence(hit: RetrievalHit) -> bool:
    text = f"{hit.chunk.section_title or ''} {hit.chunk.content}".lower()
    return any(
        term in text
        for term in (
            "dataset",
            "datasets",
            "benchmark",
            "evaluation",
            "experiment",
            "experimental",
            "comparison",
            "quantitative",
            "qualitative",
            "metric",
            "ablation",
            "table",
            "数据集",
            "实验",
            "评估",
            "对比",
            "消融",
            "指标",
        )
    )


def _is_experiment_visual(hit: RetrievalHit) -> bool:
    text = f"{hit.chunk.section_title or ''} {hit.chunk.content}".lower()
    return any(
        term in text
        for term in (
            "table",
            "ablation",
            "comparison",
            "quantitative",
            "qualitative",
            "benchmark",
            "metric",
            "overview",
            "表",
            "消融",
            "对比",
            "指标",
        )
    )


def _is_informative_visual(hit: RetrievalHit) -> bool:
    text = f"{hit.chunk.section_title or ''} {hit.chunk.content}".lower()
    return any(
        term in text
        for term in (
            "pipeline",
            "framework",
            "architecture",
            "overview",
            "module",
            "component",
            "diagram",
            "flow",
            "method",
            "table",
            "ablation",
            "comparison",
            "流程",
            "框架",
            "架构",
            "模块",
            "方法",
            "表",
            "消融",
            "对比",
        )
    )


def _is_method_or_experiment_question(question: str) -> bool:
    q = question.lower()
    return _is_experiment_question(question) or any(
        term in q
        for term in (
            "framework",
            "architecture",
            "pipeline",
            "method",
            "algorithm",
            "technical",
            "技术框架",
            "框架",
            "方法",
            "算法",
            "流程",
        )
    )


def _load_figure_images(chunks: list[RAGChunk]) -> tuple[list, list[str], list[int]] | None:
    """Load figure images for image-vector indexing."""
    try:
        from PIL import Image
    except Exception:
        return None
    images = []
    captions = []
    ids = []
    for chunk in chunks:
        if chunk.modality != "figure" or not chunk.image_path:
            continue
        image_path = Path(chunk.image_path)
        if not image_path.exists():
            continue
        try:
            images.append(Image.open(image_path).convert("RGB"))
        except Exception:
            continue
        captions.append(chunk.content)
        ids.append(chunk.db_chunk_id)
    if not images:
        return None
    return images, captions, ids


def _same_indexed_chunks(existing: list[RAGChunk], rows: list[dict]) -> bool:
    """Return true when DB chunks and metadata already match exactly."""
    if len(existing) != len(rows):
        return False
    existing_by_db_id = {chunk.db_chunk_id: chunk for chunk in existing}
    for row in rows:
        chunk = existing_by_db_id.get(int(row["id"]))
        if chunk is None:
            return False
        if chunk.content != str(row.get("content") or ""):
            return False
        if chunk.section_title != row.get("section_title"):
            return False
        if chunk.page_start != row.get("page_start") or chunk.page_end != row.get("page_end"):
            return False
        if chunk.modality != str(row.get("modality") or "text"):
            return False
        if chunk.image_path != row.get("image_path"):
            return False
    return True


def _decompose_question(question: str) -> list[str]:
    separators = [" and ", "以及", "并且", "同时", "；", ";"]
    parts = [question.strip()]
    for sep in separators:
        expanded = []
        for part in parts:
            expanded.extend(item.strip() for item in part.split(sep) if item.strip())
        parts = expanded
    unique = []
    for part in [question.strip(), *parts]:
        if part and part not in unique:
            unique.append(part)
    return unique[:4]
