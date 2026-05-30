"""Optional BGE reranker for RAG retrieval results."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PaperTracker.services.rag.downloader import resolve_model_source
from PaperTracker.services.rag.types import RetrievalHit


class RAGReranker:
    """Cross-encoder reranker with no-op fallback when dependencies are missing."""

    def __init__(
        self,
        model_name: str,
        *,
        models_dir: Path,
        allow_remote_model_loading: bool = False,
    ) -> None:
        self.model_name = model_name
        self._model = None
        try:
            from sentence_transformers import CrossEncoder
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            source = resolve_model_source(
                models_dir,
                model_name,
                allow_remote=allow_remote_model_loading,
            )
            if source is not None:
                self._model = CrossEncoder(source, device=device)
        except Exception:
            self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def rerank(self, question: str, hits: list[RetrievalHit], *, top_k: int) -> list[RetrievalHit]:
        """Rerank hits by cross-encoder score if available."""
        if not hits:
            return []
        if self._model is None:
            return [replace(hit, rank=rank) for rank, hit in enumerate(hits[:top_k], start=1)]
        pairs = [(question, hit.chunk.content) for hit in hits]
        scores = self._model.predict(pairs).tolist()
        ranked = sorted(
            (_adjust_score(question, float(score), hit), hit)
            for score, hit in zip(scores, hits)
        )
        ranked.reverse()
        reranked = []
        for rank, (score, hit) in enumerate(ranked[:top_k], start=1):
            reranked.append(
                replace(
                    hit,
                    rank=rank,
                    score=float(score),
                    rerank_score=float(score),
                )
            )
        return reranked


def _adjust_score(question: str, score: float, hit: RetrievalHit) -> float:
    """Apply small modality priors after reranking."""
    q = question.lower()
    framework_intent = any(
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
    if not framework_intent:
        return score

    content = hit.chunk.content.lower()
    if hit.chunk.modality != "figure":
        return score + 0.15
    if any(term in content for term in ("architecture", "framework", "pipeline", "overview", "method", "算法", "框架", "流程")):
        return score
    if any(term in content for term in ("example", "result", "rendered", "visualization", "dataset", "示例", "结果", "可视化")):
        return score - 0.35
    return score - 0.15
