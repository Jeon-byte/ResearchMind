"""Model download helpers for RAG backends."""

from __future__ import annotations

from pathlib import Path


def download_rag_models(
    *,
    models_dir: Path,
    embedding_model: str,
    reranker_model: str,
) -> list[Path]:
    """Download embedding and reranker models into a local models directory."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - depends on optional environment
        raise RuntimeError("Install huggingface-hub before downloading RAG models") from error

    models_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for model_id in (embedding_model, reranker_model):
        local_dir = _local_model_dir(models_dir, model_id)
        snapshot_download(repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False)
        downloaded.append(local_dir)
    return downloaded


def resolve_model_source(models_dir: Path, model_id: str, *, allow_remote: bool = False) -> str | None:
    """Prefer local model directory when present, otherwise return the model id."""
    local_dir = _local_model_dir(models_dir, model_id)
    if _looks_like_complete_model(local_dir):
        return str(local_dir)
    return model_id if allow_remote else None


def _local_model_dir(models_dir: Path, model_id: str) -> Path:
    return models_dir / model_id


def _looks_like_complete_model(path: Path) -> bool:
    return path.exists() and (path / "config.json").exists()
