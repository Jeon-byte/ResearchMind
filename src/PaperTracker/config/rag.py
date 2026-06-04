"""RAG configuration for ResearchMind Knowledge Bases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from PaperTracker.config.common import expect_bool, expect_int, expect_str, get_section


@dataclass(frozen=True, slots=True)
class RAGConfig:
    """Configuration for Knowledge Base RAG workspaces."""

    models_dir: Path
    embedding_model: str
    embedding_dim: int
    embedding_batch_size: int
    allow_remote_model_loading: bool
    top_k: int
    candidate_k: int
    hybrid_enabled: bool
    vector_weight: float
    fts_weight: float
    reranker_enabled: bool
    reranker_model: str
    debug_enabled: bool
    default_mode: str
    decompose_enabled: bool
    agent_enabled: bool
    agent_max_steps: int
    agent_queries_per_step: int
    agent_evidence_cap: int
    agent_planner_enabled: bool
    multimodal_enabled: bool
    vl_caption_enabled: bool
    vl_caption_model: str
    vl_caption_model_dir: str
    vl_caption_adapter_path: str
    vl_caption_device_map: str
    vl_caption_max_memory: str
    vl_caption_max_figures_per_paper: int
    vl_caption_max_new_tokens: int
    image_vector_enabled: bool
    image_embedding_model: str
    image_embedding_model_dir: str
    image_embedding_device_map: str
    image_embedding_max_memory: str
    image_embedding_dim: int
    image_embedding_batch_size: int
    image_vector_weight: float


def load_rag(raw: Mapping[str, Any]) -> RAGConfig:
    """Load RAG config from root config mapping."""
    section = get_section(raw, "rag", required=False)
    return RAGConfig(
        models_dir=Path(expect_str(section.get("models_dir", "models"), "rag.models_dir")),
        embedding_model=expect_str(section.get("embedding_model", "BAAI/bge-m3"), "rag.embedding_model"),
        embedding_dim=expect_int(section.get("embedding_dim", 1024), "rag.embedding_dim"),
        embedding_batch_size=expect_int(section.get("embedding_batch_size", 32), "rag.embedding_batch_size"),
        allow_remote_model_loading=expect_bool(
            section.get("allow_remote_model_loading", False),
            "rag.allow_remote_model_loading",
        ),
        top_k=expect_int(section.get("top_k", 6), "rag.top_k"),
        candidate_k=expect_int(section.get("candidate_k", 24), "rag.candidate_k"),
        hybrid_enabled=expect_bool(section.get("hybrid_enabled", True), "rag.hybrid_enabled"),
        vector_weight=_expect_float(section.get("vector_weight", 0.65), "rag.vector_weight"),
        fts_weight=_expect_float(section.get("fts_weight", 0.35), "rag.fts_weight"),
        reranker_enabled=expect_bool(section.get("reranker_enabled", True), "rag.reranker_enabled"),
        reranker_model=expect_str(section.get("reranker_model", "BAAI/bge-reranker-base"), "rag.reranker_model"),
        debug_enabled=expect_bool(section.get("debug_enabled", True), "rag.debug_enabled"),
        default_mode=expect_str(section.get("default_mode", "standard"), "rag.default_mode").strip().lower(),
        decompose_enabled=expect_bool(section.get("decompose_enabled", True), "rag.decompose_enabled"),
        agent_enabled=expect_bool(section.get("agent_enabled", False), "rag.agent_enabled"),
        agent_max_steps=expect_int(section.get("agent_max_steps", 3), "rag.agent_max_steps"),
        agent_queries_per_step=expect_int(section.get("agent_queries_per_step", 2), "rag.agent_queries_per_step"),
        agent_evidence_cap=expect_int(section.get("agent_evidence_cap", 16), "rag.agent_evidence_cap"),
        agent_planner_enabled=expect_bool(
            section.get("agent_planner_enabled", False),
            "rag.agent_planner_enabled",
        ),
        multimodal_enabled=expect_bool(section.get("multimodal_enabled", False), "rag.multimodal_enabled"),
        vl_caption_enabled=expect_bool(section.get("vl_caption_enabled", False), "rag.vl_caption_enabled"),
        vl_caption_model=expect_str(
            section.get("vl_caption_model", "Qwen/Qwen3-VL-8B-Instruct"),
            "rag.vl_caption_model",
        ),
        vl_caption_model_dir=expect_str(section.get("vl_caption_model_dir", ""), "rag.vl_caption_model_dir"),
        vl_caption_adapter_path=expect_str(
            section.get("vl_caption_adapter_path", ""),
            "rag.vl_caption_adapter_path",
        ),
        vl_caption_device_map=expect_str(section.get("vl_caption_device_map", "auto"), "rag.vl_caption_device_map"),
        vl_caption_max_memory=expect_str(section.get("vl_caption_max_memory", ""), "rag.vl_caption_max_memory"),
        vl_caption_max_figures_per_paper=expect_int(
            section.get("vl_caption_max_figures_per_paper", 8),
            "rag.vl_caption_max_figures_per_paper",
        ),
        vl_caption_max_new_tokens=expect_int(section.get("vl_caption_max_new_tokens", 220), "rag.vl_caption_max_new_tokens"),
        image_vector_enabled=expect_bool(section.get("image_vector_enabled", False), "rag.image_vector_enabled"),
        image_embedding_model=expect_str(
            section.get("image_embedding_model", "Qwen/Qwen3-VL-Embedding-8B"),
            "rag.image_embedding_model",
        ),
        image_embedding_model_dir=expect_str(
            section.get("image_embedding_model_dir", ""),
            "rag.image_embedding_model_dir",
        ),
        image_embedding_device_map=expect_str(
            section.get("image_embedding_device_map", "auto"),
            "rag.image_embedding_device_map",
        ),
        image_embedding_max_memory=expect_str(
            section.get("image_embedding_max_memory", ""),
            "rag.image_embedding_max_memory",
        ),
        image_embedding_dim=expect_int(section.get("image_embedding_dim", 4096), "rag.image_embedding_dim"),
        image_embedding_batch_size=expect_int(
            section.get("image_embedding_batch_size", 2),
            "rag.image_embedding_batch_size",
        ),
        image_vector_weight=_expect_float(section.get("image_vector_weight", 0.35), "rag.image_vector_weight"),
    )


def check_rag(config: RAGConfig) -> None:
    """Validate RAG config values."""
    if config.embedding_dim <= 0:
        raise ValueError("rag.embedding_dim must be positive")
    if config.embedding_batch_size <= 0:
        raise ValueError("rag.embedding_batch_size must be positive")
    if config.top_k <= 0:
        raise ValueError("rag.top_k must be positive")
    if config.candidate_k < config.top_k:
        raise ValueError("rag.candidate_k must be >= rag.top_k")
    if config.vector_weight < 0 or config.fts_weight < 0:
        raise ValueError("rag.vector_weight and rag.fts_weight must be >= 0")
    if config.vector_weight == 0 and config.fts_weight == 0:
        raise ValueError("rag.vector_weight and rag.fts_weight cannot both be 0")
    if config.default_mode not in {"standard", "decompose", "agent"}:
        raise ValueError("rag.default_mode must be standard, decompose, or agent")
    if config.agent_max_steps <= 0:
        raise ValueError("rag.agent_max_steps must be positive")
    if config.agent_queries_per_step <= 0:
        raise ValueError("rag.agent_queries_per_step must be positive")
    if config.agent_evidence_cap <= 0:
        raise ValueError("rag.agent_evidence_cap must be positive")
    if config.vl_caption_max_figures_per_paper == 0 or config.vl_caption_max_figures_per_paper < -1:
        raise ValueError("rag.vl_caption_max_figures_per_paper must be -1 or a positive integer")
    if config.vl_caption_max_new_tokens <= 0:
        raise ValueError("rag.vl_caption_max_new_tokens must be positive")
    if config.image_embedding_dim <= 0:
        raise ValueError("rag.image_embedding_dim must be positive")
    if config.image_embedding_batch_size <= 0:
        raise ValueError("rag.image_embedding_batch_size must be positive")
    if config.image_vector_weight < 0:
        raise ValueError("rag.image_vector_weight must be >= 0")


def _expect_float(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be a number")
    return float(value)
