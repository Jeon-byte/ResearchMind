"""Qwen3-VL image embedding backend for figure retrieval."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from PaperTracker.utils.log import log


class ImageEmbedder:
    """Text-to-image shared-space embedder backed by Qwen3-VL-Embedding."""

    _IMAGE_INSTRUCTION = "Represent this image for retrieval."
    _TEXT_INSTRUCTION = "Represent the query for image retrieval:"

    def __init__(
        self,
        model_name: str,
        *,
        model_dir: str = "",
        dim: int = 4096,
        batch_size: int = 2,
        device_map: str = "auto",
        max_memory: str = "",
    ) -> None:
        self.model_name = model_name
        self.model_dir = model_dir
        self._dim = dim
        self._batch_size = batch_size
        self._device_map = device_map
        self._max_memory = max_memory
        self._model: Any = None
        self._processor: Any = None
        self._device: Any = None
        self._available = False
        self._lock = threading.Lock()

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def available(self) -> bool:
        if self._available:
            return True
        try:
            self._load()
        except Exception as error:  # noqa: BLE001 - optional image retrieval must degrade gracefully
            log.warning("Image embedder unavailable: %s", error)
            return False
        return self._available

    def encode_images(self, images: list[Any], captions: list[str] | None = None) -> np.ndarray:
        """Encode PIL images into normalized vectors."""
        if not images:
            return np.empty((0, self.dim), dtype="float32")
        self._load()
        all_vecs: list[np.ndarray] = []
        batch_size = max(1, self._batch_size)
        for start in range(0, len(images), batch_size):
            batch_images = images[start : start + batch_size]
            batch_captions = captions[start : start + batch_size] if captions else [None] * len(batch_images)
            conversations = [
                self._build_conversation(
                    image=image,
                    text=f"Caption: {caption}" if caption else None,
                    instruction=self._IMAGE_INSTRUCTION,
                )
                for image, caption in zip(batch_images, batch_captions, strict=True)
            ]
            inputs = self._prepare_batch_inputs(conversations, images=batch_images)
            all_vecs.append(self._forward(inputs))
        return _normalize(np.vstack(all_vecs))

    def encode_text_query(self, query: str) -> np.ndarray:
        """Encode a text query into the image retrieval space."""
        self._load()
        conversations = [
            self._build_conversation(
                text=query,
                instruction=self._TEXT_INSTRUCTION,
            )
        ]
        inputs = self._prepare_batch_inputs(conversations)
        return _normalize(self._forward(inputs))

    def _load(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        with self._lock:
            if self._model is not None and self._processor is not None:
                return
            import torch
            from transformers.models.qwen3_vl.modeling_qwen3_vl import (
                Qwen3VLConfig,
                Qwen3VLModel,
                Qwen3VLPreTrainedModel,
            )
            from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
            from transformers.modeling_outputs import ModelOutput

            @dataclass
            class Qwen3VLForEmbeddingOutput(ModelOutput):
                last_hidden_state: torch.FloatTensor | None = None
                attention_mask: torch.Tensor | None = None

            class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
                config_class = Qwen3VLConfig
                _checkpoint_conversion_mapping = {}
                accepts_loss_kwargs = False

                def __init__(self, config: Qwen3VLConfig):
                    super().__init__(config)
                    self.model = Qwen3VLModel(config)
                    self.post_init()

                @property
                def language_model(self):  # noqa: ANN202 - mirrors transformers wrapper API
                    return self.model.language_model

                @property
                def visual(self):  # noqa: ANN202 - mirrors transformers wrapper API
                    return self.model.visual

                def forward(self, **kwargs: Any) -> Qwen3VLForEmbeddingOutput:
                    attention_mask = kwargs.get("attention_mask")
                    outputs = self.model(**kwargs)
                    return Qwen3VLForEmbeddingOutput(
                        last_hidden_state=outputs.last_hidden_state,
                        attention_mask=attention_mask,
                    )

            source = self._model_source()
            log.info("Loading image embedding model: %s", source)
            self._model = Qwen3VLForEmbedding.from_pretrained(
                source,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=_resolve_device_map(self._device_map) if torch.cuda.is_available() else "cpu",
                max_memory=_parse_max_memory(self._max_memory) if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            self._processor = Qwen3VLProcessor.from_pretrained(source, padding_side="right")
            self._model.eval()
            self._device = _input_device()
            self._validate_dim()
            self._available = True

    def _model_source(self) -> str:
        if self.model_dir and Path(self.model_dir).expanduser().exists():
            return str(Path(self.model_dir).expanduser())
        return self.model_name

    def _validate_dim(self) -> None:
        actual = getattr(getattr(self._model.config, "text_config", None), "hidden_size", None)
        if actual is not None and int(actual) != self.dim:
            raise ValueError(f"rag.image_embedding_dim={self.dim} but model hidden_size={actual}")

    def _build_conversation(
        self,
        *,
        text: str | None = None,
        image: Any | None = None,
        instruction: str,
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": text or "NULL"})
        return [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content},
        ]

    def _prepare_batch_inputs(
        self,
        conversations: list[list[dict[str, Any]]],
        images: list[Any] | None = None,
    ) -> dict[str, Any]:
        texts = self._processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
        inputs = self._processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return {key: value.to(self._device) for key, value in inputs.items()}

    def _forward(self, inputs: dict[str, Any]) -> np.ndarray:
        import torch

        with torch.no_grad():
            outputs = self._model(**inputs)
        pooled = _last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return pooled.float().cpu().numpy().astype("float32")


def _last_token_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    import torch

    flipped_mask = attention_mask.flip(dims=[1])
    last_one_pos = flipped_mask.argmax(dim=1)
    col = attention_mask.shape[1] - last_one_pos - 1
    row = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
    return last_hidden_state[row, col]


def _normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (embeddings / norms).astype("float32")


def _resolve_device_map(value: str) -> Any:
    cleaned = (value or "auto").strip()
    if cleaned in {"auto", "balanced", "balanced_low_0", "sequential"}:
        return cleaned
    if cleaned.startswith("cuda:"):
        return {"": cleaned}
    if cleaned.isdigit():
        return {"": f"cuda:{cleaned}"}
    return cleaned


def _parse_max_memory(value: str) -> dict[int | str, str] | None:
    cleaned = (value or "").strip()
    if not cleaned:
        return None
    result: dict[int | str, str] = {}
    for item in cleaned.split(","):
        if ":" not in item:
            continue
        key, memory = item.split(":", 1)
        key = key.strip()
        memory = memory.strip()
        if not key or not memory:
            continue
        result[int(key) if key.isdigit() else key] = memory
    return result or None


def _input_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"
