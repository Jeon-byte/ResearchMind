"""Optional VLM figure captioning for visually grounded retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PaperTracker.config.rag import RAGConfig
from PaperTracker.utils.log import log


_PROMPT_WITH_CAPTION = """Generate a compact retrieval caption for an academic figure.

Requirements:
- One line, max 80 words
- Preserve key technical terms from the original caption
- Identify the figure type
- Describe the figure as indexed content, not as an explanation
- Include important components, entities, relations, and operations shown in the figure
- Include the main model or method name if visible
- Add 1-2 short query-style phrases
- Avoid generic words
- Do not invent unseen details

Format:
<type>; <content>; <components/relations>; <keywords + query phrases>

Original caption:
"{caption}"
"""


_PROMPT_WITHOUT_CAPTION = """Generate a retrieval-oriented caption for an academic figure.

Rules:
- One line, max 80 words
- If informative:
  <type>; <content>; <components>; <keywords + query phrases>
- If clearly non-informative:
  irrelevant; non-informative; none; skip
- Prefer recall over filtering
- Include method or model name if visible
- Do not hallucinate details

Types:
architecture, pipeline, mechanism, comparison, curve, chart, table, qualitative result, quantitative result
"""


@dataclass(slots=True)
class VisionCaptionGenerator:
    """Lazy Qwen3-VL caption generator with optional LoRA adapter."""

    model_name: str
    model_dir: Path | None = None
    adapter_path: Path | None = None
    device_map: str = "auto"
    max_memory: str = ""
    max_new_tokens: int = 220

    _model: Any = None
    _processor: Any = None

    @classmethod
    def from_config(cls, config: RAGConfig) -> "VisionCaptionGenerator":
        """Build a generator from RAG config."""
        model_dir = Path(config.vl_caption_model_dir).expanduser() if config.vl_caption_model_dir else None
        adapter_path = Path(config.vl_caption_adapter_path).expanduser() if config.vl_caption_adapter_path else None
        return cls(
            model_name=config.vl_caption_model,
            model_dir=model_dir,
            adapter_path=adapter_path,
            device_map=config.vl_caption_device_map,
            max_memory=config.vl_caption_max_memory,
            max_new_tokens=config.vl_caption_max_new_tokens,
        )

    def generate_from_bytes(self, image_bytes: bytes, original_caption: str = "") -> str:
        """Generate a retrieval-oriented caption for one extracted image."""
        try:
            from PIL import Image

            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as error:  # noqa: BLE001 - invalid PDF assets should not break ingestion
            log.warning("Could not decode extracted figure image for VLM captioning: %s", error)
            return ""
        return self.generate(image, original_caption)

    def generate(self, image: Any, original_caption: str = "") -> str:
        """Generate a caption using Qwen3-VL, returning an empty string on failure."""
        if image is None:
            return ""
        try:
            if self._model is None or self._processor is None:
                self._load()
            return self._generate_loaded(image, original_caption)
        except Exception as error:  # noqa: BLE001 - optional VLM should degrade gracefully
            log.warning("VLM figure captioning failed: %s", error)
            return ""

    def _load(self) -> None:
        """Lazy-load Qwen3-VL and optional PEFT adapter."""
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        source = self._model_source()
        log.info("Loading VLM caption model: %s", source)
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            source,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=_resolve_device_map(self.device_map) if torch.cuda.is_available() else None,
            max_memory=_parse_max_memory(self.max_memory) if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        if self.adapter_path and self.adapter_path.exists():
            from peft import PeftModel

            log.info("Loading VLM LoRA adapter: %s", self.adapter_path)
            self._model = PeftModel.from_pretrained(base_model, str(self.adapter_path))
        else:
            self._model = base_model
        self._processor = AutoProcessor.from_pretrained(source, trust_remote_code=True)
        self._model.eval()

    def _model_source(self) -> str:
        if self.model_dir and self.model_dir.exists():
            return str(self.model_dir)
        return self.model_name

    def _generate_loaded(self, image: Any, original_caption: str) -> str:
        import torch

        prompt = _build_prompt(original_caption)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = _process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        device = next(self._model.parameters()).device
        inputs = inputs.to(device if str(device) != "meta" else "cuda:0")
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=8,
                do_sample=False,
                num_beams=3,
                repetition_penalty=1.25,
                no_repeat_ngram_size=4,
                early_stopping=True,
            )
        generated_ids = generated[0][input_len:]
        return self._processor.decode(generated_ids, skip_special_tokens=True).strip()

    def release(self) -> None:
        """Release model memory after an ingestion batch."""
        try:
            import gc
            import torch

            self._model = None
            self._processor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            self._model = None
            self._processor = None


def merge_captions(original_caption: str, generated_caption: str) -> str:
    """Merge original and VLM captions for indexing."""
    parts = [part.strip() for part in (original_caption, generated_caption) if part and part.strip()]
    return "\n".join(parts)


def _build_prompt(original_caption: str) -> str:
    caption = original_caption.strip()
    if caption:
        return _PROMPT_WITH_CAPTION.replace("{caption}", caption)
    return _PROMPT_WITHOUT_CAPTION


def _process_vision_info(messages: list[dict[str, Any]]) -> tuple[Any, Any]:
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as error:
        raise RuntimeError("qwen-vl-utils is required for rag.vl_caption_enabled=true") from error
    return process_vision_info(messages)


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
