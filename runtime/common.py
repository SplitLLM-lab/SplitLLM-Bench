from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.generation.logits_process import (
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from model.codec import ActivationCodec, EncodedActivation


def pick_device_and_dtype(
    device_arg: str | None,
    dtype_arg: str,
) -> tuple[torch.device, torch.dtype]:
    if device_arg in (None, "auto"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)

    if dtype_arg == "auto":
        if device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
    else:
        if not hasattr(torch, dtype_arg):
            raise ValueError(f"unsupported dtype: {dtype_arg}")
        dtype = getattr(torch, dtype_arg)

    return device, dtype


def resolve_dir(path_or_repo: str, revision: str | None = None) -> str:
    p = Path(path_or_repo).expanduser()
    if p.is_dir():
        return str(p.resolve())

    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id=path_or_repo,
        revision=revision,
    )
    return str(Path(local_dir).resolve())


def normalize_quant_mode(raw: str | None) -> str:
    mode = (raw or "none").strip().lower()
    aliases = {
        "": "none",
        "none": "none",
        "default": "none",
        "fp": "none",
        "fp16": "none",
        "bf16": "none",
        "float": "none",
        "hf_int8": "bnb_8bit",
        "int8": "bnb_8bit",
        "bnb": "bnb_8bit",
        "bnb_8bit": "bnb_8bit",
    }
    if mode not in aliases:
        raise ValueError(
            f"unsupported quant_mode: {raw!r}. "
            "supported: none, bnb_8bit (aliases: hf_int8, int8, bnb)"
        )
    return aliases[mode]


def _load_bnb_8bit_model(model_dir: str, device: torch.device, dtype: torch.dtype):
    if device.type != "cuda":
        raise ValueError("bnb_8bit quantization requires CUDA device")
    try:
        from transformers import BitsAndBytesConfig
    except Exception as exc:
        raise ModuleNotFoundError(
            "bitsandbytes is required for bnb_8bit quantization."
        ) from exc

    quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
    m = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        quantization_config=quant_cfg,
        device_map={"": str(device)},
    )
    m.eval()
    return m


def load_front_model(
    front_dir: str,
    device: torch.device,
    dtype: torch.dtype,
    quant_mode: str = "none",
):
    mode = normalize_quant_mode(quant_mode)
    if mode == "bnb_8bit":
        m = _load_bnb_8bit_model(front_dir, device, dtype)
        try:
            m.model.norm = nn.Identity()
            m.lm_head = nn.Identity()
        except Exception:
            pass
        m.eval()
        return m

    cfg = AutoConfig.from_pretrained(front_dir)
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(cfg)

    try:
        m.model.norm = nn.Identity()
        m.lm_head = nn.Identity()
    except Exception:
        pass

    m = load_checkpoint_and_dispatch(
        m,
        checkpoint=front_dir,
        device_map={"": str(device)},
        offload_folder=None,
        dtype=dtype,
    )
    m.eval()
    return m


def load_back_model(
    back_dir: str,
    device: torch.device,
    dtype: torch.dtype,
    quant_mode: str = "none",
):
    mode = normalize_quant_mode(quant_mode)
    if mode == "bnb_8bit":
        m = _load_bnb_8bit_model(back_dir, device, dtype)
        try:
            m.model.embed_tokens = nn.Identity()
        except Exception:
            pass
        m.eval()
        return m

    cfg = AutoConfig.from_pretrained(back_dir)
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(cfg)

    try:
        m.model.embed_tokens = nn.Identity()
    except Exception:
        pass

    m = load_checkpoint_and_dispatch(
        m,
        checkpoint=back_dir,
        device_map={"": str(device)},
        offload_folder=None,
        dtype=dtype,
    )
    m.eval()
    return m


@dataclass
class SamplingConfig:
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    min_new_tokens: int = 0
    no_repeat_ngram_size: int = 0
    repetition_penalty: float = 1.0
    bad_words_ids: Optional[list[list[int]]] = None

    def validate(self) -> None:
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if not (0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.min_new_tokens < 0:
            raise ValueError(
                f"min_new_tokens must be >= 0, got {self.min_new_tokens}"
            )
        if self.no_repeat_ngram_size < 0:
            raise ValueError(
                "no_repeat_ngram_size must be >= 0, "
                f"got {self.no_repeat_ngram_size}"
            )
        if self.repetition_penalty <= 0:
            raise ValueError(
                f"repetition_penalty must be > 0, got {self.repetition_penalty}"
            )


@dataclass
class RuntimeGenerateResult:
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    finish_reason: str
    ttft_ms: float
    decode_step_ms: list[float] = field(default_factory=list)
    per_token_rtt_ms: list[float] = field(default_factory=list)
    prefill_rtt_ms: float = 0.0
    decode_rtt_ms: list[float] = field(default_factory=list)
    prefill_server_ms: float = 0.0
    decode_server_ms: list[float] = field(default_factory=list)
    server_ms: list[float] = field(default_factory=list)
    prefill_codec_encode_ms: float = 0.0
    prefill_codec_decode_ms: float = 0.0
    prefill_codec_wire_bytes: int = 0
    decode_codec_encode_ms: list[float] = field(default_factory=list)
    decode_codec_decode_ms: list[float] = field(default_factory=list)
    decode_codec_wire_bytes: list[int] = field(default_factory=list)
    total_ms: float = 0.0


def build_logits_processors(
    *,
    prompt_length: int,
    eos_token_id: int | None,
    cfg: SamplingConfig,
) -> LogitsProcessorList:
    processors = LogitsProcessorList()

    if cfg.min_new_tokens > 0 and eos_token_id is not None:
        min_length = prompt_length + cfg.min_new_tokens
        processors.append(
            MinLengthLogitsProcessor(min_length=min_length, eos_token_id=eos_token_id)
        )

    if cfg.no_repeat_ngram_size > 0:
        processors.append(NoRepeatNGramLogitsProcessor(cfg.no_repeat_ngram_size))

    if cfg.repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(cfg.repetition_penalty))

    if cfg.bad_words_ids:
        processors.append(
            NoBadWordsLogitsProcessor(
                bad_words_ids=cfg.bad_words_ids,
                eos_token_id=eos_token_id,
            )
        )

    return processors


def build_logits_warpers(cfg: SamplingConfig) -> LogitsProcessorList:
    warpers = LogitsProcessorList()

    if not cfg.do_sample:
        return warpers

    if cfg.temperature != 1.0:
        warpers.append(TemperatureLogitsWarper(cfg.temperature))

    if cfg.top_k > 0:
        warpers.append(TopKLogitsWarper(cfg.top_k))

    if cfg.top_p < 1.0:
        warpers.append(TopPLogitsWarper(cfg.top_p))

    return warpers


def select_next_token(
    *,
    input_ids: torch.Tensor,
    logits: torch.Tensor,
    processors: LogitsProcessorList,
    warpers: LogitsProcessorList,
    do_sample: bool,
) -> torch.Tensor:
    scores = logits
    if len(processors) > 0:
        scores = processors(input_ids, scores)

    if do_sample and len(warpers) > 0:
        scores = warpers(input_ids, scores)

    if do_sample:
        probs = torch.softmax(scores.float(), dim=-1)
        if not torch.isfinite(probs).all():
            return torch.argmax(scores, dim=-1, keepdim=True)
        return torch.multinomial(probs, num_samples=1)

    return torch.argmax(scores, dim=-1, keepdim=True)


def encoded_to_payload(codec: ActivationCodec, encoded: EncodedActivation) -> dict[str, Any]:
    wire = codec.pack(encoded)
    if "b64" not in wire:
        raise ValueError("codec.pack() must return dict with 'b64'")
    payload: dict[str, Any] = {
        "b64": wire["b64"],
        "meta": dict(wire.get("meta", {})),
    }
    if "shape" in wire:
        payload["shape"] = wire["shape"]
    if "dtype" in wire:
        payload["dtype"] = wire["dtype"]
    return payload


__all__ = [
    "SamplingConfig",
    "RuntimeGenerateResult",
    "pick_device_and_dtype",
    "resolve_dir",
    "normalize_quant_mode",
    "load_front_model",
    "load_back_model",
    "build_logits_processors",
    "build_logits_warpers",
    "select_next_token",
    "encoded_to_payload",
]
