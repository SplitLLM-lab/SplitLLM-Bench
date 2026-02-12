from __future__ import annotations

import importlib
import inspect
import json
import math
import time
from dataclasses import dataclass
from typing import Any

import torch
from datasets import load_dataset

from model import DefaultCodec, IdentityActivationCodec
from model.codec import ActivationCodec, CodecContext, EncodedActivation


@dataclass
class CodecTiming:
    encode_ms: float = 0.0
    decode_ms: float = 0.0


def parse_codec_extras(raw: str | None) -> dict[str, Any]:
    if raw is None:
        return {}
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("--codec_extras_json must decode to JSON object")
    return dict(obj)


def build_codec(name: str) -> ActivationCodec:
    builtin: dict[str, ActivationCodec] = {
        "default": DefaultCodec(),
        "identity_fp32": IdentityActivationCodec(),
    }
    if name in builtin:
        return builtin[name]
    return _load_custom_codec(name)


def _codec_from_object(obj: Any, spec: str) -> ActivationCodec:
    if isinstance(obj, ActivationCodec):
        return obj

    if inspect.isclass(obj):
        if not issubclass(obj, ActivationCodec):
            raise TypeError(
                f"{spec!r} resolved to class {obj.__name__}, "
                "but it is not a subclass of ActivationCodec"
            )
        return obj()

    if callable(obj):
        built = obj()
        if not isinstance(built, ActivationCodec):
            raise TypeError(
                f"{spec!r} callable returned {type(built).__name__}, "
                "expected ActivationCodec"
            )
        return built

    raise TypeError(
        f"{spec!r} resolved to unsupported object type: {type(obj).__name__}. "
        "Expected ActivationCodec instance/class or callable returning one."
    )


def _codec_from_module(module: Any, spec: str) -> ActivationCodec:
    if hasattr(module, "build_codec"):
        return _codec_from_object(getattr(module, "build_codec"), f"{spec}:build_codec")

    for attr in ("codec", "CODEC"):
        if hasattr(module, attr):
            return _codec_from_object(getattr(module, attr), f"{spec}:{attr}")

    classes = [
        v
        for v in vars(module).values()
        if inspect.isclass(v) and issubclass(v, ActivationCodec) and v is not ActivationCodec
    ]
    if len(classes) == 1:
        return classes[0]()

    raise ValueError(
        f"cannot build codec from module {spec!r}. "
        "Expected one of: build_codec(), codec, CODEC, or exactly one "
        "ActivationCodec subclass."
    )


def _load_custom_codec(spec: str) -> ActivationCodec:
    if ":" in spec:
        module_name, attr = spec.split(":", 1)
        module_name = module_name.strip()
        attr = attr.strip()
        if not module_name or not attr:
            raise ValueError(
                f"invalid codec spec {spec!r}; expected format module:attr"
            )
        module = importlib.import_module(module_name)
        if not hasattr(module, attr):
            raise ValueError(
                f"module {module_name!r} has no attribute {attr!r} for codec spec {spec!r}"
            )
        return _codec_from_object(getattr(module, attr), spec)

    try:
        module = importlib.import_module(spec)
    except ModuleNotFoundError as exc:
        if exc.name != spec:
            raise
        if "." not in spec:
            raise ValueError(
                f"unsupported codec: {spec!r}. "
                "Use builtins [default, identity_fp32] or custom module path."
            )
        module_name, attr = spec.rsplit(".", 1)
        module = importlib.import_module(module_name)
        if not hasattr(module, attr):
            raise ValueError(
                f"module {module_name!r} has no attribute {attr!r} for codec spec {spec!r}"
            )
        return _codec_from_object(getattr(module, attr), spec)

    return _codec_from_module(module, spec)


def load_texts(
    *,
    dataset_name: str,
    dataset_config: str,
    split: str,
    text_column: str,
    samples: int,
) -> list[str]:
    ds = load_dataset(dataset_name, dataset_config, split=split)
    if text_column not in ds.column_names:
        cols = ", ".join(ds.column_names)
        raise ValueError(f"text_column {text_column!r} not in dataset columns: {cols}")

    ds = ds.filter(
        lambda ex: ex.get(text_column) is not None and len(str(ex[text_column]).strip()) > 0
    )
    n = min(int(samples), len(ds))
    return [str(x) for x in ds.select(range(n))[text_column]]


def safe_exp(x: float) -> float:
    if x >= 80:
        return float("inf")
    return float(math.exp(x))


def apply_codec(
    *,
    codec: ActivationCodec,
    hidden: torch.Tensor,
    context: CodecContext,
    device: torch.device,
    dtype: torch.dtype,
    timing: CodecTiming | None = None,
) -> torch.Tensor:
    t0 = time.perf_counter()
    encoded: EncodedActivation = codec.encode(hidden, context=context)
    if timing is not None:
        timing.encode_ms += (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    decoded = codec.decode(
        encoded,
        context=context,
        device=device,
        dtype=dtype,
    )
    if timing is not None:
        timing.decode_ms += (time.perf_counter() - t0) * 1000.0
    return decoded


__all__ = [
    "CodecTiming",
    "parse_codec_extras",
    "build_codec",
    "load_texts",
    "safe_exp",
    "apply_codec",
]
