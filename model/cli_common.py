from __future__ import annotations

import importlib
import inspect
import json
from typing import Any

from .codec import ActivationCodec, DefaultCodec, IdentityActivationCodec


def parse_codec_extras(raw: str | None) -> dict[str, Any]:
    if raw is None:
        return {}
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("--codec_extras_json must decode to JSON object")
    return dict(obj)


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


def build_codec(spec: str) -> ActivationCodec:
    builtin: dict[str, ActivationCodec] = {
        "default": DefaultCodec(),
        "identity_fp32": IdentityActivationCodec(),
    }
    if spec in builtin:
        return builtin[spec]

    if ":" in spec:
        module_name, attr = spec.split(":", 1)
        module_name = module_name.strip()
        attr = attr.strip()
        if not module_name or not attr:
            raise ValueError(f"invalid codec spec {spec!r}; expected format module:attr")
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


__all__ = ["build_codec", "parse_codec_extras"]
