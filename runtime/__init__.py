from __future__ import annotations

"""Compatibility facade for split runtime components."""

from typing import Any

_COMMON_EXPORTS = {
    "SamplingConfig",
    "RuntimeGenerateResult",
    "pick_device_and_dtype",
    "resolve_dir",
    "load_front_model",
    "load_back_model",
    "build_logits_processors",
    "build_logits_warpers",
    "select_next_token",
    "encoded_to_payload",
}

_LOCAL_EXPORTS = {"LocalSplitRuntime"}
_REMOTE_EXPORTS = {"RemoteBackClient", "RemoteTokenResponse", "RemoteSplitRuntime"}
_SERVER_EXPORTS = {"RemoteBackServer"}

__all__ = [
    "SamplingConfig",
    "RuntimeGenerateResult",
    "pick_device_and_dtype",
    "resolve_dir",
    "load_front_model",
    "load_back_model",
    "build_logits_processors",
    "build_logits_warpers",
    "select_next_token",
    "encoded_to_payload",
    "LocalSplitRuntime",
    "RemoteBackClient",
    "RemoteTokenResponse",
    "RemoteSplitRuntime",
    "RemoteBackServer",
]


def __getattr__(name: str) -> Any:
    if name in _COMMON_EXPORTS:
        from .common import (
            RuntimeGenerateResult,
            SamplingConfig,
            build_logits_processors,
            build_logits_warpers,
            encoded_to_payload,
            load_back_model,
            load_front_model,
            pick_device_and_dtype,
            resolve_dir,
            select_next_token,
        )

        common_exports: dict[str, Any] = {
            "SamplingConfig": SamplingConfig,
            "RuntimeGenerateResult": RuntimeGenerateResult,
            "pick_device_and_dtype": pick_device_and_dtype,
            "resolve_dir": resolve_dir,
            "load_front_model": load_front_model,
            "load_back_model": load_back_model,
            "build_logits_processors": build_logits_processors,
            "build_logits_warpers": build_logits_warpers,
            "select_next_token": select_next_token,
            "encoded_to_payload": encoded_to_payload,
        }
        return common_exports[name]

    if name in _LOCAL_EXPORTS:
        from .local import LocalSplitRuntime

        return LocalSplitRuntime

    if name in _REMOTE_EXPORTS:
        from .remote import RemoteBackClient, RemoteSplitRuntime, RemoteTokenResponse

        remote_exports: dict[str, Any] = {
            "RemoteBackClient": RemoteBackClient,
            "RemoteTokenResponse": RemoteTokenResponse,
            "RemoteSplitRuntime": RemoteSplitRuntime,
        }
        return remote_exports[name]

    if name in _SERVER_EXPORTS:
        from .server import RemoteBackServer

        return RemoteBackServer

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
