"""SplitLLM model package."""

from __future__ import annotations

from typing import Any

from .codec import (
    ActivationCodec,
    CodecContext,
    DefaultCodec,
    EncodedActivation,
    FunctionalActivationCodec,
    IdentityActivationCodec,
)

_MODEL_EXPORTS = {
    "SplitLLMModel",
    "GenerateResult",
}

_RUNTIME_EXPORTS = {
    "SamplingConfig",
    "RuntimeGenerateResult",
    "LocalSplitRuntime",
    "RemoteBackClient",
    "RemoteTokenResponse",
    "RemoteSplitRuntime",
    "RemoteBackServer",
}

__all__ = [
    "SplitLLMModel",
    "GenerateResult",
    "ActivationCodec",
    "DefaultCodec",
    "IdentityActivationCodec",
    "FunctionalActivationCodec",
    "CodecContext",
    "EncodedActivation",
    "SamplingConfig",
    "RuntimeGenerateResult",
    "LocalSplitRuntime",
    "RemoteBackClient",
    "RemoteTokenResponse",
    "RemoteSplitRuntime",
    "RemoteBackServer",
]


def __getattr__(name: str) -> Any:
    if name in _MODEL_EXPORTS:
        from .model import GenerateResult, SplitLLMModel

        model_exports: dict[str, Any] = {
            "SplitLLMModel": SplitLLMModel,
            "GenerateResult": GenerateResult,
        }
        return model_exports[name]

    if name in _RUNTIME_EXPORTS:
        from runtime import (
            LocalSplitRuntime,
            RemoteBackClient,
            RemoteBackServer,
            RemoteSplitRuntime,
            RemoteTokenResponse,
            RuntimeGenerateResult,
            SamplingConfig,
        )

        runtime_exports: dict[str, Any] = {
            "SamplingConfig": SamplingConfig,
            "RuntimeGenerateResult": RuntimeGenerateResult,
            "LocalSplitRuntime": LocalSplitRuntime,
            "RemoteBackClient": RemoteBackClient,
            "RemoteTokenResponse": RemoteTokenResponse,
            "RemoteSplitRuntime": RemoteSplitRuntime,
            "RemoteBackServer": RemoteBackServer,
        }
        return runtime_exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
