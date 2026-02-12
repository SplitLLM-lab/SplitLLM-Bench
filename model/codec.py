from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import torch


@dataclass
class EncodedActivation:
    data: Any
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class CodecContext:
    phase: str
    side: str
    session_id: str | None
    seq_len: int
    step: int
    extras: dict[str, Any] = field(default_factory=dict)


class ActivationCodec:
    name: str = "activation_codec"

    def start_session(self, session_id: str) -> None:
        del session_id

    def end_session(self, session_id: str) -> None:
        del session_id

    def encode(
        self,
        hidden: torch.Tensor,
        *,
        context: CodecContext,
    ) -> EncodedActivation:
        raise NotImplementedError

    def decode(
        self,
        payload: EncodedActivation,
        *,
        context: CodecContext,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        raise NotImplementedError

    def pack(self, encoded: EncodedActivation) -> dict[str, Any]:
        data = encoded.data
        meta = dict(encoded.meta or {})

        if isinstance(data, str):
            return {"b64": data, "meta": meta}

        if isinstance(data, (bytes, bytearray)):
            return {
                "b64": base64.b64encode(bytes(data)).decode("ascii"),
                "meta": {
                    **meta,
                    "shape": [len(data)],
                    "dtype": "uint8",
                    "_wire_type": "bytes",
                },
            }

        if isinstance(data, np.ndarray):
            return {
                "b64": base64.b64encode(data.tobytes()).decode("ascii"),
                "meta": {
                    **meta,
                    "shape": list(data.shape),
                    "dtype": str(data.dtype),
                    "_wire_type": "ndarray",
                },
            }

        if isinstance(data, torch.Tensor):
            x = data.detach().cpu()
            if x.dtype == torch.bfloat16:
                x = x.to(torch.float32)
                meta = {**meta, "_source_dtype": "bfloat16"}
            arr = x.numpy()
            return {
                "b64": base64.b64encode(arr.tobytes()).decode("ascii"),
                "meta": {
                    **meta,
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "_wire_type": "tensor",
                },
            }

        raise TypeError(
            "unsupported encoded.data type for wire payload: "
            f"{type(data).__name__}. "
            "Use str/bytes/np.ndarray/torch.Tensor or override pack()."
        )

    def unpack(self, b64: str, meta: Optional[dict[str, Any]] = None) -> EncodedActivation:
        return EncodedActivation(data=b64, meta=dict(meta or {}))


class DefaultCodec(ActivationCodec):
    name = "no_transport"

    def encode(
        self,
        hidden: torch.Tensor,
        *,
        context: CodecContext,
    ) -> EncodedActivation:
        del context
        return EncodedActivation(data=hidden, meta={})

    def decode(
        self,
        payload: EncodedActivation,
        *,
        context: CodecContext,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        del context
        data = payload.data

        if isinstance(data, torch.Tensor):
            hidden = data
        elif isinstance(data, np.ndarray):
            hidden = torch.from_numpy(data)
        elif isinstance(data, str):
            shape = payload.meta.get("shape")
            np_dtype = payload.meta.get("dtype")
            if shape is None or np_dtype is None:
                raise ValueError("string payload requires meta['shape'] and meta['dtype']")
            raw = base64.b64decode(data.encode("ascii"))
            arr = np.frombuffer(raw, dtype=np.dtype(np_dtype)).reshape(tuple(shape))
            hidden = torch.from_numpy(arr)
        else:
            raise TypeError(
                "DefaultCodec.decode got unsupported payload type: "
                f"{type(data).__name__}"
            )

        if hidden.device != device:
            hidden = hidden.to(device)
        if hidden.dtype != dtype and hidden.is_floating_point():
            hidden = hidden.to(dtype)
        return hidden


class IdentityActivationCodec(ActivationCodec):
    name = "identity_fp32"

    def encode(
        self,
        hidden: torch.Tensor,
        *,
        context: CodecContext,
    ) -> EncodedActivation:
        del context
        x = hidden.detach().to(torch.float32).cpu().numpy()
        return EncodedActivation(
            data=base64.b64encode(x.tobytes()).decode("ascii"),
            meta={
                "shape": list(x.shape),
                "dtype": str(x.dtype),
            },
        )

    def decode(
        self,
        payload: EncodedActivation,
        *,
        context: CodecContext,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        del context
        if not isinstance(payload.data, str):
            raise TypeError("IdentityActivationCodec expects string payload data")
        shape = payload.meta.get("shape")
        np_dtype = payload.meta.get("dtype", "float32")
        if shape is None:
            raise ValueError("IdentityActivationCodec requires meta['shape']")
        raw = base64.b64decode(payload.data.encode("ascii"))
        arr = np.frombuffer(raw, dtype=np.dtype(np_dtype)).reshape(tuple(shape))
        hidden = torch.from_numpy(arr).to(device=device)
        if hidden.dtype != dtype and hidden.is_floating_point():
            hidden = hidden.to(dtype)
        return hidden


class FunctionalActivationCodec(ActivationCodec):
    def __init__(
        self,
        *,
        encode_fn: Callable[[torch.Tensor, CodecContext], EncodedActivation | dict[str, Any]],
        decode_fn: Callable[
            [EncodedActivation, CodecContext, torch.device, torch.dtype],
            torch.Tensor | np.ndarray,
        ],
        name: str = "functional_codec",
        on_start_session: Optional[Callable[[str], None]] = None,
        on_end_session: Optional[Callable[[str], None]] = None,
        pack_fn: Optional[Callable[[EncodedActivation], dict[str, Any]]] = None,
        unpack_fn: Optional[Callable[[str, dict[str, Any]], EncodedActivation]] = None,
    ):
        self.name = name
        self._encode_fn = encode_fn
        self._decode_fn = decode_fn
        self._on_start_session = on_start_session
        self._on_end_session = on_end_session
        self._pack_fn = pack_fn
        self._unpack_fn = unpack_fn

    def start_session(self, session_id: str) -> None:
        if self._on_start_session is not None:
            self._on_start_session(session_id)

    def end_session(self, session_id: str) -> None:
        if self._on_end_session is not None:
            self._on_end_session(session_id)

    def encode(
        self,
        hidden: torch.Tensor,
        *,
        context: CodecContext,
    ) -> EncodedActivation:
        encoded = self._encode_fn(hidden, context)
        if isinstance(encoded, EncodedActivation):
            return encoded
        if isinstance(encoded, dict):
            if "data" in encoded:
                return EncodedActivation(
                    data=encoded["data"],
                    meta=dict(encoded.get("meta", {})),
                )
            if "b64" in encoded:
                return EncodedActivation(
                    data=encoded["b64"],
                    meta=dict(encoded.get("meta", {})),
                )
        raise TypeError(
            "encode_fn must return EncodedActivation or dict with "
            "'data'/'meta' (or legacy 'b64'/'meta')."
        )

    def decode(
        self,
        payload: EncodedActivation,
        *,
        context: CodecContext,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        hidden = self._decode_fn(payload, context, device, dtype)
        if isinstance(hidden, np.ndarray):
            hidden = torch.from_numpy(hidden)
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("decode_fn must return torch.Tensor or np.ndarray")
        if hidden.device != device:
            hidden = hidden.to(device)
        if hidden.dtype != dtype and hidden.is_floating_point():
            hidden = hidden.to(dtype)
        return hidden

    def pack(self, encoded: EncodedActivation) -> dict[str, Any]:
        if self._pack_fn is not None:
            return self._pack_fn(encoded)
        return super().pack(encoded)

    def unpack(self, b64: str, meta: Optional[dict[str, Any]] = None) -> EncodedActivation:
        if self._unpack_fn is not None:
            return self._unpack_fn(b64, dict(meta or {}))
        return super().unpack(b64=b64, meta=meta)


def ensure_codec(codec: ActivationCodec | None) -> ActivationCodec:
    if codec is None:
        return DefaultCodec()
    return codec


__all__ = [
    "ActivationCodec",
    "DefaultCodec",
    "IdentityActivationCodec",
    "FunctionalActivationCodec",
    "CodecContext",
    "EncodedActivation",
    "ensure_codec",
]
