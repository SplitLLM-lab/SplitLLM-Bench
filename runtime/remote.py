from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import requests
import torch
from transformers.cache_utils import DynamicCache

from model.codec import ActivationCodec, CodecContext, ensure_codec
from .common import (
    RuntimeGenerateResult,
    SamplingConfig,
    encoded_to_payload,
    load_front_model,
    pick_device_and_dtype,
    resolve_dir,
)


@dataclass
class RemoteTokenResponse:
    session_id: str
    next_token_id: int
    seq_len: int
    server_ms: float
    rtt_ms: float
    codec_encode_ms: float = 0.0
    codec_decode_ms: float = 0.0
    wire_bytes: int = 0


def payload_size_bytes(payload: dict[str, Any]) -> int:
    wire_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return len(wire_json.encode("utf-8"))


class RemoteBackClient:
    def __init__(
        self,
        server_url: str,
        timeout_sec: float = 120.0,
        codec: ActivationCodec | None = None,
    ):
        self.server_url = server_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.codec = ensure_codec(codec)

    def health(self) -> dict[str, Any]:
        r = requests.get(f"{self.server_url}/health", timeout=self.timeout_sec)
        r.raise_for_status()
        return r.json()

    def reset(self, session_id: str) -> None:
        requests.post(
            f"{self.server_url}/reset",
            json={"session_id": session_id},
            timeout=self.timeout_sec,
        )

    def prefill(
        self,
        *,
        session_id: str,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_token_ids: list[int],
        sampling: SamplingConfig,
        eos_token_id: int | None,
        codec_extras: Optional[dict[str, Any]] = None,
    ) -> RemoteTokenResponse:
        context = CodecContext(
            phase="prefill",
            side="edge_encode",
            session_id=session_id,
            seq_len=int(hidden.shape[1]),
            step=0,
            extras=dict(codec_extras or {}),
        )
        t_codec0 = time.perf_counter()
        encoded = self.codec.encode(hidden, context=context)
        codec_encode_ms = (time.perf_counter() - t_codec0) * 1000.0
        hidden_payload = encoded_to_payload(self.codec, encoded)
        payload = {
            "session_id": session_id,
            "hidden": hidden_payload,
            "attention_mask": attention_mask[0].detach().to(torch.long).cpu().tolist(),
            "prompt_token_ids": prompt_token_ids,
            "codec_extras": context.extras,
            "generation": {
                "do_sample": sampling.do_sample,
                "temperature": float(sampling.temperature),
                "top_k": int(sampling.top_k),
                "top_p": float(sampling.top_p),
                "min_new_tokens": int(sampling.min_new_tokens),
                "no_repeat_ngram_size": int(sampling.no_repeat_ngram_size),
                "repetition_penalty": float(sampling.repetition_penalty),
                "bad_words_ids": sampling.bad_words_ids,
                "eos_token_id": eos_token_id,
            },
        }
        wire_bytes = payload_size_bytes(hidden_payload)

        t0 = time.perf_counter()
        r = requests.post(
            f"{self.server_url}/prefill",
            json=payload,
            timeout=self.timeout_sec,
        )
        r.raise_for_status()
        resp = r.json()
        t1 = time.perf_counter()

        return RemoteTokenResponse(
            session_id=str(resp["session_id"]),
            next_token_id=int(resp["next_token_id"]),
            seq_len=int(resp["seq_len"]),
            server_ms=float(resp["server_ms"]),
            rtt_ms=(t1 - t0) * 1000.0,
            codec_encode_ms=float(codec_encode_ms),
            codec_decode_ms=float(resp.get("codec_decode_ms", 0.0)),
            wire_bytes=int(wire_bytes),
        )

    def decode(
        self,
        *,
        session_id: str,
        hidden_last: torch.Tensor,
        seq_len: int,
        token_step: int,
        codec_extras: Optional[dict[str, Any]] = None,
    ) -> RemoteTokenResponse:
        context = CodecContext(
            phase="decode",
            side="edge_encode",
            session_id=session_id,
            seq_len=seq_len,
            step=token_step,
            extras=dict(codec_extras or {}),
        )
        t_codec0 = time.perf_counter()
        encoded = self.codec.encode(hidden_last, context=context)
        codec_encode_ms = (time.perf_counter() - t_codec0) * 1000.0
        hidden_payload = encoded_to_payload(self.codec, encoded)
        payload = {
            "session_id": session_id,
            "hidden_last": hidden_payload,
            "seq_len": int(seq_len),
            "token_step": int(token_step),
            "codec_extras": context.extras,
        }
        wire_bytes = payload_size_bytes(hidden_payload)

        t0 = time.perf_counter()
        r = requests.post(
            f"{self.server_url}/decode",
            json=payload,
            timeout=self.timeout_sec,
        )
        r.raise_for_status()
        resp = r.json()
        t1 = time.perf_counter()

        return RemoteTokenResponse(
            session_id=str(resp["session_id"]),
            next_token_id=int(resp["next_token_id"]),
            seq_len=int(resp["seq_len"]),
            server_ms=float(resp["server_ms"]),
            rtt_ms=(t1 - t0) * 1000.0,
            codec_encode_ms=float(codec_encode_ms),
            codec_decode_ms=float(resp.get("codec_decode_ms", 0.0)),
            wire_bytes=int(wire_bytes),
        )


class RemoteSplitRuntime:
    def __init__(
        self,
        *,
        front_dir: str,
        server_url: str,
        device: str | None = "auto",
        dtype: str = "auto",
        timeout_sec: float = 120.0,
        revision: str | None = None,
        codec: ActivationCodec | None = None,
        front_quant: str = "none",
    ):
        self.device, self.dtype = pick_device_and_dtype(device, dtype)
        self.codec = ensure_codec(codec)
        front_local = resolve_dir(front_dir, revision=revision)
        self.client = RemoteBackClient(
            server_url=server_url,
            timeout_sec=timeout_sec,
            codec=self.codec,
        )

        print(
            f"[info] loading remote_front front={front_local} "
            f"device={self.device} dtype={self.dtype} codec={self.codec.name} "
            f"front_quant={front_quant}"
        )
        self.front = load_front_model(
            front_local,
            self.device,
            self.dtype,
            quant_mode=front_quant,
        )
        print(f"[ok] remote_front runtime ready (server={server_url.rstrip('/')})")

    @torch.no_grad()
    def generate_from_ids(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None,
        stop_token_ids: set[int],
        sampling: SamplingConfig,
        codec_extras: Optional[dict[str, Any]] = None,
    ) -> RuntimeGenerateResult:
        sampling.validate()
        if max_new_tokens <= 0:
            return RuntimeGenerateResult(
                prompt_token_ids=input_ids[0].tolist(),
                generated_token_ids=[],
                finish_reason="length",
                ttft_ms=0.0,
                total_ms=0.0,
            )

        t_total0 = time.perf_counter()
        prompt_len = int(input_ids.shape[1])

        front_cache = DynamicCache()
        cache_position = torch.arange(0, prompt_len, device=self.device)

        t_prefill0 = time.perf_counter()
        out_front = self.front.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=front_cache,
            use_cache=True,
            return_dict=True,
        )
        hidden = out_front.last_hidden_state

        session_id = str(uuid.uuid4())
        codec_extras = dict(codec_extras or {})
        generated_ids: list[int] = []
        ttft_ms = 0.0
        total_ms = 0.0
        decode_step_ms: list[float] = []
        decode_rtt_ms: list[float] = []
        decode_server_ms: list[float] = []
        decode_codec_encode_ms: list[float] = []
        decode_codec_decode_ms: list[float] = []
        decode_codec_wire_bytes: list[int] = []
        prefill_rtt_ms = 0.0
        prefill_server_ms = 0.0
        prefill_codec_encode_ms = 0.0
        prefill_codec_decode_ms = 0.0
        prefill_codec_wire_bytes = 0
        finish_reason = "length"

        try:
            self.codec.start_session(session_id)
            prefill_resp = self.client.prefill(
                session_id=session_id,
                hidden=hidden,
                attention_mask=attention_mask,
                prompt_token_ids=input_ids[0].detach().cpu().tolist(),
                sampling=sampling,
                eos_token_id=eos_token_id,
                codec_extras=codec_extras,
            )
            t_prefill1 = time.perf_counter()
            session_id = prefill_resp.session_id

            next_token_id = int(prefill_resp.next_token_id)
            ttft_ms = (t_prefill1 - t_prefill0) * 1000.0
            prefill_rtt_ms = prefill_resp.rtt_ms
            prefill_server_ms = prefill_resp.server_ms
            prefill_codec_encode_ms = prefill_resp.codec_encode_ms
            prefill_codec_decode_ms = prefill_resp.codec_decode_ms
            prefill_codec_wire_bytes = prefill_resp.wire_bytes
            generated_ids.append(next_token_id)

            if next_token_id in stop_token_ids:
                finish_reason = "stop"

            seq_len = prompt_len
            next_token = torch.tensor([[next_token_id]], device=self.device)

            for _ in range(max(0, max_new_tokens - 1)):
                if finish_reason == "stop":
                    break

                t_step0 = time.perf_counter()
                cache_position = cache_position[-1:] + 1
                seq_len += 1

                out_front = self.front.model(
                    input_ids=next_token,
                    cache_position=cache_position,
                    past_key_values=front_cache,
                    use_cache=True,
                    return_dict=True,
                )
                hidden_last = out_front.last_hidden_state

                decode_resp = self.client.decode(
                    session_id=session_id,
                    hidden_last=hidden_last,
                    seq_len=seq_len,
                    token_step=len(generated_ids),
                    codec_extras=codec_extras,
                )
                next_token_id = int(decode_resp.next_token_id)
                next_token = torch.tensor([[next_token_id]], device=self.device)
                t_step1 = time.perf_counter()

                generated_ids.append(next_token_id)
                decode_step_ms.append((t_step1 - t_step0) * 1000.0)
                decode_rtt_ms.append(decode_resp.rtt_ms)
                decode_server_ms.append(decode_resp.server_ms)
                decode_codec_encode_ms.append(decode_resp.codec_encode_ms)
                decode_codec_decode_ms.append(decode_resp.codec_decode_ms)
                decode_codec_wire_bytes.append(decode_resp.wire_bytes)

                if next_token_id in stop_token_ids:
                    finish_reason = "stop"

            total_ms = (time.perf_counter() - t_total0) * 1000.0

        finally:
            if session_id:
                try:
                    self.client.reset(session_id)
                except Exception:
                    pass
                try:
                    self.codec.end_session(session_id)
                except Exception:
                    pass

        return RuntimeGenerateResult(
            prompt_token_ids=input_ids[0].detach().cpu().tolist(),
            generated_token_ids=generated_ids,
            finish_reason=finish_reason,
            ttft_ms=ttft_ms,
            decode_step_ms=decode_step_ms,
            per_token_rtt_ms=decode_rtt_ms,
            prefill_rtt_ms=prefill_rtt_ms,
            decode_rtt_ms=decode_rtt_ms,
            prefill_server_ms=prefill_server_ms,
            decode_server_ms=decode_server_ms,
            server_ms=[prefill_server_ms] + decode_server_ms,
            prefill_codec_encode_ms=prefill_codec_encode_ms,
            prefill_codec_decode_ms=prefill_codec_decode_ms,
            prefill_codec_wire_bytes=prefill_codec_wire_bytes,
            decode_codec_encode_ms=decode_codec_encode_ms,
            decode_codec_decode_ms=decode_codec_decode_ms,
            decode_codec_wire_bytes=decode_codec_wire_bytes,
            total_ms=total_ms,
        )


__all__ = [
    "RemoteTokenResponse",
    "RemoteBackClient",
    "RemoteSplitRuntime",
]
