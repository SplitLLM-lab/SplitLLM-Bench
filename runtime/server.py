from __future__ import annotations

import argparse
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model.codec import (
    ActivationCodec,
    CodecContext,
    EncodedActivation,
    ensure_codec,
)
from .common import (
    SamplingConfig,
    build_logits_processors,
    build_logits_warpers,
    load_back_model,
    pick_device_and_dtype,
    resolve_dir,
    select_next_token,
)


class HiddenPayload(BaseModel):
    b64: str
    meta: dict[str, Any] = Field(default_factory=dict)
    shape: Optional[list[int]] = None
    dtype: Optional[str] = None


class GenerationPayload(BaseModel):
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    min_new_tokens: int = 0
    no_repeat_ngram_size: int = 0
    repetition_penalty: float = 1.0
    bad_words_ids: Optional[list[list[int]]] = None
    eos_token_id: Optional[int] = None

    def to_sampling_config(self) -> SamplingConfig:
        return SamplingConfig(
            do_sample=bool(self.do_sample),
            temperature=float(self.temperature),
            top_k=int(self.top_k),
            top_p=float(self.top_p),
            min_new_tokens=int(self.min_new_tokens),
            no_repeat_ngram_size=int(self.no_repeat_ngram_size),
            repetition_penalty=float(self.repetition_penalty),
            bad_words_ids=self.bad_words_ids,
        )


class PrefillRequest(BaseModel):
    session_id: Optional[str] = None
    hidden: HiddenPayload
    attention_mask: list[int]
    prompt_token_ids: list[int]
    codec_extras: dict[str, Any] = Field(default_factory=dict)
    generation: GenerationPayload = Field(default_factory=GenerationPayload)


class DecodeRequest(BaseModel):
    session_id: str
    hidden_last: HiddenPayload
    seq_len: int
    token_step: int = 0
    codec_extras: dict[str, Any] = Field(default_factory=dict)


class SpecVerifyRequest(BaseModel):
    session_id: str
    hidden: HiddenPayload
    candidate_token_ids: list[int]
    token_step: int = 0
    codec_extras: dict[str, Any] = Field(default_factory=dict)


class SpecCommitRequest(BaseModel):
    session_id: str
    hidden_last: HiddenPayload
    token_id: int
    seq_len: int
    token_step: int = 0
    codec_extras: dict[str, Any] = Field(default_factory=dict)


class ResetRequest(BaseModel):
    session_id: str


class TokenResponse(BaseModel):
    session_id: str
    next_token_id: int
    seq_len: int
    server_ms: float
    codec_decode_ms: float = 0.0


class SpecPrefillResponse(BaseModel):
    session_id: str
    seq_len: int
    server_ms: float
    codec_decode_ms: float = 0.0


class SpecVerifyResponse(BaseModel):
    session_id: str
    accepted_token_ids: list[int]
    fallback_token_id: Optional[int] = None
    seq_len: int
    server_ms: float
    codec_decode_ms: float = 0.0


@dataclass
class SessionState:
    past_key_values: Any
    seq_len: int
    token_ids: torch.Tensor
    processors: Any
    warpers: Any
    do_sample: bool
    last_touch: float


@dataclass
class SpecSessionState:
    past_key_values: Any
    seq_len: int
    token_ids: torch.Tensor
    next_logits: torch.Tensor
    processors: Any
    warpers: Any
    do_sample: bool
    last_touch: float


class RemoteBackServer:
    def __init__(
        self,
        *,
        back_dir: str,
        device: str | None = "auto",
        dtype: str = "auto",
        revision: str | None = None,
        session_ttl_sec: int = 1800,
        codec: ActivationCodec | None = None,
        back_quant: str = "none",
    ):
        self.device, self.dtype = pick_device_and_dtype(device, dtype)
        self.codec = ensure_codec(codec)
        self.back_dir = resolve_dir(back_dir, revision=revision)
        self.session_ttl_sec = session_ttl_sec
        self.sessions: dict[str, SessionState] = {}
        self.spec_sessions: dict[str, SpecSessionState] = {}

        print(
            f"[info] loading remote back={self.back_dir} "
            f"device={self.device} dtype={self.dtype} codec={self.codec.name} "
            f"back_quant={back_quant}"
        )
        self.back = load_back_model(
            self.back_dir,
            self.device,
            self.dtype,
            quant_mode=back_quant,
        )
        print("[ok] remote back runtime ready")

        self.app = FastAPI(title="SplitLLM Remote Back Server")
        self._register_routes()

    def _cleanup_sessions(self) -> None:
        now = time.time()
        dead = [
            sid
            for sid, st in self.sessions.items()
            if now - st.last_touch > self.session_ttl_sec
        ]
        for sid in dead:
            self.sessions.pop(sid, None)
            try:
                self.codec.end_session(sid)
            except Exception:
                pass
        dead_spec = [
            sid
            for sid, st in self.spec_sessions.items()
            if now - st.last_touch > self.session_ttl_sec
        ]
        for sid in dead_spec:
            self.spec_sessions.pop(sid, None)
            try:
                self.codec.end_session(sid)
            except Exception:
                pass

    def _payload_to_encoded(self, payload: HiddenPayload) -> EncodedActivation:
        meta = dict(payload.meta or {})
        if payload.shape is not None and "shape" not in meta:
            meta["shape"] = payload.shape
        if payload.dtype is not None and "dtype" not in meta:
            meta["dtype"] = payload.dtype
        return self.codec.unpack(b64=payload.b64, meta=meta)

    def _payload_to_hidden(
        self,
        payload: HiddenPayload,
        *,
        context: CodecContext,
    ) -> torch.Tensor:
        encoded = self._payload_to_encoded(payload)
        return self.codec.decode(
            encoded,
            context=context,
            device=self.device,
            dtype=self.dtype,
        )

    def _register_routes(self) -> None:
        app = self.app

        @app.get("/health")
        def health() -> dict[str, Any]:
            self._cleanup_sessions()
            return {
                "ok": True,
                "device": str(self.device),
                "dtype": str(self.dtype),
                "sessions": len(self.sessions) + len(self.spec_sessions),
            }

        @app.post("/reset")
        def reset(req: ResetRequest) -> dict[str, bool]:
            self.sessions.pop(req.session_id, None)
            self.spec_sessions.pop(req.session_id, None)
            try:
                self.codec.end_session(req.session_id)
            except Exception:
                pass
            return {"ok": True}

        @app.post("/prefill", response_model=TokenResponse)
        @torch.no_grad()
        def prefill(req: PrefillRequest) -> TokenResponse:
            self._cleanup_sessions()
            sid = req.session_id or str(uuid.uuid4())
            if sid in self.sessions:
                self.sessions.pop(sid, None)
                try:
                    self.codec.end_session(sid)
                except Exception:
                    pass
            self.codec.start_session(sid)
            session_registered = False
            try:
                prefill_ctx = CodecContext(
                    phase="prefill",
                    side="cloud_decode",
                    session_id=sid,
                    seq_len=len(req.prompt_token_ids),
                    step=0,
                    extras=dict(req.codec_extras or {}),
                )
                t_codec0 = time.perf_counter()
                hidden = self._payload_to_hidden(req.hidden, context=prefill_ctx)
                codec_decode_ms = (time.perf_counter() - t_codec0) * 1000.0
                if hidden.ndim != 3:
                    raise HTTPException(status_code=400, detail="hidden must be [B, T, H]")

                seq = int(hidden.shape[1])
                if len(req.attention_mask) != seq:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "attention_mask length mismatch: "
                            f"got {len(req.attention_mask)} expected {seq}"
                        ),
                    )
                if len(req.prompt_token_ids) != seq:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "prompt_token_ids length mismatch: "
                            f"got {len(req.prompt_token_ids)} expected {seq}"
                        ),
                    )

                sampling = req.generation.to_sampling_config()
                try:
                    sampling.validate()
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc

                processors = build_logits_processors(
                    prompt_length=seq,
                    eos_token_id=req.generation.eos_token_id,
                    cfg=sampling,
                )
                warpers = build_logits_warpers(sampling)

                attention_mask = torch.tensor(
                    [req.attention_mask], device=self.device, dtype=torch.long
                )
                cache_position = torch.arange(0, seq, device=self.device)

                t0 = time.perf_counter()
                out = self.back.model(
                    inputs_embeds=hidden,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    use_cache=True,
                    return_dict=True,
                )
                logits = self.back.lm_head(out.last_hidden_state)[:, -1, :]

                token_ids = torch.tensor(
                    [req.prompt_token_ids],
                    device=self.device,
                    dtype=torch.long,
                )
                next_token = select_next_token(
                    input_ids=token_ids,
                    logits=logits,
                    processors=processors,
                    warpers=warpers,
                    do_sample=sampling.do_sample,
                )
                next_token_id = int(next_token.item())

                self.sessions[sid] = SessionState(
                    past_key_values=out.past_key_values,
                    seq_len=seq,
                    token_ids=torch.cat([token_ids, next_token], dim=1),
                    processors=processors,
                    warpers=warpers,
                    do_sample=sampling.do_sample,
                    last_touch=time.time(),
                )
                session_registered = True

                t1 = time.perf_counter()
                return TokenResponse(
                    session_id=sid,
                    next_token_id=next_token_id,
                    seq_len=seq,
                    server_ms=(t1 - t0) * 1000.0,
                    codec_decode_ms=codec_decode_ms,
                )
            except HTTPException:
                if not session_registered:
                    try:
                        self.codec.end_session(sid)
                    except Exception:
                        pass
                raise
            except Exception as exc:
                if not session_registered:
                    try:
                        self.codec.end_session(sid)
                    except Exception:
                        pass
                raise HTTPException(
                    status_code=400,
                    detail=f"codec/server prefill failed: {exc}",
                ) from exc

        @app.post("/decode", response_model=TokenResponse)
        @torch.no_grad()
        def decode(req: DecodeRequest) -> TokenResponse:
            self._cleanup_sessions()
            st = self.sessions.get(req.session_id)
            if st is None:
                raise HTTPException(
                    status_code=404,
                    detail="unknown session_id; call /prefill first",
                )

            decode_ctx = CodecContext(
                phase="decode",
                side="cloud_decode",
                session_id=req.session_id,
                seq_len=req.seq_len,
                step=req.token_step,
                extras=dict(req.codec_extras or {}),
            )
            try:
                t_codec0 = time.perf_counter()
                hidden_last = self._payload_to_hidden(
                    req.hidden_last,
                    context=decode_ctx,
                )
                codec_decode_ms = (time.perf_counter() - t_codec0) * 1000.0
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"codec decode failed at decode step: {exc}",
                ) from exc
            if hidden_last.ndim != 3 or hidden_last.shape[1] != 1:
                raise HTTPException(
                    status_code=400,
                    detail="hidden_last must be [B, 1, H]",
                )

            expected = st.seq_len + 1
            if req.seq_len != expected:
                raise HTTPException(
                    status_code=400,
                    detail=f"seq_len mismatch: got {req.seq_len}, expected {expected}",
                )

            attention_mask = torch.ones(
                (1, req.seq_len),
                device=self.device,
                dtype=torch.long,
            )
            cache_position = torch.tensor([st.seq_len], device=self.device)

            t0 = time.perf_counter()
            out = self.back.model(
                inputs_embeds=hidden_last,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=st.past_key_values,
                use_cache=True,
                return_dict=True,
            )
            logits = self.back.lm_head(out.last_hidden_state)[:, -1, :]

            next_token = select_next_token(
                input_ids=st.token_ids,
                logits=logits,
                processors=st.processors,
                warpers=st.warpers,
                do_sample=st.do_sample,
            )
            next_token_id = int(next_token.item())

            st.past_key_values = out.past_key_values
            st.seq_len = req.seq_len
            st.token_ids = torch.cat([st.token_ids, next_token], dim=1)
            st.last_touch = time.time()

            t1 = time.perf_counter()
            return TokenResponse(
                session_id=req.session_id,
                next_token_id=next_token_id,
                seq_len=st.seq_len,
                server_ms=(t1 - t0) * 1000.0,
                codec_decode_ms=codec_decode_ms,
            )

        @app.post("/spec_prefill", response_model=SpecPrefillResponse)
        @torch.no_grad()
        def spec_prefill(req: PrefillRequest) -> SpecPrefillResponse:
            self._cleanup_sessions()
            sid = req.session_id or str(uuid.uuid4())
            self.sessions.pop(sid, None)
            old_spec = self.spec_sessions.pop(sid, None)
            if old_spec is not None:
                try:
                    self.codec.end_session(sid)
                except Exception:
                    pass
            self.codec.start_session(sid)
            session_registered = False
            try:
                prefill_ctx = CodecContext(
                    phase="prefill",
                    side="cloud_self_spec",
                    session_id=sid,
                    seq_len=len(req.prompt_token_ids),
                    step=0,
                    extras=dict(req.codec_extras or {}),
                )
                t_codec0 = time.perf_counter()
                hidden = self._payload_to_hidden(req.hidden, context=prefill_ctx)
                codec_decode_ms = (time.perf_counter() - t_codec0) * 1000.0
                if hidden.ndim != 3:
                    raise HTTPException(status_code=400, detail="hidden must be [B, T, H]")

                seq = int(hidden.shape[1])
                if len(req.attention_mask) != seq:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "attention_mask length mismatch: "
                            f"got {len(req.attention_mask)} expected {seq}"
                        ),
                    )
                if len(req.prompt_token_ids) != seq:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "prompt_token_ids length mismatch: "
                            f"got {len(req.prompt_token_ids)} expected {seq}"
                        ),
                    )

                sampling = req.generation.to_sampling_config()
                try:
                    sampling.validate()
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                if sampling.do_sample:
                    raise HTTPException(
                        status_code=400,
                        detail="self-speculative remote mode supports greedy decoding only",
                    )

                processors = build_logits_processors(
                    prompt_length=seq,
                    eos_token_id=req.generation.eos_token_id,
                    cfg=sampling,
                )
                warpers = build_logits_warpers(sampling)

                attention_mask = torch.tensor(
                    [req.attention_mask], device=self.device, dtype=torch.long
                )
                cache_position = torch.arange(0, seq, device=self.device)

                t0 = time.perf_counter()
                out = self.back.model(
                    inputs_embeds=hidden,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    use_cache=True,
                    return_dict=True,
                )
                next_logits = self.back.lm_head(out.last_hidden_state)[:, -1, :]

                token_ids = torch.tensor(
                    [req.prompt_token_ids],
                    device=self.device,
                    dtype=torch.long,
                )
                self.spec_sessions[sid] = SpecSessionState(
                    past_key_values=out.past_key_values,
                    seq_len=seq,
                    token_ids=token_ids,
                    next_logits=next_logits,
                    processors=processors,
                    warpers=warpers,
                    do_sample=sampling.do_sample,
                    last_touch=time.time(),
                )
                session_registered = True

                t1 = time.perf_counter()
                return SpecPrefillResponse(
                    session_id=sid,
                    seq_len=seq,
                    server_ms=(t1 - t0) * 1000.0,
                    codec_decode_ms=codec_decode_ms,
                )
            except HTTPException:
                if not session_registered:
                    try:
                        self.codec.end_session(sid)
                    except Exception:
                        pass
                raise
            except Exception as exc:
                if not session_registered:
                    try:
                        self.codec.end_session(sid)
                    except Exception:
                        pass
                raise HTTPException(
                    status_code=400,
                    detail=f"codec/server spec prefill failed: {exc}",
                ) from exc

        @app.post("/spec_verify", response_model=SpecVerifyResponse)
        @torch.no_grad()
        def spec_verify(req: SpecVerifyRequest) -> SpecVerifyResponse:
            self._cleanup_sessions()
            st = self.spec_sessions.get(req.session_id)
            if st is None:
                raise HTTPException(
                    status_code=404,
                    detail="unknown session_id; call /spec_prefill first",
                )
            if not req.candidate_token_ids:
                raise HTTPException(status_code=400, detail="candidate_token_ids is empty")

            t0 = time.perf_counter()
            first_target = select_next_token(
                input_ids=st.token_ids,
                logits=st.next_logits,
                processors=st.processors,
                warpers=st.warpers,
                do_sample=st.do_sample,
            )
            if int(first_target.item()) != int(req.candidate_token_ids[0]):
                st.last_touch = time.time()
                t1 = time.perf_counter()
                return SpecVerifyResponse(
                    session_id=req.session_id,
                    accepted_token_ids=[],
                    fallback_token_id=int(first_target.item()),
                    seq_len=st.seq_len,
                    server_ms=(t1 - t0) * 1000.0,
                    codec_decode_ms=0.0,
                )

            verify_ctx = CodecContext(
                phase="decode",
                side="cloud_self_spec",
                session_id=req.session_id,
                seq_len=st.seq_len + len(req.candidate_token_ids),
                step=req.token_step,
                extras=dict(req.codec_extras or {}),
            )
            try:
                t_codec0 = time.perf_counter()
                hidden = self._payload_to_hidden(req.hidden, context=verify_ctx)
                codec_decode_ms = (time.perf_counter() - t_codec0) * 1000.0
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"codec decode failed at spec verify: {exc}",
                ) from exc
            if hidden.ndim != 3 or hidden.shape[1] != len(req.candidate_token_ids):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "hidden must be [B, K, H] and match candidate_token_ids length"
                    ),
                )

            candidate_ids = torch.tensor(
                [req.candidate_token_ids],
                device=self.device,
                dtype=torch.long,
            )
            cache_position = torch.arange(
                st.seq_len,
                st.seq_len + len(req.candidate_token_ids),
                device=self.device,
                dtype=torch.long,
            )
            attention_mask = torch.ones(
                (1, st.seq_len + len(req.candidate_token_ids)),
                device=self.device,
                dtype=torch.long,
            )

            out = self.back.model(
                inputs_embeds=hidden,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=st.past_key_values,
                use_cache=True,
                return_dict=True,
            )
            logits = self.back.lm_head(out.last_hidden_state)

            accepted = 1
            fallback_token_id: int | None = None
            for idx in range(1, len(req.candidate_token_ids)):
                context_ids = torch.cat([st.token_ids, candidate_ids[:, :idx]], dim=1)
                target_token = select_next_token(
                    input_ids=context_ids,
                    logits=logits[:, idx - 1, :],
                    processors=st.processors,
                    warpers=st.warpers,
                    do_sample=st.do_sample,
                )
                if int(target_token.item()) == int(req.candidate_token_ids[idx]):
                    accepted += 1
                else:
                    fallback_token_id = int(target_token.item())
                    break

            st.past_key_values = out.past_key_values
            if accepted < len(req.candidate_token_ids):
                st.past_key_values.crop(st.seq_len + accepted)
            st.seq_len += accepted
            accepted_ids = req.candidate_token_ids[:accepted]
            st.token_ids = torch.cat(
                [
                    st.token_ids,
                    torch.tensor([accepted_ids], device=self.device, dtype=torch.long),
                ],
                dim=1,
            )
            st.next_logits = logits[:, accepted - 1, :]
            st.last_touch = time.time()

            t1 = time.perf_counter()
            return SpecVerifyResponse(
                session_id=req.session_id,
                accepted_token_ids=accepted_ids,
                fallback_token_id=fallback_token_id,
                seq_len=st.seq_len,
                server_ms=(t1 - t0) * 1000.0,
                codec_decode_ms=codec_decode_ms,
            )

        @app.post("/spec_commit", response_model=SpecPrefillResponse)
        @torch.no_grad()
        def spec_commit(req: SpecCommitRequest) -> SpecPrefillResponse:
            self._cleanup_sessions()
            st = self.spec_sessions.get(req.session_id)
            if st is None:
                raise HTTPException(
                    status_code=404,
                    detail="unknown session_id; call /spec_prefill first",
                )
            expected = st.seq_len + 1
            if req.seq_len != expected:
                raise HTTPException(
                    status_code=400,
                    detail=f"seq_len mismatch: got {req.seq_len}, expected {expected}",
                )

            commit_ctx = CodecContext(
                phase="decode",
                side="cloud_self_spec",
                session_id=req.session_id,
                seq_len=req.seq_len,
                step=req.token_step,
                extras=dict(req.codec_extras or {}),
            )
            try:
                t_codec0 = time.perf_counter()
                hidden_last = self._payload_to_hidden(req.hidden_last, context=commit_ctx)
                codec_decode_ms = (time.perf_counter() - t_codec0) * 1000.0
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"codec decode failed at spec commit: {exc}",
                ) from exc
            if hidden_last.ndim != 3 or hidden_last.shape[1] != 1:
                raise HTTPException(
                    status_code=400,
                    detail="hidden_last must be [B, 1, H]",
                )

            attention_mask = torch.ones(
                (1, req.seq_len),
                device=self.device,
                dtype=torch.long,
            )
            cache_position = torch.tensor([st.seq_len], device=self.device)

            t0 = time.perf_counter()
            out = self.back.model(
                inputs_embeds=hidden_last,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=st.past_key_values,
                use_cache=True,
                return_dict=True,
            )
            st.past_key_values = out.past_key_values
            st.seq_len = req.seq_len
            st.token_ids = torch.cat(
                [
                    st.token_ids,
                    torch.tensor([[int(req.token_id)]], device=self.device, dtype=torch.long),
                ],
                dim=1,
            )
            st.next_logits = self.back.lm_head(out.last_hidden_state)[:, -1, :]
            st.last_touch = time.time()

            t1 = time.perf_counter()
            return SpecPrefillResponse(
                session_id=req.session_id,
                seq_len=st.seq_len,
                server_ms=(t1 - t0) * 1000.0,
                codec_decode_ms=codec_decode_ms,
            )

    def run(self, host: str = "0.0.0.0", port: int = 8000, log_level: str = "info"):
        import uvicorn

        print(f"[info] start remote back server at http://{host}:{port}")
        uvicorn.run(self.app, host=host, port=port, log_level=log_level)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SplitLLM remote back server.")
    p.add_argument("--back_dir", type=str, default="./split_out/back")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--log_level", type=str, default="info")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument(
        "--back_quant",
        type=str,
        default="none",
        help="Back model quant mode: none or bnb_8bit (aliases: hf_int8/int8/bnb).",
    )
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--session_ttl_sec", type=int, default=1800)
    p.add_argument(
        "--codec",
        type=str,
        default="default",
        help=(
            "Activation codec spec. "
            "Builtins: default, identity_fp32. "
            "Custom: module / module:attr / module.attr"
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from model.cli_common import build_codec

        codec = build_codec(args.codec)
        server = RemoteBackServer(
            back_dir=args.back_dir,
            device=args.device,
            dtype=args.dtype,
            back_quant=args.back_quant,
            revision=args.revision,
            session_ttl_sec=int(args.session_ttl_sec),
            codec=codec,
        )
        server.run(
            host=args.host,
            port=int(args.port),
            log_level=args.log_level,
        )
        return 0
    except KeyboardInterrupt:
        print("[error] interrupted by user")
        return 130
    except Exception as exc:
        print(f"[error] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["RemoteBackServer", "parse_args", "main"]
