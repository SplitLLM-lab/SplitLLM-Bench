from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests
import torch
import torch.nn as nn
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
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

from .codec import (
    ActivationCodec,
    CodecContext,
    EncodedActivation,
    ensure_codec,
)


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


def load_front_model(front_dir: str, device: torch.device, dtype: torch.dtype):
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


def load_back_model(back_dir: str, device: torch.device, dtype: torch.dtype):
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
    per_token_rtt_ms: list[float] = field(default_factory=list)
    server_ms: list[float] = field(default_factory=list)
    total_ms: float = 0.0


@dataclass
class RemoteTokenResponse:
    session_id: str
    next_token_id: int
    seq_len: int
    server_ms: float
    rtt_ms: float


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


class LocalSplitRuntime:
    def __init__(
        self,
        *,
        front_dir: str,
        back_dir: str,
        device: str | None = "auto",
        dtype: str = "auto",
        revision: str | None = None,
        codec: ActivationCodec | None = None,
    ):
        self.device, self.dtype = pick_device_and_dtype(device, dtype)
        self.codec = ensure_codec(codec)
        front_local = resolve_dir(front_dir, revision=revision)
        back_local = resolve_dir(back_dir, revision=revision)

        print(
            f"[info] loading local_split front={front_local} back={back_local} "
            f"device={self.device} dtype={self.dtype} codec={self.codec.name}"
        )
        self.front = load_front_model(front_local, self.device, self.dtype)
        self.back = load_back_model(back_local, self.device, self.dtype)
        print("[ok] local_split runtime ready")

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
        processors = build_logits_processors(
            prompt_length=prompt_len,
            eos_token_id=eos_token_id,
            cfg=sampling,
        )
        warpers = build_logits_warpers(sampling)

        front_cache = DynamicCache()
        back_cache = DynamicCache()

        cache_position = torch.arange(0, prompt_len, device=self.device)
        codec_extras = dict(codec_extras or {})

        t_prefill0 = time.perf_counter()

        mid = self.front.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=front_cache,
            use_cache=True,
            return_dict=True,
        )
        prefill_ctx = CodecContext(
            phase="prefill",
            side="local",
            session_id=None,
            seq_len=prompt_len,
            step=0,
            extras=codec_extras,
        )
        encoded_prefill = self.codec.encode(mid.last_hidden_state, context=prefill_ctx)
        prefill_hidden = self.codec.decode(
            encoded_prefill,
            context=prefill_ctx,
            device=self.device,
            dtype=self.dtype,
        )
        out = self.back.model(
            inputs_embeds=prefill_hidden,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=back_cache,
            use_cache=True,
            return_dict=True,
        )
        logits = self.back.lm_head(out.last_hidden_state)[:, -1, :]

        all_ids = input_ids
        next_token = select_next_token(
            input_ids=all_ids,
            logits=logits,
            processors=processors,
            warpers=warpers,
            do_sample=sampling.do_sample,
        )

        t_prefill1 = time.perf_counter()
        generated_ids = [int(next_token.item())]
        per_token_rtt_ms: list[float] = []
        finish_reason = "length"

        all_ids = torch.cat([all_ids, next_token], dim=1)
        if generated_ids[-1] in stop_token_ids:
            finish_reason = "stop"

        for _ in range(max(0, max_new_tokens - 1)):
            if finish_reason == "stop":
                break

            t_step0 = time.perf_counter()
            cache_position = cache_position[-1:] + 1

            hid = self.front.model(
                input_ids=next_token,
                cache_position=cache_position,
                past_key_values=front_cache,
                use_cache=True,
                return_dict=True,
            ).last_hidden_state
            decode_ctx = CodecContext(
                phase="decode",
                side="local",
                session_id=None,
                seq_len=int(all_ids.shape[1]),
                step=len(generated_ids),
                extras=codec_extras,
            )
            encoded_decode = self.codec.encode(hid, context=decode_ctx)
            decode_hidden = self.codec.decode(
                encoded_decode,
                context=decode_ctx,
                device=self.device,
                dtype=self.dtype,
            )

            out = self.back.model(
                inputs_embeds=decode_hidden,
                cache_position=cache_position,
                past_key_values=back_cache,
                use_cache=True,
                return_dict=True,
            )
            logits = self.back.lm_head(out.last_hidden_state)[:, -1, :]

            next_token = select_next_token(
                input_ids=all_ids,
                logits=logits,
                processors=processors,
                warpers=warpers,
                do_sample=sampling.do_sample,
            )

            t_step1 = time.perf_counter()
            per_token_rtt_ms.append((t_step1 - t_step0) * 1000.0)

            generated_ids.append(int(next_token.item()))
            all_ids = torch.cat([all_ids, next_token], dim=1)
            if generated_ids[-1] in stop_token_ids:
                finish_reason = "stop"

        t_total1 = time.perf_counter()
        return RuntimeGenerateResult(
            prompt_token_ids=input_ids[0].tolist(),
            generated_token_ids=generated_ids,
            finish_reason=finish_reason,
            ttft_ms=(t_prefill1 - t_prefill0) * 1000.0,
            per_token_rtt_ms=per_token_rtt_ms,
            total_ms=(t_total1 - t_total0) * 1000.0,
        )


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
        encoded = self.codec.encode(hidden, context=context)
        payload = {
            "session_id": session_id,
            "hidden": encoded_to_payload(self.codec, encoded),
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
        encoded = self.codec.encode(hidden_last, context=context)
        payload = {
            "session_id": session_id,
            "hidden_last": encoded_to_payload(self.codec, encoded),
            "seq_len": int(seq_len),
            "token_step": int(token_step),
            "codec_extras": context.extras,
        }

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


class ResetRequest(BaseModel):
    session_id: str


class TokenResponse(BaseModel):
    session_id: str
    next_token_id: int
    seq_len: int
    server_ms: float


@dataclass
class SessionState:
    past_key_values: Any
    seq_len: int
    token_ids: torch.Tensor
    processors: LogitsProcessorList
    warpers: LogitsProcessorList
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
    ):
        self.device, self.dtype = pick_device_and_dtype(device, dtype)
        self.codec = ensure_codec(codec)
        self.back_dir = resolve_dir(back_dir, revision=revision)
        self.session_ttl_sec = session_ttl_sec
        self.sessions: dict[str, SessionState] = {}

        print(
            f"[info] loading remote back={self.back_dir} "
            f"device={self.device} dtype={self.dtype} codec={self.codec.name}"
        )
        self.back = load_back_model(self.back_dir, self.device, self.dtype)
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
                "sessions": len(self.sessions),
            }

        @app.post("/reset")
        def reset(req: ResetRequest) -> dict[str, bool]:
            self.sessions.pop(req.session_id, None)
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
                hidden = self._payload_to_hidden(req.hidden, context=prefill_ctx)
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
                hidden_last = self._payload_to_hidden(
                    req.hidden_last,
                    context=decode_ctx,
                )
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
            )

    def run(self, host: str = "0.0.0.0", port: int = 8000, log_level: str = "info"):
        import uvicorn

        print(f"[info] start remote back server at http://{host}:{port}")
        uvicorn.run(self.app, host=host, port=port, log_level=log_level)


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
            f"device={self.device} dtype={self.dtype} codec={self.codec.name}"
        )
        self.front = load_front_model(front_local, self.device, self.dtype)
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
        per_token_rtt_ms: list[float] = []
        server_ms: list[float] = []
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
            session_id = prefill_resp.session_id

            next_token_id = int(prefill_resp.next_token_id)
            ttft_ms = prefill_resp.rtt_ms
            generated_ids.append(next_token_id)
            server_ms.append(prefill_resp.server_ms)

            if next_token_id in stop_token_ids:
                finish_reason = "stop"

            seq_len = prompt_len
            next_token = torch.tensor([[next_token_id]], device=self.device)

            for _ in range(max(0, max_new_tokens - 1)):
                if finish_reason == "stop":
                    break

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

                generated_ids.append(next_token_id)
                per_token_rtt_ms.append(decode_resp.rtt_ms)
                server_ms.append(decode_resp.server_ms)

                if next_token_id in stop_token_ids:
                    finish_reason = "stop"

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

        t_total1 = time.perf_counter()
        return RuntimeGenerateResult(
            prompt_token_ids=input_ids[0].detach().cpu().tolist(),
            generated_token_ids=generated_ids,
            finish_reason=finish_reason,
            ttft_ms=ttft_ms,
            per_token_rtt_ms=per_token_rtt_ms,
            server_ms=server_ms,
            total_ms=(t_total1 - t_total0) * 1000.0,
        )


__all__ = [
    "SamplingConfig",
    "RuntimeGenerateResult",
    "LocalSplitRuntime",
    "RemoteSplitRuntime",
    "RemoteBackServer",
]
