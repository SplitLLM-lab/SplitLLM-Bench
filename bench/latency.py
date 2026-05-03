from __future__ import annotations

import argparse
import json
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache

from model import CodecContext, LocalSplitRuntime, RemoteSplitRuntime, SamplingConfig
from model.codec import ActivationCodec, EncodedActivation
from runtime import (
    build_logits_processors,
    build_logits_warpers,
    encoded_to_payload,
    select_next_token,
)

from bench.utils import build_codec, load_texts, parse_codec_extras


RuntimeType = LocalSplitRuntime | RemoteSplitRuntime


@dataclass
class CodecRoundMetric:
    phase: str
    encode_ms: float
    decode_ms: float
    wire_bytes: int


@dataclass
class SampleMetric:
    ttft_ms: float
    total_ms: float
    prompt_tokens: int
    generated_tokens: int
    finish_reason: str
    decode_step_ms: list[float] = field(default_factory=list)
    codec_rounds: list[CodecRoundMetric] = field(default_factory=list)
    remote_prefill_rtt_ms: float = 0.0
    remote_decode_rtt_ms: list[float] = field(default_factory=list)
    remote_prefill_server_ms: float = 0.0
    remote_decode_server_ms: list[float] = field(default_factory=list)
    self_speculative: bool = False
    draft_tokens: int = 0
    accepted_tokens: int = 0
    verify_rounds: int = 0
    fallback_tokens: int = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark split generation latency with codec metrics "
            "(TTFT, system latency, codec encode/decode latency, transfer bytes)."
        ),
    )
    p.add_argument(
        "--runtime_mode",
        type=str,
        default="local_split",
        choices=["local_split", "remote_back"],
    )
    p.add_argument("--front_dir", type=str, default="./split_out/front")
    p.add_argument("--back_dir", type=str, default="./split_out/back")
    p.add_argument("--server_url", type=str, default=None)
    p.add_argument("--timeout_sec", type=float, default=120.0)
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--tokenizer_id", type=str, default="Qwen/Qwen3-1.7B")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument(
        "--front_quant",
        type=str,
        default="none",
        help="Front model quant mode: none or bnb_8bit (aliases: hf_int8/int8/bnb).",
    )
    p.add_argument(
        "--back_quant",
        type=str,
        default="none",
        help="Back model quant mode: none or bnb_8bit (aliases: hf_int8/int8/bnb).",
    )
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
    p.add_argument(
        "--codec_extras_json",
        type=str,
        default=None,
        help="Optional JSON object passed into CodecContext.extras",
    )

    p.add_argument("--dataset_name", type=str, default="wikitext")
    p.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--max_prompt_length", type=int, default=256)

    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument(
        "--generation_mode",
        type=str,
        default="autoregressive",
        choices=["autoregressive", "self_speculative"],
    )
    p.add_argument(
        "--assistant_early_exit",
        type=int,
        default=None,
        help="Optional self-spec override; default uses all front layers.",
    )
    p.add_argument("--num_speculations", type=int, default=3)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--min_new_tokens", type=int, default=0)
    p.add_argument("--no_repeat_ngram_size", type=int, default=0)
    p.add_argument("--repetition_penalty", type=float, default=1.0)

    p.add_argument("--progress_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_json", type=str, default=None)
    return p.parse_args()


def build_runtime(args: argparse.Namespace, codec: ActivationCodec) -> RuntimeType:
    enable_self_speculative = str(args.generation_mode) == "self_speculative"
    if args.runtime_mode == "local_split":
        return LocalSplitRuntime(
            front_dir=args.front_dir,
            back_dir=args.back_dir,
            device=args.device,
            dtype=args.dtype,
            front_quant=args.front_quant,
            back_quant=args.back_quant,
            revision=args.revision,
            codec=codec,
            enable_self_speculative=enable_self_speculative,
        )
    if not args.server_url:
        raise ValueError("remote_back mode requires --server_url")
    return RemoteSplitRuntime(
        front_dir=args.front_dir,
        server_url=args.server_url,
        device=args.device,
        dtype=args.dtype,
        front_quant=args.front_quant,
        timeout_sec=float(args.timeout_sec),
        revision=args.revision,
        codec=codec,
        enable_self_speculative=enable_self_speculative,
    )


def effective_front_early_exit_layers(
    runtime: RuntimeType,
    sampling: SamplingConfig,
) -> int | None:
    if not (sampling.self_speculative or sampling.assistant_early_exit is not None):
        return None
    if sampling.assistant_early_exit is not None:
        return int(sampling.assistant_early_exit)
    return int(runtime.front.model.config.num_hidden_layers)


def mean(xs: list[float]) -> float:
    if not xs:
        return 0.0
    return float(sum(xs) / len(xs))


def max_or_zero(xs: list[float] | list[int]) -> float:
    if not xs:
        return 0.0
    return float(max(xs))


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    if q <= 0:
        return float(min(xs))
    if q >= 100:
        return float(max(xs))
    arr = sorted(float(x) for x in xs)
    pos = (len(arr) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(arr) - 1)
    w = pos - lo
    return arr[lo] * (1.0 - w) + arr[hi] * w


def stat_block(xs: list[float]) -> Dict[str, Any]:
    return {
        "count": int(len(xs)),
        "mean": float(mean(xs)),
        "max": float(max_or_zero(xs)),
        "p50": float(percentile(xs, 50)),
        "p95": float(percentile(xs, 95)),
    }


def wire_size_bytes(codec: ActivationCodec, encoded: EncodedActivation) -> int:
    payload = encoded_to_payload(codec, encoded)
    wire_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return len(wire_json.encode("utf-8"))


def codec_roundtrip(
    *,
    codec: ActivationCodec,
    hidden: torch.Tensor,
    context: CodecContext,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, CodecRoundMetric]:
    t0 = time.perf_counter()
    encoded = codec.encode(hidden, context=context)
    encode_ms = (time.perf_counter() - t0) * 1000.0

    transfer_bytes = wire_size_bytes(codec, encoded)

    t1 = time.perf_counter()
    decoded = codec.decode(
        encoded,
        context=context,
        device=device,
        dtype=dtype,
    )
    decode_ms = (time.perf_counter() - t1) * 1000.0

    metric = CodecRoundMetric(
        phase=context.phase,
        encode_ms=float(encode_ms),
        decode_ms=float(decode_ms),
        wire_bytes=int(transfer_bytes),
    )
    return decoded, metric


@torch.no_grad()
def run_one_sample(
    *,
    runtime: LocalSplitRuntime,
    codec: ActivationCodec,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    eos_token_id: int | None,
    stop_token_ids: set[int],
    sampling: SamplingConfig,
    max_new_tokens: int,
    codec_extras: dict[str, Any],
    amp_enabled: bool,
) -> SampleMetric:
    prompt_len = int(input_ids.shape[1])
    if max_new_tokens <= 0:
        return SampleMetric(
            ttft_ms=0.0,
            total_ms=0.0,
            prompt_tokens=prompt_len,
            generated_tokens=0,
            finish_reason="length",
        )

    processors = build_logits_processors(
        prompt_length=prompt_len,
        eos_token_id=eos_token_id,
        cfg=sampling,
    )
    warpers = build_logits_warpers(sampling)

    front_cache = DynamicCache()
    back_cache = DynamicCache()
    cache_position = torch.arange(0, prompt_len, device=runtime.device)
    session_id = str(uuid.uuid4())

    codec_rounds: list[CodecRoundMetric] = []
    decode_step_ms: list[float] = []
    generated_ids: list[int] = []
    finish_reason = "length"

    ttft_ms = 0.0
    total_ms = 0.0
    t_total0 = time.perf_counter()
    all_ids = input_ids

    amp_context = (
        torch.autocast(device_type="cuda", dtype=runtime.dtype)
        if amp_enabled
        else nullcontext()
    )

    codec.start_session(session_id)
    try:
        with amp_context:
            t_prefill0 = time.perf_counter()
            mid = runtime.front.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=front_cache,
                use_cache=True,
                return_dict=True,
            )

            prefill_ctx = CodecContext(
                phase="prefill",
                side="bench_latency",
                session_id=session_id,
                seq_len=prompt_len,
                step=0,
                extras=dict(codec_extras),
            )
            prefill_hidden, prefill_metric = codec_roundtrip(
                codec=codec,
                hidden=mid.last_hidden_state,
                context=prefill_ctx,
                device=runtime.device,
                dtype=runtime.dtype,
            )
            codec_rounds.append(prefill_metric)

            out = runtime.back.model(
                inputs_embeds=prefill_hidden,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=back_cache,
                use_cache=True,
                return_dict=True,
            )
            logits = runtime.back.lm_head(out.last_hidden_state)[:, -1, :]

            next_token = select_next_token(
                input_ids=all_ids,
                logits=logits,
                processors=processors,
                warpers=warpers,
                do_sample=sampling.do_sample,
            )

            t_prefill1 = time.perf_counter()
            ttft_ms = (t_prefill1 - t_prefill0) * 1000.0

            generated_ids.append(int(next_token.item()))
            all_ids = torch.cat([all_ids, next_token], dim=1)
            if generated_ids[-1] in stop_token_ids:
                finish_reason = "stop"

            for _ in range(max(0, max_new_tokens - 1)):
                if finish_reason == "stop":
                    break

                t_step0 = time.perf_counter()
                cache_position = cache_position[-1:] + 1

                hid = runtime.front.model(
                    input_ids=next_token,
                    cache_position=cache_position,
                    past_key_values=front_cache,
                    use_cache=True,
                    return_dict=True,
                ).last_hidden_state

                decode_ctx = CodecContext(
                    phase="decode",
                    side="bench_latency",
                    session_id=session_id,
                    seq_len=int(all_ids.shape[1]),
                    step=len(generated_ids),
                    extras=dict(codec_extras),
                )
                decode_hidden, decode_metric = codec_roundtrip(
                    codec=codec,
                    hidden=hid,
                    context=decode_ctx,
                    device=runtime.device,
                    dtype=runtime.dtype,
                )
                codec_rounds.append(decode_metric)

                out = runtime.back.model(
                    inputs_embeds=decode_hidden,
                    cache_position=cache_position,
                    past_key_values=back_cache,
                    use_cache=True,
                    return_dict=True,
                )
                logits = runtime.back.lm_head(out.last_hidden_state)[:, -1, :]

                next_token = select_next_token(
                    input_ids=all_ids,
                    logits=logits,
                    processors=processors,
                    warpers=warpers,
                    do_sample=sampling.do_sample,
                )

                t_step1 = time.perf_counter()
                decode_step_ms.append((t_step1 - t_step0) * 1000.0)

                generated_ids.append(int(next_token.item()))
                all_ids = torch.cat([all_ids, next_token], dim=1)
                if generated_ids[-1] in stop_token_ids:
                    finish_reason = "stop"
            total_ms = (time.perf_counter() - t_total0) * 1000.0
    finally:
        try:
            codec.end_session(session_id)
        except Exception:
            pass

    return SampleMetric(
        ttft_ms=float(ttft_ms),
        total_ms=float(total_ms),
        prompt_tokens=prompt_len,
        generated_tokens=len(generated_ids),
        finish_reason=finish_reason,
        decode_step_ms=decode_step_ms,
        codec_rounds=codec_rounds,
    )


@torch.no_grad()
def metric_from_runtime_result(
    *,
    result,
    prompt_len: int,
) -> SampleMetric:
    codec_rounds: list[CodecRoundMetric] = []
    if result.generated_token_ids:
        codec_rounds.append(
            CodecRoundMetric(
                phase="prefill",
                encode_ms=float(result.prefill_codec_encode_ms),
                decode_ms=float(result.prefill_codec_decode_ms),
                wire_bytes=int(result.prefill_codec_wire_bytes),
            )
        )
    decode_codec_rounds = max(
        len(result.decode_codec_encode_ms),
        len(result.decode_codec_decode_ms),
        len(result.decode_codec_wire_bytes),
    )
    for i in range(decode_codec_rounds):
        codec_rounds.append(
            CodecRoundMetric(
                phase="decode",
                encode_ms=(
                    float(result.decode_codec_encode_ms[i])
                    if i < len(result.decode_codec_encode_ms)
                    else 0.0
                ),
                decode_ms=(
                    float(result.decode_codec_decode_ms[i])
                    if i < len(result.decode_codec_decode_ms)
                    else 0.0
                ),
                wire_bytes=(
                    int(result.decode_codec_wire_bytes[i])
                    if i < len(result.decode_codec_wire_bytes)
                    else 0
                ),
            )
        )
    return SampleMetric(
        ttft_ms=float(result.ttft_ms),
        total_ms=float(result.total_ms),
        prompt_tokens=prompt_len,
        generated_tokens=len(result.generated_token_ids),
        finish_reason=result.finish_reason,
        decode_step_ms=[float(x) for x in result.decode_step_ms],
        codec_rounds=codec_rounds,
        remote_prefill_rtt_ms=float(result.prefill_rtt_ms),
        remote_decode_rtt_ms=[float(x) for x in result.decode_rtt_ms],
        remote_prefill_server_ms=float(result.prefill_server_ms),
        remote_decode_server_ms=[float(x) for x in result.decode_server_ms],
        self_speculative=bool(result.self_speculative),
        draft_tokens=int(result.draft_tokens),
        accepted_tokens=int(result.accepted_tokens),
        verify_rounds=int(result.verify_rounds),
        fallback_tokens=int(result.fallback_tokens),
    )


@torch.no_grad()
def run_one_sample_runtime(
    *,
    runtime: RuntimeType,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    eos_token_id: int | None,
    stop_token_ids: set[int],
    sampling: SamplingConfig,
    max_new_tokens: int,
    codec_extras: dict[str, Any],
) -> SampleMetric:
    prompt_len = int(input_ids.shape[1])
    result = runtime.generate_from_ids(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        stop_token_ids=stop_token_ids,
        sampling=sampling,
        codec_extras=codec_extras,
    )
    return metric_from_runtime_result(result=result, prompt_len=prompt_len)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    codec_extras = parse_codec_extras(args.codec_extras_json)
    codec = build_codec(args.codec)
    self_speculative = str(args.generation_mode) == "self_speculative"

    runtime = build_runtime(args, codec)
    codec = runtime.codec
    mode = str(args.runtime_mode)
    is_remote = isinstance(runtime, RemoteSplitRuntime)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_id,
        use_fast=True,
        trust_remote_code=bool(args.trust_remote_code),
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = load_texts(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        text_column=args.text_column,
        samples=args.samples,
    )
    n = len(texts)
    if n <= 0:
        raise ValueError("no valid samples found from dataset")

    eos_token_id = (
        int(tokenizer.eos_token_id)
        if tokenizer.eos_token_id is not None
        else None
    )
    stop_token_ids: set[int] = set()
    if eos_token_id is not None:
        stop_token_ids.add(eos_token_id)
    vocab = tokenizer.get_vocab()
    if "<|im_end|>" in vocab:
        stop_token_ids.add(int(vocab["<|im_end|>"]))

    sampling = SamplingConfig(
        do_sample=bool(args.do_sample),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        top_p=float(args.top_p),
        min_new_tokens=int(args.min_new_tokens),
        no_repeat_ngram_size=int(args.no_repeat_ngram_size),
        repetition_penalty=float(args.repetition_penalty),
        self_speculative=self_speculative,
        assistant_early_exit=(
            int(args.assistant_early_exit)
            if self_speculative and args.assistant_early_exit is not None
            else None
        ),
        num_speculations=int(args.num_speculations),
    )
    sampling.validate()
    front_early_exit_layers = effective_front_early_exit_layers(runtime, sampling)

    amp_enabled = runtime.device.type == "cuda" and runtime.dtype in (
        torch.float16,
        torch.bfloat16,
    )

    print(
        f"[info] benchmark=latency mode={mode} codec={codec.name} "
        f"generation_mode={args.generation_mode}"
    )
    if sampling.self_speculative or sampling.assistant_early_exit is not None:
        print(
            "[info] self_speculative "
            f"front_early_exit_layers={front_early_exit_layers} "
            f"assistant_early_exit_override={sampling.assistant_early_exit} "
            f"num_speculations={sampling.num_speculations}"
        )
    print(f"[info] device={runtime.device}, dtype={runtime.dtype}, autocast={amp_enabled}")
    if is_remote:
        print(
            f"[info] remote server={args.server_url}, timeout_sec={float(args.timeout_sec):.1f}"
        )
        print(
            "[info] remote_back TTFT/decode steps are end-to-end; HTTP RTT/server_ms are reported separately"
        )
    print(f"[info] loaded {n} samples from {args.dataset_name}/{args.dataset_config}:{args.split}")
    print(f"[info] max_prompt_length={args.max_prompt_length}, max_new_tokens={args.max_new_tokens}")

    progress_every = max(1, int(args.progress_every))

    ttft_ms_all: list[float] = []
    total_ms_all: list[float] = []
    prompt_tokens_all: list[int] = []
    generated_tokens_all: list[int] = []
    decode_step_ms_all: list[float] = []
    prefill_encode_ms_all: list[float] = []
    prefill_decode_ms_all: list[float] = []
    decode_encode_ms_all: list[float] = []
    decode_decode_ms_all: list[float] = []
    prefill_wire_bytes_all: list[int] = []
    decode_wire_bytes_all: list[int] = []
    remote_prefill_rtt_ms_all: list[float] = []
    remote_decode_rtt_ms_all: list[float] = []
    remote_prefill_server_ms_all: list[float] = []
    remote_decode_server_ms_all: list[float] = []
    finish_reason_count: dict[str, int] = {}
    draft_tokens_all: list[int] = []
    accepted_tokens_all: list[int] = []
    verify_rounds_all: list[int] = []
    fallback_tokens_all: list[int] = []

    t_bench0 = time.perf_counter()

    for i, text in enumerate(texts):
        enc = tokenizer(
            text,
            max_length=int(args.max_prompt_length),
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(runtime.device)
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(
            runtime.device
        )

        if sampling.self_speculative or sampling.assistant_early_exit is not None:
            metric = run_one_sample_runtime(
                runtime=runtime,
                input_ids=input_ids,
                attention_mask=attention_mask,
                eos_token_id=eos_token_id,
                stop_token_ids=stop_token_ids,
                sampling=sampling,
                max_new_tokens=int(args.max_new_tokens),
                codec_extras=codec_extras,
            )
        elif is_remote:
            metric = run_one_sample_runtime(
                runtime=runtime,
                input_ids=input_ids,
                attention_mask=attention_mask,
                eos_token_id=eos_token_id,
                stop_token_ids=stop_token_ids,
                sampling=sampling,
                max_new_tokens=int(args.max_new_tokens),
                codec_extras=codec_extras,
            )
        else:
            metric = run_one_sample(
                runtime=runtime,
                codec=codec,
                input_ids=input_ids,
                attention_mask=attention_mask,
                eos_token_id=eos_token_id,
                stop_token_ids=stop_token_ids,
                sampling=sampling,
                max_new_tokens=int(args.max_new_tokens),
                codec_extras=codec_extras,
                amp_enabled=amp_enabled,
            )

        ttft_ms_all.append(metric.ttft_ms)
        total_ms_all.append(metric.total_ms)
        prompt_tokens_all.append(metric.prompt_tokens)
        generated_tokens_all.append(metric.generated_tokens)
        decode_step_ms_all.extend(metric.decode_step_ms)
        if is_remote:
            remote_prefill_rtt_ms_all.append(metric.remote_prefill_rtt_ms)
            remote_decode_rtt_ms_all.extend(metric.remote_decode_rtt_ms)
            remote_prefill_server_ms_all.append(metric.remote_prefill_server_ms)
            remote_decode_server_ms_all.extend(metric.remote_decode_server_ms)
        finish_reason_count[metric.finish_reason] = (
            finish_reason_count.get(metric.finish_reason, 0) + 1
        )
        draft_tokens_all.append(int(metric.draft_tokens))
        accepted_tokens_all.append(int(metric.accepted_tokens))
        verify_rounds_all.append(int(metric.verify_rounds))
        fallback_tokens_all.append(int(metric.fallback_tokens))

        for round_metric in metric.codec_rounds:
            if round_metric.phase == "prefill":
                prefill_encode_ms_all.append(round_metric.encode_ms)
                prefill_decode_ms_all.append(round_metric.decode_ms)
                prefill_wire_bytes_all.append(round_metric.wire_bytes)
            elif round_metric.phase == "decode":
                decode_encode_ms_all.append(round_metric.encode_ms)
                decode_decode_ms_all.append(round_metric.decode_ms)
                decode_wire_bytes_all.append(round_metric.wire_bytes)

        if ((i + 1) % progress_every) == 0:
            print(
                "[progress] processed "
                f"{i + 1}/{n}, "
                f"ttft_mean_ms={mean(ttft_ms_all):.3f}, "
                f"total_mean_ms={mean(total_ms_all):.3f}"
            )

    elapsed_sec = time.perf_counter() - t_bench0
    total_generated_tokens = int(sum(generated_tokens_all))
    samples_per_sec = float(n / elapsed_sec) if elapsed_sec > 0 else 0.0
    tokens_per_sec = (
        float(total_generated_tokens / elapsed_sec) if elapsed_sec > 0 else 0.0
    )

    total_encode_ms = (
        sum(prefill_encode_ms_all) + sum(decode_encode_ms_all)
    )
    total_decode_ms = (
        sum(prefill_decode_ms_all) + sum(decode_decode_ms_all)
    )
    total_prefill_wire = int(sum(prefill_wire_bytes_all))
    total_decode_wire = int(sum(decode_wire_bytes_all))
    total_draft_tokens = int(sum(draft_tokens_all))
    total_accepted_tokens = int(sum(accepted_tokens_all))
    acceptance_rate = (
        float(total_accepted_tokens / total_draft_tokens)
        if total_draft_tokens > 0
        else 0.0
    )

    result: Dict[str, Any] = {
        "benchmark": {
            "name": "latency",
            "mode": mode,
        },
        "model": {
            "front_dir": str(Path(args.front_dir).expanduser()),
            "back_dir": str(Path(args.back_dir).expanduser()) if not is_remote else None,
            "server_url": args.server_url if is_remote else None,
            "tokenizer_id": args.tokenizer_id,
            "revision": args.revision,
        },
        "codec": {
            "name": codec.name,
            "extras": codec_extras,
        },
        "dataset": {
            "name": args.dataset_name,
            "config": args.dataset_config,
            "split": args.split,
            "text_column": args.text_column,
        },
        "runtime": {
            "device": str(runtime.device),
            "dtype": str(runtime.dtype),
            "seed": int(args.seed),
            "autocast": bool(amp_enabled),
            "timeout_sec": float(args.timeout_sec) if is_remote else None,
        },
        "eval": {
            "samples": int(n),
            "max_prompt_length": int(args.max_prompt_length),
            "max_new_tokens": int(args.max_new_tokens),
            "generation_mode": str(args.generation_mode),
            "front_early_exit_layers": front_early_exit_layers,
            "assistant_early_exit": sampling.assistant_early_exit,
            "num_speculations": int(sampling.num_speculations),
            "elapsed_sec": float(elapsed_sec),
            "samples_per_sec": float(samples_per_sec),
            "generated_tokens_total": int(total_generated_tokens),
            "generated_tokens_per_sample_mean": float(mean(generated_tokens_all)),
            "generated_tokens_per_sec": float(tokens_per_sec),
            "prompt_tokens_total": int(sum(prompt_tokens_all)),
            "decode_steps_total": int(len(decode_step_ms_all)),
            "finish_reason_count": finish_reason_count,
            "self_speculative": {
                "enabled": bool(
                    sampling.self_speculative
                    or sampling.assistant_early_exit is not None
                ),
                "front_early_exit_layers": front_early_exit_layers,
                "assistant_early_exit_override": sampling.assistant_early_exit,
                "draft_tokens_total": int(total_draft_tokens),
                "accepted_tokens_total": int(total_accepted_tokens),
                "acceptance_rate": float(acceptance_rate),
                "verify_rounds_total": int(sum(verify_rounds_all)),
                "fallback_tokens_total": int(sum(fallback_tokens_all)),
            },
            "ttft_ms": stat_block(ttft_ms_all),
            "system_latency_ms": stat_block(total_ms_all),
            "decode_step_latency_ms": stat_block(decode_step_ms_all),
            "codec_latency_ms": {
                "encode_total": float(total_encode_ms),
                "decode_total": float(total_decode_ms),
                "encode_prefill": {
                    "mean": float(mean(prefill_encode_ms_all)),
                    "max": float(max_or_zero(prefill_encode_ms_all)),
                },
                "decode_prefill": {
                    "mean": float(mean(prefill_decode_ms_all)),
                    "max": float(max_or_zero(prefill_decode_ms_all)),
                },
                "encode_decode": {
                    "mean": float(mean(decode_encode_ms_all)),
                    "max": float(max_or_zero(decode_encode_ms_all)),
                },
                "decode_decode": {
                    "mean": float(mean(decode_decode_ms_all)),
                    "max": float(max_or_zero(decode_decode_ms_all)),
                },
            },
            "codec_transfer_bytes": {
                "prefill": {
                    "rounds": int(len(prefill_wire_bytes_all)),
                    "total": int(total_prefill_wire),
                    "avg_per_round": float(mean(prefill_wire_bytes_all)),
                    "max_per_round": float(max_or_zero(prefill_wire_bytes_all)),
                },
                "decode": {
                    "rounds": int(len(decode_wire_bytes_all)),
                    "total": int(total_decode_wire),
                    "avg_per_round": float(mean(decode_wire_bytes_all)),
                    "max_per_round": float(max_or_zero(decode_wire_bytes_all)),
                },
            },
            "remote": {
                "rtt_ms": {
                    "prefill": stat_block(remote_prefill_rtt_ms_all),
                    "decode": stat_block(remote_decode_rtt_ms_all),
                },
                "server_ms": {
                    "prefill": stat_block(remote_prefill_server_ms_all),
                    "decode": stat_block(remote_decode_server_ms_all),
                },
            },
        },
    }

    print("\n[result]")
    print(f"Samples: {n}")
    print(
        f"TTFT mean/max (ms): {mean(ttft_ms_all):.3f} / {max_or_zero(ttft_ms_all):.3f}"
    )
    print(
        "System latency mean/max (ms): "
        f"{mean(total_ms_all):.3f} / {max_or_zero(total_ms_all):.3f}"
    )
    print(
        "Codec prefill transfer avg/max (bytes): "
        f"{mean(prefill_wire_bytes_all):.1f} / {max_or_zero(prefill_wire_bytes_all):.1f}"
    )
    print(
        "Codec decode transfer avg/max (bytes): "
        f"{mean(decode_wire_bytes_all):.1f} / {max_or_zero(decode_wire_bytes_all):.1f}"
    )
    print(
        "Codec encode/decode total (ms): "
        f"{total_encode_ms:.3f} / {total_decode_ms:.3f}"
    )
    if sampling.self_speculative or sampling.assistant_early_exit is not None:
        print(
            "Self-spec draft/accepted/fallback: "
            f"{total_draft_tokens} / {total_accepted_tokens} / {sum(fallback_tokens_all)} "
            f"(acceptance={acceptance_rate:.3f})"
        )

    if args.out_json:
        out_path = Path(args.out_json).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[ok] wrote {out_path.resolve()}")

    return result


def main() -> int:
    args = parse_args()
    try:
        run(args)
        return 0
    except KeyboardInterrupt:
        print("[error] interrupted by user")
        return 130
    except Exception as exc:
        print(f"[error] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
