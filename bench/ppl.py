from __future__ import annotations

import argparse
import json
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model import CodecContext, LocalSplitRuntime
from model.codec import ActivationCodec

from bench.utils import (
    CodecTiming,
    apply_codec,
    build_codec,
    load_texts,
    parse_codec_extras,
    safe_exp,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark perplexity (PPL) for split front/back checkpoints.",
    )
    p.add_argument("--front_dir", type=str, default="./split_out/front")
    p.add_argument("--back_dir", type=str, default="./split_out/back")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--tokenizer_id", type=str, default="Qwen/Qwen3-1.7B")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="auto")
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
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--progress_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_json", type=str, default=None)
    return p.parse_args()


@torch.no_grad()
def batch_nll(
    *,
    runtime: LocalSplitRuntime,
    codec: ActivationCodec,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    session_id: str,
    step: int,
    codec_extras: dict[str, Any],
    timing: CodecTiming,
) -> tuple[float, int]:
    out_front = runtime.front.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    hidden = out_front.last_hidden_state

    context = CodecContext(
        phase="prefill",
        side="bench_local",
        session_id=session_id,
        seq_len=int(hidden.shape[1]),
        step=int(step),
        extras=dict(codec_extras),
    )
    hidden = apply_codec(
        codec=codec,
        hidden=hidden,
        context=context,
        device=runtime.device,
        dtype=runtime.dtype,
        timing=timing,
    )

    out_back = runtime.back.model(
        inputs_embeds=hidden,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    logits = runtime.back.lm_head(out_back.last_hidden_state)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()
    shift_labels = shift_labels.masked_fill(shift_mask == 0, -100)

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    token_cnt = int(shift_mask.sum().item())
    return float(loss.item()), token_cnt


def run(args: argparse.Namespace) -> Dict[str, Any]:
    torch.manual_seed(int(args.seed))
    codec_extras = parse_codec_extras(args.codec_extras_json)
    codec = build_codec(args.codec)

    runtime = LocalSplitRuntime(
        front_dir=args.front_dir,
        back_dir=args.back_dir,
        device=args.device,
        dtype=args.dtype,
        revision=args.revision,
        codec=codec,
    )
    codec = runtime.codec

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

    print(f"[info] benchmark=ppl mode=local_split codec={codec.name}")
    print(f"[info] device={runtime.device}, dtype={runtime.dtype}")
    print(f"[info] loaded {n} samples from {args.dataset_name}/{args.dataset_config}:{args.split}")

    batch_size = int(args.batch_size)
    progress_every = max(1, int(args.progress_every))
    amp_enabled = runtime.device.type == "cuda" and runtime.dtype in (
        torch.float16,
        torch.bfloat16,
    )

    timing = CodecTiming()
    session_id = str(uuid.uuid4())
    total_nll = 0.0
    total_tok = 0
    t0 = time.perf_counter()

    codec.start_session(session_id)
    try:
        for i in range(0, n, batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = tokenizer(
                batch_texts,
                max_length=int(args.max_length),
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(runtime.device)
            attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(
                runtime.device
            )

            with (
                torch.autocast(device_type="cuda", dtype=runtime.dtype)
                if amp_enabled
                else nullcontext()
            ):
                nll, tok_cnt = batch_nll(
                    runtime=runtime,
                    codec=codec,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    session_id=session_id,
                    step=(i // batch_size),
                    codec_extras=codec_extras,
                    timing=timing,
                )
            total_nll += nll
            total_tok += tok_cnt

            if ((i // batch_size) % progress_every) == 0:
                done = min(i + batch_size, n)
                running_ppl = safe_exp(total_nll / max(1, total_tok))
                print(f"[progress] processed {done}/{n}, running ppl={running_ppl:.6f}")
    finally:
        codec.end_session(session_id)

    elapsed_sec = time.perf_counter() - t0
    ppl = safe_exp(total_nll / max(1, total_tok))
    tokens_per_sec = float(total_tok / elapsed_sec) if elapsed_sec > 0 else 0.0
    samples_per_sec = float(n / elapsed_sec) if elapsed_sec > 0 else 0.0
    num_batches = (n + batch_size - 1) // batch_size if n > 0 else 0

    result: Dict[str, Any] = {
        "benchmark": {
            "name": "ppl",
            "mode": "local_split",
        },
        "model": {
            "front_dir": str(Path(args.front_dir).expanduser()),
            "back_dir": str(Path(args.back_dir).expanduser()),
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
        },
        "eval": {
            "samples": int(n),
            "batch_size": batch_size,
            "max_length": int(args.max_length),
            "total_tokens": int(total_tok),
            "mean_token_nll": float(total_nll / max(1, total_tok)),
            "ppl": float(ppl),
            "elapsed_sec": float(elapsed_sec),
            "tokens_per_sec": float(tokens_per_sec),
            "samples_per_sec": float(samples_per_sec),
            "num_batches": int(num_batches),
        },
        "codec_timing_ms": {
            "encode_total": float(timing.encode_ms),
            "decode_total": float(timing.decode_ms),
            "encode_per_batch": float(timing.encode_ms / max(1, num_batches)),
            "decode_per_batch": float(timing.decode_ms / max(1, num_batches)),
        },
    }

    print("\n[result]")
    print(f"Samples: {n}")
    print(f"Max length: {args.max_length}, Batch size: {batch_size}")
    print(f"Total tokens: {total_tok}")
    print(f"PPL: {ppl:.6f}")
    print(f"Elapsed: {elapsed_sec:.3f}s, token/s: {tokens_per_sec:.3f}")

    if args.out_json:
        out_path = Path(args.out_json).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
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
