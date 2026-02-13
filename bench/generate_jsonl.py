from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from transformers import AutoTokenizer

from model import LocalSplitRuntime, SplitLLMModel

from bench.utils import (
    build_codec,
    build_generation_kwargs,
    generate_jsonl_with_model,
    parse_codec_extras,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run line-by-line JSONL generation with SplitLLM local split runtime. "
            "Input JSONL must contain one prompt field per row."
        ),
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

    p.add_argument("--input_jsonl", type=str, required=True)
    p.add_argument("--output_jsonl", type=str, required=True)
    p.add_argument("--prompt_key", type=str, default="prompt")
    p.add_argument("--result_key", type=str, default="result")
    p.add_argument("--index_key", type=str, default="index")
    p.add_argument(
        "--drop_index",
        action="store_true",
        help="If set, output rows keep only generated text without index field.",
    )

    p.add_argument(
        "--decoding",
        type=str,
        default="greedy",
        choices=["greedy", "top_k", "top_p"],
        help="Decoding strategy.",
    )
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--min_new_tokens", type=int, default=0)
    p.add_argument("--no_repeat_ngram_size", type=int, default=0)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--use_chat_template", action="store_true")
    p.add_argument("--add_generation_prompt", action="store_true")
    p.add_argument("--keep_special_tokens", action="store_true")

    p.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Worker count for JSONL generation pipeline. >1 requires thread-safe model/runtime.",
    )
    p.add_argument(
        "--max_in_flight",
        type=int,
        default=0,
        help="Max queued futures. <=0 means auto.",
    )
    p.add_argument("--progress_every", type=int, default=100)
    p.add_argument("--out_json", type=str, default=None)
    return p.parse_args()


def run(args: argparse.Namespace) -> Dict[str, Any]:
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

    model = SplitLLMModel(
        mode="local_split",
        tokenizer=tokenizer,
        runtime=runtime,
    )
    generation_kwargs = build_generation_kwargs(
        decoding=str(args.decoding),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        top_p=float(args.top_p),
        min_new_tokens=int(args.min_new_tokens),
        no_repeat_ngram_size=int(args.no_repeat_ngram_size),
        repetition_penalty=float(args.repetition_penalty),
    )
    generation_kwargs["codec_extras"] = codec_extras
    generation_kwargs["use_chat_template"] = bool(args.use_chat_template)
    generation_kwargs["add_generation_prompt"] = bool(args.add_generation_prompt)
    generation_kwargs["skip_special_tokens"] = not bool(args.keep_special_tokens)

    max_in_flight = None if int(args.max_in_flight) <= 0 else int(args.max_in_flight)
    input_path = Path(args.input_jsonl).expanduser()
    output_path = Path(args.output_jsonl).expanduser()

    print(f"[info] benchmark=generate_jsonl mode=local_split codec={codec.name}")
    print(f"[info] device={runtime.device}, dtype={runtime.dtype}")
    print(
        f"[info] input={input_path} output={output_path} "
        f"prompt_key={args.prompt_key} result_key={args.result_key} index_key={args.index_key}"
    )
    print(
        f"[info] decoding={args.decoding} max_new_tokens={int(args.max_new_tokens)} "
        f"workers={int(args.num_workers)} max_in_flight={max_in_flight if max_in_flight is not None else 'auto'}"
    )

    rows = generate_jsonl_with_model(
        model=model,
        input_path=input_path,
        output_path=output_path,
        generation_kwargs=generation_kwargs,
        prompt_key=str(args.prompt_key),
        result_key=str(args.result_key),
        index_key=str(args.index_key),
        num_workers=int(args.num_workers),
        max_in_flight=max_in_flight,
        progress_every=max(1, int(args.progress_every)),
        keep_index=not bool(args.drop_index),
    )

    result: Dict[str, Any] = {
        "benchmark": {
            "name": "generate_jsonl",
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
            "input_jsonl": str(input_path),
            "prompt_key": str(args.prompt_key),
        },
        "runtime": {
            "device": str(runtime.device),
            "dtype": str(runtime.dtype),
        },
        "eval": {
            "rows": int(rows),
            "output_jsonl": str(output_path),
            "result_key": str(args.result_key),
            "index_key": str(args.index_key),
            "keep_index": bool(not args.drop_index),
            "decoding": str(args.decoding),
            "max_new_tokens": int(args.max_new_tokens),
        },
    }

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
