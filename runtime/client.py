from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from model.cli_common import build_codec, parse_codec_extras
from model.model import SplitLLMModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run SplitLLM remote client generation.",
    )
    p.add_argument("--front_dir", type=str, default="./split_out/front")
    p.add_argument("--tokenizer_id", type=str, default="Qwen/Qwen3-1.7B")
    p.add_argument("--server_url", type=str, required=True)
    p.add_argument("--timeout_sec", type=float, default=120.0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument(
        "--front_quant",
        type=str,
        default="none",
        help="Front model quant mode: none or bnb_8bit (aliases: hf_int8/int8/bnb).",
    )
    p.add_argument("--revision", type=str, default=None)
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
    p.add_argument("--prompt", type=str, default="Explain split inference in two sentences.")
    p.add_argument("--prompt_file", type=str, default=None)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument(
        "--generation_mode",
        type=str,
        default="autoregressive",
        choices=["autoregressive", "self_speculative"],
    )
    p.add_argument("--assistant_early_exit", type=int, default=None)
    p.add_argument("--num_speculations", type=int, default=3)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--min_new_tokens", type=int, default=0)
    p.add_argument("--no_repeat_ngram_size", type=int, default=0)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--no_chat_template", action="store_true")
    p.add_argument("--no_generation_prompt", action="store_true")
    p.add_argument("--out_json", type=str, default=None)
    return p.parse_args()


def resolve_prompt(prompt: str, prompt_file: str | None) -> str:
    if prompt_file is None:
        return prompt
    text = Path(prompt_file).expanduser().read_text(encoding="utf-8").strip()
    if text == "":
        raise ValueError("prompt_file is empty")
    return text


def run(args: argparse.Namespace) -> dict[str, Any]:
    codec = build_codec(args.codec)
    codec_extras = parse_codec_extras(args.codec_extras_json)
    prompt = resolve_prompt(args.prompt, args.prompt_file)
    if str(args.generation_mode) == "self_speculative" and args.assistant_early_exit is None:
        raise ValueError(
            "self_speculative generation requires --assistant_early_exit "
            "(for LayerSkip Llama3 8B, try 4)"
        )

    print("[info] building remote client model")
    model = SplitLLMModel.from_pretrained(
        mode="remote_back",
        tokenizer_id=args.tokenizer_id,
        front_dir=args.front_dir,
        server_url=args.server_url,
        device=args.device,
        dtype=args.dtype,
        front_quant=args.front_quant,
        timeout_sec=float(args.timeout_sec),
        revision=args.revision,
        codec=codec,
        enable_self_speculative=str(args.generation_mode) == "self_speculative",
    )

    print(
        f"[info] generating max_new_tokens={int(args.max_new_tokens)} "
        f"do_sample={bool(args.do_sample)}"
    )
    result = model.generate(
        prompt=prompt,
        max_new_tokens=int(args.max_new_tokens),
        do_sample=bool(args.do_sample),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        top_p=float(args.top_p),
        min_new_tokens=int(args.min_new_tokens),
        no_repeat_ngram_size=int(args.no_repeat_ngram_size),
        repetition_penalty=float(args.repetition_penalty),
        codec_extras=codec_extras,
        assistant_early_exit=(
            int(args.assistant_early_exit)
            if str(args.generation_mode) == "self_speculative"
            else None
        ),
        num_speculations=int(args.num_speculations),
        use_chat_template=not bool(args.no_chat_template),
        add_generation_prompt=not bool(args.no_generation_prompt),
    )

    payload: dict[str, Any] = {
        "text": result.text,
        "generated_token_ids": result.generated_token_ids,
        "prompt_token_ids": result.prompt_token_ids,
        "full_token_ids": result.full_token_ids,
        "finish_reason": result.finish_reason,
        "timing": result.timing,
    }

    print("\n[result]")
    print(result.text)
    print(
        f"[ok] finish_reason={result.finish_reason} "
        f"tokens={len(result.generated_token_ids)} "
        f"ttft_ms={float(result.timing.get('ttft_ms', 0.0)):.3f}"
    )

    if args.out_json:
        out_path = Path(args.out_json).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] wrote {out_path.resolve()}")

    return payload


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
