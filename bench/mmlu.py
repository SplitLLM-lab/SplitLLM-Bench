from __future__ import annotations

import argparse
import json
import random
import string
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from model import LocalSplitRuntime, SplitLLMModel

from bench.utils import (
    build_codec,
    build_generation_kwargs,
    generate_jsonl_with_model,
    parse_codec_extras,
)


CHOICE_LABELS = string.ascii_uppercase


@dataclass
class MCQItem:
    question: str
    choices: list[str]
    answer_idx: int
    subject: str


@dataclass
class PreparedSample:
    subject: str
    labels: str
    answer_idx: int
    used_shots: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Simple MMLU benchmark: load dataset, generate predictions from prompt JSONL, "
            "then score accuracy."
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

    p.add_argument("--dataset_name", type=str, default="cais/mmlu")
    p.add_argument(
        "--dataset_config",
        type=str,
        default="all",
        help="Dataset config/subset. Use empty string to disable config.",
    )
    p.add_argument("--fewshot_split", type=str, default="dev")
    p.add_argument("--eval_split", type=str, default="test")
    p.add_argument("--question_column", type=str, default="question")
    p.add_argument("--choices_column", type=str, default="choices")
    p.add_argument("--answer_column", type=str, default="answer")
    p.add_argument("--subject_column", type=str, default="subject")
    p.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated subjects. Default: all subjects.",
    )
    p.add_argument("--n_shot", type=int, default=5)
    p.add_argument("--samples", type=int, default=-1)
    p.add_argument("--max_samples_per_subject", type=int, default=-1)
    p.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Max sequence length for prompt + answer token(s).",
    )

    p.add_argument(
        "--decoding",
        type=str,
        default="greedy",
        choices=["greedy", "top_k", "top_p"],
    )
    p.add_argument("--max_new_tokens", type=int, default=1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--min_new_tokens", type=int, default=0)
    p.add_argument("--no_repeat_ngram_size", type=int, default=0)
    p.add_argument("--repetition_penalty", type=float, default=1.0)

    p.add_argument("--jsonl_num_workers", type=int, default=1)
    p.add_argument("--jsonl_max_in_flight", type=int, default=0)
    p.add_argument("--prompt_jsonl", type=str, default=None)
    p.add_argument("--pred_jsonl", type=str, default=None)

    p.add_argument("--progress_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_json", type=str, default=None)
    return p.parse_args()


def parse_dataset_config(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = raw.strip()
    if value == "" or value.lower() in {"none", "null"}:
        return None
    return value


def parse_subjects(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    out = {x.strip() for x in raw.split(",") if x.strip()}
    return out or None


def load_split(*, dataset_name: str, dataset_config: str | None, split: str):
    if dataset_config is None:
        return load_dataset(dataset_name, split=split)
    return load_dataset(dataset_name, dataset_config, split=split)


def normalize_choices(raw: Any) -> list[str]:
    if isinstance(raw, (list, tuple)):
        choices = [str(x).strip() for x in raw]
    elif isinstance(raw, dict):
        values = raw.get("text", raw.get("choices"))
        if not isinstance(values, (list, tuple)):
            raise ValueError("choices dict must contain list field 'text' or 'choices'")
        choices = [str(x).strip() for x in values]
    else:
        raise ValueError(f"unsupported choices type: {type(raw).__name__}")

    choices = [x for x in choices if x != ""]
    if len(choices) < 2:
        raise ValueError("choices must contain at least 2 options")
    if len(choices) > len(CHOICE_LABELS):
        raise ValueError("choices count exceeds supported labels")
    return choices


def parse_answer_index(raw: Any, choices: list[str]) -> int:
    n = len(choices)
    if isinstance(raw, bool):
        raise ValueError("boolean answer is invalid")

    if isinstance(raw, int):
        idx = int(raw)
        if 0 <= idx < n:
            return idx
        raise ValueError("answer index out of range")

    if isinstance(raw, float):
        if raw.is_integer():
            idx = int(raw)
            if 0 <= idx < n:
                return idx
        raise ValueError("answer float is invalid")

    txt = str(raw).strip()
    if txt == "":
        raise ValueError("empty answer")

    if txt.isdigit():
        idx = int(txt)
        if 0 <= idx < n:
            return idx
        raise ValueError("answer index out of range")

    up = txt.upper()
    if len(up) == 1 and up in CHOICE_LABELS[:n]:
        return CHOICE_LABELS.index(up)

    for i, choice in enumerate(choices):
        if txt == choice:
            return i

    raise ValueError(f"cannot parse answer value: {raw!r}")


def parse_items(
    *,
    ds,
    question_column: str,
    choices_column: str,
    answer_column: str,
    subject_column: str,
    default_subject: str,
    subject_filter: set[str] | None,
) -> list[MCQItem]:
    items: list[MCQItem] = []
    dropped = 0

    for row in ds:
        try:
            question = str(row[question_column]).strip()
            choices = normalize_choices(row[choices_column])
            answer_idx = parse_answer_index(row[answer_column], choices)
            subject = str(row.get(subject_column, default_subject)).strip() or default_subject
            if subject_filter is not None and subject not in subject_filter:
                continue
            if question == "":
                raise ValueError("empty question")
            items.append(
                MCQItem(
                    question=question,
                    choices=choices,
                    answer_idx=answer_idx,
                    subject=subject,
                )
            )
        except Exception:
            dropped += 1

    if dropped > 0:
        print(f"[warn] dropped {dropped} invalid rows while parsing dataset")
    return items


def select_eval_items(
    *,
    items: list[MCQItem],
    max_samples_per_subject: int,
    samples: int,
    seed: int,
) -> list[MCQItem]:
    rng = random.Random(seed)
    if max_samples_per_subject <= 0:
        selected = list(items)
    else:
        groups: dict[str, list[MCQItem]] = defaultdict(list)
        for item in items:
            groups[item.subject].append(item)
        selected = []
        for subject in sorted(groups):
            rows = list(groups[subject])
            rng.shuffle(rows)
            selected.extend(rows[: int(max_samples_per_subject)])

    if samples > 0 and len(selected) > samples:
        rng.shuffle(selected)
        selected = selected[: int(samples)]
    return selected


def build_support_pool(*, items: list[MCQItem], n_shot: int, seed: int) -> dict[str, list[MCQItem]]:
    groups: dict[str, list[MCQItem]] = defaultdict(list)
    for item in items:
        groups[item.subject].append(item)

    pool: dict[str, list[MCQItem]] = {}
    for offset, subject in enumerate(sorted(groups)):
        rows = list(groups[subject])
        random.Random(seed + offset).shuffle(rows)
        if n_shot > 0:
            pool[subject] = rows[: int(n_shot)]
        else:
            pool[subject] = []
    return pool


def subject_display_name(subject: str) -> str:
    return subject.replace("_", " ").strip()


def format_question(item: MCQItem, answer_label: str | None) -> str:
    lines = [item.question]
    for i, choice in enumerate(item.choices):
        lines.append(f"{CHOICE_LABELS[i]}. {choice}")
    if answer_label is None:
        lines.append("Answer (single uppercase letter only):")
    else:
        lines.append(f"Answer: {answer_label}")
    return "\n".join(lines)


def build_prompt(*, subject: str, support_examples: list[MCQItem], query_item: MCQItem) -> str:
    lines = [
        (
            "The following are multiple choice questions (with answers) "
            f"about {subject_display_name(subject)}."
        ),
        "Respond with only one uppercase letter from the provided options.",
        "",
    ]
    for ex in support_examples:
        lines.append(format_question(ex, CHOICE_LABELS[ex.answer_idx]))
        lines.append("")
    lines.append(format_question(query_item, None))
    return "\n".join(lines)


def prepare_prompts(
    *,
    eval_items: list[MCQItem],
    support_pool: dict[str, list[MCQItem]],
    tokenizer,
    n_shot: int,
    max_length: int,
    max_new_tokens: int,
    fewshot_split: str,
    eval_split: str,
    progress_every: int,
) -> tuple[list[str], list[PreparedSample], int]:
    prompts: list[str] = []
    prepared: list[PreparedSample] = []
    skipped_too_long = 0
    answer_budget = max(1, int(max_new_tokens))
    progress_step = max(1, int(progress_every))

    for i, item in enumerate(eval_items, start=1):
        labels = CHOICE_LABELS[: len(item.choices)]
        support = list(support_pool.get(item.subject, []))
        if fewshot_split == eval_split:
            support = [x for x in support if x.question != item.question]

        used_shots = min(len(support), max(0, int(n_shot)))
        chosen_prompt: str | None = None
        chosen_shots = -1
        while used_shots >= 0:
            prompt = build_prompt(
                subject=item.subject,
                support_examples=support[:used_shots],
                query_item=item,
            )
            prompt_len = int(
                tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.shape[1]
            )
            if prompt_len > 0 and (prompt_len + answer_budget) <= int(max_length):
                chosen_prompt = prompt
                chosen_shots = used_shots
                break
            used_shots -= 1

        if chosen_prompt is None:
            skipped_too_long += 1
            continue

        prompts.append(chosen_prompt)
        prepared.append(
            PreparedSample(
                subject=item.subject,
                labels=labels,
                answer_idx=item.answer_idx,
                used_shots=int(chosen_shots),
            )
        )

        if i % progress_step == 0:
            print(
                f"[progress] prepared {i}/{len(eval_items)}, "
                f"ready={len(prepared)}, skipped_too_long={skipped_too_long}"
            )

    return prompts, prepared, skipped_too_long


def write_prompt_jsonl(path: Path, prompts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for prompt in prompts:
            fout.write(json.dumps({"prompt": prompt}, ensure_ascii=False))
            fout.write("\n")
    print(f"[ok] wrote prompt jsonl rows={len(prompts)} path={path}")


def load_predictions(path: Path, expected_rows: int) -> list[str]:
    out: list[str | None] = [None] * int(expected_rows)
    seen = 0
    with path.open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"line {line_no} in {path} is not a JSON object")
            if "index" not in obj:
                raise ValueError(f"line {line_no} in {path} missing key 'index'")
            idx = int(obj["index"])
            if idx < 0 or idx >= int(expected_rows):
                raise ValueError(f"line {line_no} in {path} has out-of-range index {idx}")
            if out[idx] is not None:
                raise ValueError(f"line {line_no} in {path} duplicated index {idx}")
            out[idx] = str(obj.get("result", ""))
            seen += 1

    if seen != int(expected_rows):
        raise ValueError(f"prediction rows mismatch: expected {expected_rows}, found {seen}")
    return [str(x) for x in out]


def parse_predicted_label(text: str, labels: str) -> int | None:
    for ch in text.strip().upper():
        if ch in labels:
            return labels.index(ch)
    return None


def mean(xs: list[float]) -> float:
    if not xs:
        return 0.0
    return float(sum(xs) / len(xs))


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    dataset_config = parse_dataset_config(args.dataset_config)
    subject_filter = parse_subjects(args.subjects)
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

    default_subject = dataset_config if dataset_config is not None else "default"
    fewshot_ds = load_split(
        dataset_name=args.dataset_name,
        dataset_config=dataset_config,
        split=args.fewshot_split,
    )
    eval_ds = load_split(
        dataset_name=args.dataset_name,
        dataset_config=dataset_config,
        split=args.eval_split,
    )

    fewshot_items = parse_items(
        ds=fewshot_ds,
        question_column=args.question_column,
        choices_column=args.choices_column,
        answer_column=args.answer_column,
        subject_column=args.subject_column,
        default_subject=default_subject,
        subject_filter=subject_filter,
    )
    eval_items = parse_items(
        ds=eval_ds,
        question_column=args.question_column,
        choices_column=args.choices_column,
        answer_column=args.answer_column,
        subject_column=args.subject_column,
        default_subject=default_subject,
        subject_filter=subject_filter,
    )

    if not eval_items:
        raise ValueError("no valid eval rows after parsing/filtering")
    if int(args.n_shot) > 0 and not fewshot_items:
        print("[warn] few-shot split has no usable rows; fallback to zero-shot")

    eval_items = select_eval_items(
        items=eval_items,
        max_samples_per_subject=int(args.max_samples_per_subject),
        samples=int(args.samples),
        seed=int(args.seed),
    )
    if not eval_items:
        raise ValueError("no eval rows remain after sample limits")

    support_pool = build_support_pool(
        items=fewshot_items,
        n_shot=max(0, int(args.n_shot)),
        seed=int(args.seed),
    )

    max_in_flight = None if int(args.jsonl_max_in_flight) <= 0 else int(args.jsonl_max_in_flight)
    progress_every = max(1, int(args.progress_every))

    print(f"[info] benchmark=mmlu mode=local_split codec={codec.name}")
    print(
        f"[info] device={runtime.device}, dtype={runtime.dtype}, "
        f"decoding={args.decoding}, max_new_tokens={int(args.max_new_tokens)}"
    )
    print(
        "[info] dataset="
        f"{args.dataset_name}/{dataset_config if dataset_config is not None else '<none>'} "
        f"fewshot_split={args.fewshot_split} eval_split={args.eval_split}"
    )
    print(
        f"[info] eval_rows={len(eval_items)} n_shot={int(args.n_shot)} "
        f"max_length={int(args.max_length)}"
    )

    t0 = time.perf_counter()
    prompts, prepared, skipped_too_long = prepare_prompts(
        eval_items=eval_items,
        support_pool=support_pool,
        tokenizer=tokenizer,
        n_shot=int(args.n_shot),
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
        fewshot_split=str(args.fewshot_split),
        eval_split=str(args.eval_split),
        progress_every=progress_every,
    )
    if not prepared:
        raise ValueError("all eval rows were skipped by max_length; increase --max_length")

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    try:
        if args.prompt_jsonl:
            prompt_path = Path(args.prompt_jsonl).expanduser()
        else:
            temp_dir = tempfile.TemporaryDirectory(prefix="splitllm_mmlu_")
            prompt_path = Path(temp_dir.name) / "prompts.jsonl"

        if args.pred_jsonl:
            pred_path = Path(args.pred_jsonl).expanduser()
            if temp_dir is None:
                temp_dir = tempfile.TemporaryDirectory(prefix="splitllm_mmlu_")
        else:
            if temp_dir is None:
                temp_dir = tempfile.TemporaryDirectory(prefix="splitllm_mmlu_")
            pred_path = Path(temp_dir.name) / "predictions.jsonl"

        write_prompt_jsonl(prompt_path, prompts)

        model = SplitLLMModel(mode="local_split", tokenizer=tokenizer, runtime=runtime)
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
        generation_kwargs["use_chat_template"] = False
        generation_kwargs["add_generation_prompt"] = False
        generation_kwargs["skip_special_tokens"] = True

        generated_rows = generate_jsonl_with_model(
            model=model,
            input_path=prompt_path,
            output_path=pred_path,
            generation_kwargs=generation_kwargs,
            prompt_key="prompt",
            result_key="result",
            index_key="index",
            num_workers=int(args.jsonl_num_workers),
            max_in_flight=max_in_flight,
            progress_every=progress_every,
            keep_index=True,
        )
        if generated_rows != len(prepared):
            raise RuntimeError(
                f"generated rows mismatch: expected {len(prepared)}, got {generated_rows}"
            )

        predictions = load_predictions(pred_path, expected_rows=len(prepared))
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    correct = 0
    evaluated = 0
    skipped_invalid_answer = 0
    used_shots: list[float] = []
    subject_total: dict[str, int] = defaultdict(int)
    subject_correct: dict[str, int] = defaultdict(int)

    for i, meta in enumerate(prepared, start=1):
        pred_idx = parse_predicted_label(predictions[i - 1], meta.labels)
        if pred_idx is None:
            skipped_invalid_answer += 1
            continue

        ok = pred_idx == meta.answer_idx
        evaluated += 1
        correct += int(ok)
        used_shots.append(float(meta.used_shots))
        subject_total[meta.subject] += 1
        subject_correct[meta.subject] += int(ok)

        if i % progress_every == 0:
            running_acc = float(correct / max(1, evaluated))
            print(
                f"[progress] scored {i}/{len(prepared)}, "
                f"evaluated={evaluated}, acc={running_acc:.4f}, "
                f"skipped_too_long={skipped_too_long}, skipped_invalid={skipped_invalid_answer}"
            )

    if evaluated <= 0:
        raise ValueError("all generated rows were invalid during answer parsing")

    elapsed_sec = time.perf_counter() - t0
    accuracy = float(correct / max(1, evaluated))
    samples_per_sec = float(evaluated / elapsed_sec) if elapsed_sec > 0 else 0.0

    per_subject = []
    for subject in sorted(subject_total):
        total = int(subject_total[subject])
        c = int(subject_correct.get(subject, 0))
        per_subject.append(
            {
                "subject": subject,
                "correct": c,
                "total": total,
                "accuracy": float(c / max(1, total)),
            }
        )

    result: dict[str, Any] = {
        "benchmark": {
            "name": "mmlu",
            "mode": "local_split",
            "scoring": "jsonl_generate_then_single_letter_parse",
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
            "config": dataset_config,
            "fewshot_split": args.fewshot_split,
            "eval_split": args.eval_split,
            "question_column": args.question_column,
            "choices_column": args.choices_column,
            "answer_column": args.answer_column,
            "subject_column": args.subject_column,
            "subjects": sorted(subject_filter) if subject_filter is not None else None,
        },
        "runtime": {
            "device": str(runtime.device),
            "dtype": str(runtime.dtype),
            "seed": int(args.seed),
            "autocast": False,
        },
        "eval": {
            "samples_selected": int(len(eval_items)),
            "samples_prepared": int(len(prepared)),
            "samples_evaluated": int(evaluated),
            "samples_skipped_too_long": int(skipped_too_long),
            "samples_skipped_invalid_answer": int(skipped_invalid_answer),
            "samples_limit": int(args.samples),
            "max_samples_per_subject": int(args.max_samples_per_subject),
            "n_shot_requested": int(args.n_shot),
            "n_shot_used_mean": float(mean(used_shots)),
            "max_length": int(args.max_length),
            "decoding": str(args.decoding),
            "max_new_tokens": int(args.max_new_tokens),
            "correct": int(correct),
            "accuracy": float(accuracy),
            "elapsed_sec": float(elapsed_sec),
            "samples_per_sec": float(samples_per_sec),
            "num_subjects_evaluated": int(len(per_subject)),
            "per_subject": per_subject,
        },
    }

    print("\n[result]")
    print(
        f"Evaluated: {evaluated}/{len(prepared)} "
        f"(skipped_too_long={skipped_too_long}, skipped_invalid={skipped_invalid_answer})"
    )
    print(f"Accuracy: {accuracy:.6f} ({correct}/{max(1, evaluated)})")
    print(f"Subjects evaluated: {len(per_subject)}")
    print(f"Elapsed: {elapsed_sec:.3f}s, samples/s: {samples_per_sec:.3f}")

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
