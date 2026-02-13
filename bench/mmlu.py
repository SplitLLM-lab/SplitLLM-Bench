from __future__ import annotations

import argparse
import json
import random
import string
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from model import LocalSplitRuntime, SamplingConfig

from bench.utils import build_codec, parse_codec_extras


CHOICE_LABELS = string.ascii_uppercase


@dataclass
class MCQItem:
    question: str
    choices: list[str]
    answer_idx: int
    subject: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark MMLU multiple-choice accuracy for local split front/back checkpoints "
            "using greedy single-letter answer generation (A/B/C/...)."
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
    p.add_argument(
        "--n_shot",
        type=int,
        default=5,
        help="Few-shot examples per subject (default: 5).",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=-1,
        help="Global max eval samples after filtering. <=0 means all.",
    )
    p.add_argument(
        "--max_samples_per_subject",
        type=int,
        default=-1,
        help="Per-subject max eval samples. <=0 means no per-subject cap.",
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Max sequence length for prompt + answer label token(s).",
    )

    p.add_argument("--progress_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_json", type=str, default=None)
    return p.parse_args()


def parse_subjects(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    out = {x.strip() for x in raw.split(",") if x.strip()}
    if not out:
        return None
    return out


def parse_dataset_config(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = raw.strip()
    if value == "" or value.lower() in {"none", "null"}:
        return None
    return value


def load_split(
    *,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
):
    if dataset_config is None:
        return load_dataset(dataset_name, split=split)
    return load_dataset(dataset_name, dataset_config, split=split)


def normalize_choices(raw: Any) -> list[str]:
    if isinstance(raw, (list, tuple)):
        choices = [str(x).strip() for x in raw]
    elif isinstance(raw, dict):
        if "text" in raw and isinstance(raw["text"], (list, tuple)):
            choices = [str(x).strip() for x in raw["text"]]
        elif "choices" in raw and isinstance(raw["choices"], (list, tuple)):
            choices = [str(x).strip() for x in raw["choices"]]
        else:
            raise ValueError("choices dict must contain list field 'text' or 'choices'")
    else:
        raise ValueError(f"unsupported choices type: {type(raw).__name__}")

    choices = [x for x in choices if x != ""]
    if len(choices) < 2:
        raise ValueError("choices must contain at least 2 options")
    if len(choices) > len(CHOICE_LABELS):
        raise ValueError(
            f"choices count {len(choices)} exceeds supported labels {len(CHOICE_LABELS)}"
        )
    return choices


def answer_to_index(raw: Any, choices: list[str]) -> int:
    n = len(choices)
    if isinstance(raw, bool):
        raise ValueError("boolean answer is not valid")

    if isinstance(raw, int):
        idx = int(raw)
        if 0 <= idx < n:
            return idx
        raise ValueError(f"answer index {idx} out of range for {n} choices")

    if isinstance(raw, float):
        if raw.is_integer():
            idx = int(raw)
            if 0 <= idx < n:
                return idx
        raise ValueError(f"answer float {raw} is invalid for {n} choices")

    txt = str(raw).strip()
    if txt == "":
        raise ValueError("empty answer")

    if txt.isdigit():
        idx = int(txt)
        if 0 <= idx < n:
            return idx
        raise ValueError(f"answer index {idx} out of range for {n} choices")

    up = txt.upper()
    if len(up) == 1 and up in CHOICE_LABELS[:n]:
        return CHOICE_LABELS.index(up)

    for i, choice in enumerate(choices):
        if txt == choice:
            return i

    raise ValueError(f"cannot parse answer value {raw!r}")


def to_items(
    *,
    ds,
    question_column: str,
    choices_column: str,
    answer_column: str,
    subject_column: str,
    default_subject: str,
    requested_subjects: set[str] | None,
) -> list[MCQItem]:
    items: list[MCQItem] = []
    dropped = 0

    for row in ds:
        try:
            if question_column not in row:
                raise ValueError(f"missing question column: {question_column!r}")
            if choices_column not in row:
                raise ValueError(f"missing choices column: {choices_column!r}")
            if answer_column not in row:
                raise ValueError(f"missing answer column: {answer_column!r}")

            question = str(row.get(question_column, "")).strip()
            if question == "":
                raise ValueError("empty question")

            choices = normalize_choices(row.get(choices_column))
            answer_idx = answer_to_index(row.get(answer_column), choices)

            subject = str(row.get(subject_column, default_subject)).strip()
            if subject == "":
                subject = default_subject

            if requested_subjects is not None and subject not in requested_subjects:
                continue

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


def make_subject_groups(items: list[MCQItem]) -> dict[str, list[MCQItem]]:
    groups: dict[str, list[MCQItem]] = defaultdict(list)
    for item in items:
        groups[item.subject].append(item)
    return groups


def build_support_pool(
    *,
    items: list[MCQItem],
    n_shot: int,
    seed: int,
) -> dict[str, list[MCQItem]]:
    groups = make_subject_groups(items)
    pool: dict[str, list[MCQItem]] = {}
    if n_shot <= 0:
        for subject in groups:
            pool[subject] = []
        return pool

    for offset, subject in enumerate(sorted(groups)):
        rows = list(groups[subject])
        rng = random.Random(seed + offset)
        rng.shuffle(rows)
        pool[subject] = rows[: int(n_shot)]
    return pool


def select_eval_items(
    *,
    items: list[MCQItem],
    max_samples_per_subject: int,
    samples: int,
    seed: int,
) -> list[MCQItem]:
    rng = random.Random(seed)
    selected: list[MCQItem] = []

    if max_samples_per_subject > 0:
        groups = make_subject_groups(items)
        for subject in sorted(groups):
            rows = list(groups[subject])
            rng.shuffle(rows)
            selected.extend(rows[: int(max_samples_per_subject)])
    else:
        selected = list(items)

    if samples > 0 and len(selected) > samples:
        rng.shuffle(selected)
        selected = selected[: int(samples)]

    return selected


def subject_display_name(subject: str) -> str:
    return subject.replace("_", " ").strip()


def format_question_block(item: MCQItem, answer_label: str | None) -> str:
    lines = [item.question]
    for i, choice in enumerate(item.choices):
        lines.append(f"{CHOICE_LABELS[i]}. {choice}")
    if answer_label is None:
        lines.append("Answer (single uppercase letter only):")
    else:
        lines.append(f"Answer: {answer_label}")
    return "\n".join(lines)


def build_prompt(
    *,
    subject: str,
    support_examples: list[MCQItem],
    query_item: MCQItem,
) -> str:
    lines = [
        (
            "The following are multiple choice questions (with answers) "
            f"about {subject_display_name(subject)}."
        ),
        "Respond with only one uppercase letter from the provided options.",
        "",
    ]
    for ex in support_examples:
        lines.append(format_question_block(ex, CHOICE_LABELS[ex.answer_idx]))
        lines.append("")
    lines.append(format_question_block(query_item, None))
    return "\n".join(lines)


def build_label_token_first_ids(tokenizer) -> dict[str, set[int]]:
    ids_map: dict[str, set[int]] = {}
    for label in CHOICE_LABELS:
        ids: set[int] = set()
        for text in (f" {label}", label):
            token_ids = tokenizer(text, add_special_tokens=False).input_ids
            if token_ids:
                ids.add(int(token_ids[0]))
        if not ids:
            raise ValueError(
                f"tokenizer returned empty ids for answer label {label!r}"
            )
        ids_map[label] = ids
    return ids_map


def parse_generated_label(text: str, labels: str) -> int | None:
    for ch in text.strip().upper():
        if ch in labels:
            return labels.index(ch)
    return None


@torch.no_grad()
def greedy_predict_choice(
    *,
    runtime: LocalSplitRuntime,
    tokenizer,
    prefix_ids: torch.Tensor,
    labels: str,
    label_token_first_ids: dict[str, set[int]],
    stop_token_ids: set[int],
    sampling: SamplingConfig,
    codec_extras: dict[str, Any],
) -> int | None:
    attention_mask = torch.ones_like(prefix_ids)
    result = runtime.generate_from_ids(
        input_ids=prefix_ids,
        attention_mask=attention_mask,
        max_new_tokens=1,
        eos_token_id=(
            int(tokenizer.eos_token_id)
            if tokenizer.eos_token_id is not None
            else None
        ),
        stop_token_ids=stop_token_ids,
        sampling=sampling,
        codec_extras=codec_extras,
    )
    if not result.generated_token_ids:
        return None

    token_id = int(result.generated_token_ids[0])
    decoded = tokenizer.decode([token_id], skip_special_tokens=True)
    parsed = parse_generated_label(decoded, labels)
    if parsed is not None:
        return parsed

    for i, label in enumerate(labels):
        if token_id in label_token_first_ids.get(label, set()):
            return i
    return None


def mean(xs: list[float]) -> float:
    if not xs:
        return 0.0
    return float(sum(xs) / len(xs))


def run(args: argparse.Namespace) -> Dict[str, Any]:
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    dataset_config = parse_dataset_config(args.dataset_config)
    requested_subjects = parse_subjects(args.subjects)
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
    ds_fewshot = load_split(
        dataset_name=args.dataset_name,
        dataset_config=dataset_config,
        split=args.fewshot_split,
    )
    ds_eval = load_split(
        dataset_name=args.dataset_name,
        dataset_config=dataset_config,
        split=args.eval_split,
    )

    fewshot_items = to_items(
        ds=ds_fewshot,
        question_column=args.question_column,
        choices_column=args.choices_column,
        answer_column=args.answer_column,
        subject_column=args.subject_column,
        default_subject=default_subject,
        requested_subjects=requested_subjects,
    )
    eval_items = to_items(
        ds=ds_eval,
        question_column=args.question_column,
        choices_column=args.choices_column,
        answer_column=args.answer_column,
        subject_column=args.subject_column,
        default_subject=default_subject,
        requested_subjects=requested_subjects,
    )

    if not eval_items:
        raise ValueError(
            "no valid eval rows after parsing/filtering; check dataset/split/columns/subjects"
        )
    if int(args.n_shot) > 0 and not fewshot_items:
        print("[warn] few-shot split has no usable rows; benchmark falls back to zero-shot")

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
    label_token_first_ids = build_label_token_first_ids(tokenizer)
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
    sampling = SamplingConfig(do_sample=False)
    progress_every = max(1, int(args.progress_every))

    print(f"[info] benchmark=mmlu mode=local_split codec={codec.name}")
    print(f"[info] device={runtime.device}, dtype={runtime.dtype}, decoding=greedy")
    print(
        "[info] dataset="
        f"{args.dataset_name}/{dataset_config if dataset_config is not None else '<none>'} "
        f"fewshot_split={args.fewshot_split} eval_split={args.eval_split}"
    )
    print(
        f"[info] eval_rows={len(eval_items)} n_shot={int(args.n_shot)} "
        f"max_length={int(args.max_length)}"
    )

    correct = 0
    evaluated = 0
    skipped_too_long = 0
    skipped_invalid_answer = 0
    used_shots_all: list[float] = []
    subject_total: dict[str, int] = defaultdict(int)
    subject_correct: dict[str, int] = defaultdict(int)

    t0 = time.perf_counter()

    for i, item in enumerate(eval_items):
        labels = CHOICE_LABELS[: len(item.choices)]
        max_answer_tokens = 1
        support = list(support_pool.get(item.subject, []))
        if args.fewshot_split == args.eval_split:
            support = [x for x in support if x.question != item.question]

        used_shots = min(len(support), max(0, int(args.n_shot)))
        prefix_ids: torch.Tensor | None = None
        while used_shots >= 0:
            prompt = build_prompt(
                subject=item.subject,
                support_examples=support[:used_shots],
                query_item=item,
            )
            encoded = tokenizer(
                prompt,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids.to(runtime.device)
            if int(encoded.shape[1]) <= 0:
                break
            if int(encoded.shape[1]) + max_answer_tokens <= int(args.max_length):
                prefix_ids = encoded
                break
            used_shots -= 1

        if prefix_ids is None or used_shots < 0:
            skipped_too_long += 1
            continue

        pred_idx = greedy_predict_choice(
            runtime=runtime,
            tokenizer=tokenizer,
            prefix_ids=prefix_ids,
            labels=labels,
            label_token_first_ids=label_token_first_ids,
            stop_token_ids=stop_token_ids,
            sampling=sampling,
            codec_extras=codec_extras,
        )
        if pred_idx is None:
            skipped_invalid_answer += 1
            continue

        is_correct = pred_idx == item.answer_idx

        evaluated += 1
        correct += int(is_correct)
        used_shots_all.append(float(used_shots))
        subject_total[item.subject] += 1
        subject_correct[item.subject] += int(is_correct)

        if ((i + 1) % progress_every) == 0:
            running_acc = float(correct / max(1, evaluated))
            print(
                f"[progress] processed {i + 1}/{len(eval_items)}, "
                f"evaluated={evaluated}, acc={running_acc:.4f}, "
                f"skipped_too_long={skipped_too_long}, skipped_invalid={skipped_invalid_answer}"
            )

    if evaluated <= 0:
        raise ValueError(
            "all selected eval rows were skipped (max_length too small or invalid generated answers); "
            "increase --max_length or inspect answer parsing"
        )

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

    result: Dict[str, Any] = {
        "benchmark": {
            "name": "mmlu",
            "mode": "local_split",
            "scoring": "greedy_single_letter",
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
            "subjects": sorted(requested_subjects) if requested_subjects is not None else None,
        },
        "runtime": {
            "device": str(runtime.device),
            "dtype": str(runtime.dtype),
            "seed": int(args.seed),
            "autocast": False,
        },
        "eval": {
            "samples_selected": int(len(eval_items)),
            "samples_evaluated": int(evaluated),
            "samples_skipped_too_long": int(skipped_too_long),
            "samples_skipped_invalid_answer": int(skipped_invalid_answer),
            "samples_limit": int(args.samples),
            "max_samples_per_subject": int(args.max_samples_per_subject),
            "n_shot_requested": int(args.n_shot),
            "n_shot_used_mean": float(mean(used_shots_all)),
            "max_length": int(args.max_length),
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
        f"Evaluated: {evaluated}/{len(eval_items)} "
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
