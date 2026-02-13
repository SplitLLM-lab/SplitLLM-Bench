from __future__ import annotations

from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import importlib
import inspect
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TextIO

import torch
from datasets import load_dataset

from model import DefaultCodec, IdentityActivationCodec
from model.codec import ActivationCodec, CodecContext, EncodedActivation


@dataclass
class CodecTiming:
    encode_ms: float = 0.0
    decode_ms: float = 0.0


def parse_codec_extras(raw: str | None) -> dict[str, Any]:
    if raw is None:
        return {}
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("--codec_extras_json must decode to JSON object")
    return dict(obj)


def build_codec(name: str) -> ActivationCodec:
    builtin: dict[str, ActivationCodec] = {
        "default": DefaultCodec(),
        "identity_fp32": IdentityActivationCodec(),
    }
    if name in builtin:
        return builtin[name]
    return _load_custom_codec(name)


def _codec_from_object(obj: Any, spec: str) -> ActivationCodec:
    if isinstance(obj, ActivationCodec):
        return obj

    if inspect.isclass(obj):
        if not issubclass(obj, ActivationCodec):
            raise TypeError(
                f"{spec!r} resolved to class {obj.__name__}, "
                "but it is not a subclass of ActivationCodec"
            )
        return obj()

    if callable(obj):
        built = obj()
        if not isinstance(built, ActivationCodec):
            raise TypeError(
                f"{spec!r} callable returned {type(built).__name__}, "
                "expected ActivationCodec"
            )
        return built

    raise TypeError(
        f"{spec!r} resolved to unsupported object type: {type(obj).__name__}. "
        "Expected ActivationCodec instance/class or callable returning one."
    )


def _codec_from_module(module: Any, spec: str) -> ActivationCodec:
    if hasattr(module, "build_codec"):
        return _codec_from_object(getattr(module, "build_codec"), f"{spec}:build_codec")

    for attr in ("codec", "CODEC"):
        if hasattr(module, attr):
            return _codec_from_object(getattr(module, attr), f"{spec}:{attr}")

    classes = [
        v
        for v in vars(module).values()
        if inspect.isclass(v) and issubclass(v, ActivationCodec) and v is not ActivationCodec
    ]
    if len(classes) == 1:
        return classes[0]()

    raise ValueError(
        f"cannot build codec from module {spec!r}. "
        "Expected one of: build_codec(), codec, CODEC, or exactly one "
        "ActivationCodec subclass."
    )


def _load_custom_codec(spec: str) -> ActivationCodec:
    if ":" in spec:
        module_name, attr = spec.split(":", 1)
        module_name = module_name.strip()
        attr = attr.strip()
        if not module_name or not attr:
            raise ValueError(
                f"invalid codec spec {spec!r}; expected format module:attr"
            )
        module = importlib.import_module(module_name)
        if not hasattr(module, attr):
            raise ValueError(
                f"module {module_name!r} has no attribute {attr!r} for codec spec {spec!r}"
            )
        return _codec_from_object(getattr(module, attr), spec)

    try:
        module = importlib.import_module(spec)
    except ModuleNotFoundError as exc:
        if exc.name != spec:
            raise
        if "." not in spec:
            raise ValueError(
                f"unsupported codec: {spec!r}. "
                "Use builtins [default, identity_fp32] or custom module path."
            )
        module_name, attr = spec.rsplit(".", 1)
        module = importlib.import_module(module_name)
        if not hasattr(module, attr):
            raise ValueError(
                f"module {module_name!r} has no attribute {attr!r} for codec spec {spec!r}"
            )
        return _codec_from_object(getattr(module, attr), spec)

    return _codec_from_module(module, spec)


def load_texts(
    *,
    dataset_name: str,
    dataset_config: str,
    split: str,
    text_column: str,
    samples: int,
) -> list[str]:
    ds = load_dataset(dataset_name, dataset_config, split=split)
    if text_column not in ds.column_names:
        cols = ", ".join(ds.column_names)
        raise ValueError(f"text_column {text_column!r} not in dataset columns: {cols}")

    ds = ds.filter(
        lambda ex: ex.get(text_column) is not None and len(str(ex[text_column]).strip()) > 0
    )
    n = min(int(samples), len(ds))
    return [str(x) for x in ds.select(range(n))[text_column]]


def safe_exp(x: float) -> float:
    if x >= 80:
        return float("inf")
    return float(math.exp(x))


def apply_codec(
    *,
    codec: ActivationCodec,
    hidden: torch.Tensor,
    context: CodecContext,
    device: torch.device,
    dtype: torch.dtype,
    timing: CodecTiming | None = None,
) -> torch.Tensor:
    t0 = time.perf_counter()
    encoded: EncodedActivation = codec.encode(hidden, context=context)
    if timing is not None:
        timing.encode_ms += (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    decoded = codec.decode(
        encoded,
        context=context,
        device=device,
        dtype=dtype,
    )
    if timing is not None:
        timing.decode_ms += (time.perf_counter() - t0) * 1000.0
    return decoded


def build_generation_kwargs(
    *,
    decoding: str = "greedy",
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    min_new_tokens: int = 0,
    no_repeat_ngram_size: int = 0,
    repetition_penalty: float = 1.0,
) -> dict[str, Any]:
    mode = str(decoding).strip().lower()
    if mode not in {"greedy", "top_k", "top_p"}:
        raise ValueError(
            f"unsupported decoding mode {decoding!r}; choose one of: greedy, top_k, top_p"
        )
    if int(max_new_tokens) < 0:
        raise ValueError("max_new_tokens must be >= 0")
    if float(temperature) <= 0:
        raise ValueError("temperature must be > 0")
    if int(top_k) < 0:
        raise ValueError("top_k must be >= 0")
    if not (0.0 < float(top_p) <= 1.0):
        raise ValueError("top_p must be in (0, 1]")
    if int(min_new_tokens) < 0:
        raise ValueError("min_new_tokens must be >= 0")
    if int(no_repeat_ngram_size) < 0:
        raise ValueError("no_repeat_ngram_size must be >= 0")
    if float(repetition_penalty) <= 0:
        raise ValueError("repetition_penalty must be > 0")

    do_sample = mode != "greedy"
    used_top_k = int(top_k) if mode == "top_k" else 0
    used_top_p = float(top_p) if mode == "top_p" else 1.0
    if mode == "top_k" and used_top_k <= 0:
        raise ValueError("top_k mode requires top_k > 0")

    return {
        "do_sample": do_sample,
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_k": int(used_top_k),
        "top_p": float(used_top_p),
        "min_new_tokens": int(min_new_tokens),
        "no_repeat_ngram_size": int(no_repeat_ngram_size),
        "repetition_penalty": float(repetition_penalty),
    }


def _coerce_generated_text(generated: Any) -> str:
    if isinstance(generated, str):
        return generated
    if isinstance(generated, dict) and "text" in generated:
        return str(generated["text"])
    if hasattr(generated, "text"):
        return str(getattr(generated, "text"))
    raise TypeError(
        "unsupported generation output type. Expected str, dict with 'text', or object with .text"
    )


def _generate_jsonl_record(
    *,
    idx: int,
    line: str,
    line_no: int,
    prompt_key: str,
    result_key: str,
    index_key: str,
    generate_text_fn: Callable[[str], Any],
) -> dict[str, Any]:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON at input line {line_no}") from exc
    if not isinstance(row, dict):
        raise ValueError(f"line {line_no} must be a JSON object")
    if prompt_key not in row:
        raise ValueError(f"line {line_no} missing prompt key {prompt_key!r}")

    prompt = str(row.get(prompt_key, "")).strip()
    if prompt == "":
        raise ValueError(f"line {line_no} has empty prompt in key {prompt_key!r}")

    generated_text = _coerce_generated_text(generate_text_fn(prompt))
    return {
        result_key: generated_text,
        index_key: int(idx),
    }


def _flush_completed(
    *,
    pending: dict[Future[dict[str, Any]], int],
    fout: TextIO,
    wait_all: bool,
) -> int:
    if not pending:
        return 0
    done, _ = wait(
        set(pending.keys()),
        return_when=ALL_COMPLETED if wait_all else FIRST_COMPLETED,
    )
    wrote = 0
    for fut in done:
        line_no = pending.pop(fut)
        try:
            record = fut.result()
        except Exception as exc:
            raise RuntimeError(f"generation failed for input line {line_no}") from exc
        fout.write(json.dumps(record, ensure_ascii=False))
        fout.write("\n")
        fout.flush()
        wrote += 1
    return wrote


def reorder_jsonl_by_index(
    *,
    input_path: str | Path,
    output_path: str | Path,
    result_key: str = "result",
    index_key: str = "index",
    keep_index: bool = True,
    expected_rows: int | None = None,
) -> int:
    src_path = Path(input_path).expanduser()
    dst_path = Path(output_path).expanduser()
    if expected_rows is None:
        with src_path.open("r", encoding="utf-8") as fin:
            expected_rows = sum(1 for line in fin if line.strip())
    if expected_rows < 0:
        raise ValueError("expected_rows must be >= 0")

    ordered: list[str | None] = [None] * int(expected_rows)
    seen = 0
    with src_path.open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"line {line_no} in {src_path} is not a JSON object")
            if index_key not in obj:
                raise ValueError(f"line {line_no} in {src_path} missing index key {index_key!r}")
            idx = int(obj[index_key])
            if idx < 0 or idx >= int(expected_rows):
                raise ValueError(f"line {line_no} has out-of-range index {idx}")
            if ordered[idx] is not None:
                raise ValueError(f"duplicate index {idx} found at line {line_no}")

            text = str(obj.get(result_key, ""))
            if keep_index:
                out_obj = {result_key: text, index_key: idx}
            else:
                out_obj = {result_key: text}
            ordered[idx] = json.dumps(out_obj, ensure_ascii=False)
            seen += 1

    if seen != int(expected_rows):
        raise ValueError(
            f"rows mismatch while reordering: expected {expected_rows}, found {seen}"
        )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", encoding="utf-8") as fout:
        for idx, payload in enumerate(ordered):
            if payload is None:
                raise ValueError(f"missing output row for index {idx}")
            fout.write(payload)
            fout.write("\n")

    return int(expected_rows)


def generate_jsonl_with_fn(
    *,
    input_path: str | Path,
    output_path: str | Path,
    generate_text_fn: Callable[[str], Any],
    prompt_key: str = "prompt",
    result_key: str = "result",
    index_key: str = "index",
    num_workers: int = 1,
    max_in_flight: int | None = None,
    progress_every: int = 100,
    keep_index: bool = True,
) -> int:
    src_path = Path(input_path).expanduser()
    dst_path = Path(output_path).expanduser()
    tmp_unsorted_path = dst_path.with_name(f"{dst_path.name}.unsorted")

    workers = max(1, int(num_workers))
    if max_in_flight is None:
        in_flight_limit = max(1, workers * 4)
    else:
        in_flight_limit = max(1, int(max_in_flight))
    progress_interval = max(0, int(progress_every))

    if workers > 1:
        print("[warn] num_workers > 1 requires thread-safe generation function/model")
    print(
        f"[info] jsonl generation start input={src_path} output={dst_path} "
        f"workers={workers} in_flight={in_flight_limit}"
    )

    t0 = time.perf_counter()
    total_submitted = 0
    total_written = 0
    pending: dict[Future[dict[str, Any]], int] = {}

    with src_path.open("r", encoding="utf-8") as fin, tmp_unsorted_path.open(
        "w", encoding="utf-8"
    ) as fout:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for line_no, line in enumerate(fin, start=1):
                if not line.strip():
                    continue
                idx = total_submitted
                fut = pool.submit(
                    _generate_jsonl_record,
                    idx=idx,
                    line=line,
                    line_no=line_no,
                    prompt_key=prompt_key,
                    result_key=result_key,
                    index_key=index_key,
                    generate_text_fn=generate_text_fn,
                )
                pending[fut] = line_no
                total_submitted += 1

                if len(pending) >= in_flight_limit:
                    total_written += _flush_completed(
                        pending=pending,
                        fout=fout,
                        wait_all=False,
                    )
                    if progress_interval > 0 and (total_written % progress_interval) == 0:
                        print(
                            f"[progress] generated={total_written}, queued={len(pending)}"
                        )

            while pending:
                total_written += _flush_completed(
                    pending=pending,
                    fout=fout,
                    wait_all=True,
                )
                if progress_interval > 0 and (total_written % progress_interval) == 0:
                    print(f"[progress] generated={total_written}, queued=0")

    if total_written != total_submitted:
        raise RuntimeError(
            f"written rows {total_written} do not match submitted rows {total_submitted}"
        )

    print(f"[info] reordering {total_written} rows to match input order")
    rows = reorder_jsonl_by_index(
        input_path=tmp_unsorted_path,
        output_path=dst_path,
        result_key=result_key,
        index_key=index_key,
        keep_index=keep_index,
        expected_rows=total_written,
    )

    try:
        tmp_unsorted_path.unlink()
    except OSError:
        print(f"[warn] could not remove temp file: {tmp_unsorted_path}")

    elapsed = time.perf_counter() - t0
    print(
        f"[ok] jsonl generation finished rows={rows} elapsed_sec={elapsed:.3f} output={dst_path}"
    )
    return rows


def generate_jsonl_with_model(
    *,
    model: Any,
    input_path: str | Path,
    output_path: str | Path,
    generation_kwargs: dict[str, Any] | None = None,
    prompt_key: str = "prompt",
    result_key: str = "result",
    index_key: str = "index",
    num_workers: int = 1,
    max_in_flight: int | None = None,
    progress_every: int = 100,
    keep_index: bool = True,
) -> int:
    kwargs = dict(generation_kwargs or {})

    def _generate(prompt: str) -> Any:
        return model.generate(prompt=prompt, **kwargs)

    return generate_jsonl_with_fn(
        input_path=input_path,
        output_path=output_path,
        generate_text_fn=_generate,
        prompt_key=prompt_key,
        result_key=result_key,
        index_key=index_key,
        num_workers=num_workers,
        max_in_flight=max_in_flight,
        progress_every=progress_every,
        keep_index=keep_index,
    )


__all__ = [
    "CodecTiming",
    "parse_codec_extras",
    "build_codec",
    "load_texts",
    "safe_exp",
    "apply_codec",
    "build_generation_kwargs",
    "reorder_jsonl_by_index",
    "generate_jsonl_with_fn",
    "generate_jsonl_with_model",
]
