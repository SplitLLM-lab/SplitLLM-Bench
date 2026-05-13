from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from model import LocalSplitRuntime, SamplingConfig

from bench.latency import max_or_zero, mean, run_one_sample, stat_block
from bench.utils import build_codec, parse_codec_extras


PROMPT_COLUMN_CANDIDATES = ("prompt", "question", "text", "input", "query")
DEFAULT_DOWNLINK_RESPONSE_BYTES = 512


@dataclass
class PromptSample:
    text: str
    row_index: int
    token_count: int
    token_delta: int


@dataclass
class SampleTimeline:
    real_local_ms: list[float]
    simulated_ms: list[float]
    network_rounds: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run a local SplitLLM scenario benchmark from YAML: latency with "
            "post-hoc network simulation plus optional PPL and MMLU quality metrics."
        ),
    )
    p.add_argument("--scenario", type=str, required=True)
    p.add_argument("--front_dir", type=str, default="./split_out/front")
    p.add_argument("--back_dir", type=str, default="./split_out/back")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--tokenizer_id", type=str, default="Qwen/Qwen3-1.7B")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument("--front_quant", type=str, default="none")
    p.add_argument("--back_quant", type=str, default="none")
    p.add_argument(
        "--codec",
        type=str,
        default=None,
        help="Override scenario codec.name. Defaults to scenario codec.name, then default.",
    )
    p.add_argument(
        "--codec_extras_json",
        type=str,
        default=None,
        help="JSON object merged over scenario codec.extras.",
    )
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--max_prompt_length", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=0)
    p.add_argument("--dataset_scan_limit", type=int, default=5000)
    p.add_argument(
        "--downlink_response_bytes",
        type=int,
        default=DEFAULT_DOWNLINK_RESPONSE_BYTES,
        help="Small cloud-to-edge response size used by the post-hoc network model.",
    )
    p.add_argument("--skip_ppl", action="store_true")
    p.add_argument("--skip_mmlu", action="store_true")
    p.add_argument("--ppl_samples", type=int, default=0)
    p.add_argument("--mmlu_samples", type=int, default=0)
    p.add_argument("--progress_every", type=int, default=5)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out_dir", type=str, default="./bench_out/scenario")
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Parse the YAML and sample the network model without loading a model.",
    )
    return p.parse_args()


def load_scenario(path: str | Path) -> dict[str, Any]:
    scenario_path = Path(path).expanduser()
    with scenario_path.open("r", encoding="utf-8") as fin:
        data = yaml.safe_load(fin)
    if not isinstance(data, dict):
        raise ValueError("scenario YAML must contain a mapping at the top level")
    return data


def as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def parse_dataset_config(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    return text


def load_longbench_dataset(config: str | None, split: str):
    if split != "test":
        raise ValueError("THUDM/LongBench only provides split='test'")
    if config is None:
        raise ValueError("THUDM/LongBench requires workload.dataset.config")

    zip_path = hf_hub_download(
        repo_id="THUDM/LongBench",
        filename="data.zip",
        repo_type="dataset",
    )
    member = f"data/{config}.jsonl"
    print(f"[info] loading LongBench data zip={zip_path} member={member}")
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as zf:
        if member not in zf.namelist():
            raise ValueError(f"LongBench config {config!r} not found in data.zip")
        with zf.open(member) as fin:
            for line in fin:
                rows.append(json.loads(line.decode("utf-8")))
    print(f"[ok] loaded LongBench config={config} rows={len(rows)}")
    return Dataset.from_list(rows)


def load_hf_dataset(dataset_cfg: dict[str, Any]):
    name = dataset_cfg.get("name")
    if not name:
        raise ValueError("workload.dataset.name is required")
    split = str(dataset_cfg.get("split", "test"))
    config = parse_dataset_config(dataset_cfg.get("config"))
    if str(name) == "THUDM/LongBench":
        return load_longbench_dataset(config, split)
    if config is None:
        return load_dataset(str(name), split=split)
    return load_dataset(str(name), config, split=split)


def detect_prompt_column(ds, workload_cfg: dict[str, Any], dataset_cfg: dict[str, Any]) -> str:
    explicit = workload_cfg.get("prompt_column", dataset_cfg.get("prompt_column"))
    if explicit:
        column = str(explicit)
        if column not in ds.column_names:
            cols = ", ".join(ds.column_names)
            raise ValueError(f"prompt_column {column!r} not in dataset columns: {cols}")
        return column

    for column in PROMPT_COLUMN_CANDIDATES:
        if column in ds.column_names:
            return column

    cols = ", ".join(ds.column_names)
    raise ValueError(
        "could not auto-detect prompt column. "
        f"Set workload.prompt_column. Available columns: {cols}"
    )


def token_len(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return int(len(ids))


def select_prompt_samples(
    *,
    scenario: dict[str, Any],
    tokenizer,
    samples: int,
    scan_limit: int,
    seed: int,
) -> tuple[list[PromptSample], dict[str, Any]]:
    workload_cfg = as_dict(scenario.get("workload"))
    dataset_cfg = as_dict(workload_cfg.get("dataset"))
    prompt_tokens_cfg = as_dict(workload_cfg.get("prompt_tokens"))
    target_tokens = prompt_tokens_cfg.get("target")
    tolerance = prompt_tokens_cfg.get("tolerance")

    ds = load_hf_dataset(dataset_cfg)
    prompt_column = detect_prompt_column(ds, workload_cfg, dataset_cfg)
    limit = len(ds) if int(scan_limit) <= 0 else min(len(ds), int(scan_limit))

    selected: list[PromptSample] = []
    seen_nonempty = 0
    for row_index in range(limit):
        value = ds[int(row_index)].get(prompt_column)
        text = str(value).strip() if value is not None else ""
        if not text:
            continue
        seen_nonempty += 1
        n_tok = token_len(tokenizer, text)
        if target_tokens is None:
            delta = 0
        else:
            delta = abs(n_tok - int(target_tokens))
        selected.append(
            PromptSample(
                text=text,
                row_index=int(row_index),
                token_count=int(n_tok),
                token_delta=int(delta),
            )
        )

    if not selected:
        raise ValueError("no non-empty prompt rows found in workload dataset")

    if target_tokens is None:
        rng = random.Random(seed)
        rng.shuffle(selected)
        selected = selected[: max(1, int(samples))]
    else:
        selected.sort(key=lambda x: (x.token_delta, x.row_index))
        selected = selected[: max(1, int(samples))]

    outside_tolerance = 0
    if tolerance is not None:
        outside_tolerance = sum(1 for x in selected if x.token_delta > int(tolerance))
        if outside_tolerance > 0:
            print(
                "[warn] selected "
                f"{outside_tolerance}/{len(selected)} prompts outside token tolerance "
                f"target={int(target_tokens)} tolerance={int(tolerance)}"
            )

    info = {
        "name": dataset_cfg.get("name"),
        "config": parse_dataset_config(dataset_cfg.get("config")),
        "split": dataset_cfg.get("split", "test"),
        "prompt_column": prompt_column,
        "rows_total": int(len(ds)),
        "rows_scanned": int(limit),
        "nonempty_rows_scanned": int(seen_nonempty),
        "samples_selected": int(len(selected)),
        "prompt_tokens": {
            "target": int(target_tokens) if target_tokens is not None else None,
            "tolerance": int(tolerance) if tolerance is not None else None,
            "outside_tolerance": int(outside_tolerance),
            "selected": [int(x.token_count) for x in selected],
        },
        "selected_row_indices": [int(x.row_index) for x in selected],
    }
    return selected, info


def merge_codec_extras(scenario: dict[str, Any], raw_override: str | None) -> dict[str, Any]:
    extras = as_dict(as_dict(scenario.get("codec")).get("extras"))
    if raw_override:
        extras.update(parse_codec_extras(raw_override))
    return extras


def codec_spec_from_config(scenario: dict[str, Any], cli_codec: str | None) -> str:
    if cli_codec:
        return str(cli_codec)
    codec_cfg = as_dict(scenario.get("codec"))
    return str(codec_cfg.get("name", "default"))


def workload_generation(scenario: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    workload_cfg = as_dict(scenario.get("workload"))
    generation_cfg = as_dict(workload_cfg.get("generation"))
    max_new_tokens = int(args.max_new_tokens or workload_cfg.get("max_new_tokens", 128))
    decoding = str(generation_cfg.get("decoding", "greedy")).strip().lower()
    if decoding not in {"greedy", "top_k", "top_p"}:
        raise ValueError("workload.generation.decoding must be greedy, top_k, or top_p")

    temperature = float(generation_cfg.get("temperature", 1.0))
    do_sample = decoding != "greedy"
    if not do_sample and temperature <= 0:
        temperature = 1.0
    if do_sample and temperature <= 0:
        raise ValueError("temperature must be > 0 for sampled decoding")

    return {
        "decoding": decoding,
        "do_sample": bool(do_sample),
        "temperature": float(temperature),
        "top_k": int(generation_cfg.get("top_k", 50 if decoding == "top_k" else 0)),
        "top_p": float(generation_cfg.get("top_p", 0.9 if decoding == "top_p" else 1.0)),
        "min_new_tokens": int(generation_cfg.get("min_new_tokens", 0)),
        "no_repeat_ngram_size": int(generation_cfg.get("no_repeat_ngram_size", 0)),
        "repetition_penalty": float(generation_cfg.get("repetition_penalty", 1.0)),
        "max_new_tokens": int(max_new_tokens),
        "stop_at_eos": bool(workload_cfg.get("stop_at_eos", True)),
    }


def sampling_from_generation(generation: dict[str, Any]) -> SamplingConfig:
    sampling = SamplingConfig(
        do_sample=bool(generation["do_sample"]),
        temperature=float(generation["temperature"]),
        top_k=int(generation["top_k"]),
        top_p=float(generation["top_p"]),
        min_new_tokens=int(generation["min_new_tokens"]),
        no_repeat_ngram_size=int(generation["no_repeat_ngram_size"]),
        repetition_penalty=float(generation["repetition_penalty"]),
    )
    sampling.validate()
    return sampling


def max_prompt_length_from_config(
    scenario: dict[str, Any],
    args: argparse.Namespace,
) -> int:
    if int(args.max_prompt_length) > 0:
        return int(args.max_prompt_length)

    workload_cfg = as_dict(scenario.get("workload"))
    prompt_tokens_cfg = as_dict(workload_cfg.get("prompt_tokens"))
    target = prompt_tokens_cfg.get("target")
    tolerance = prompt_tokens_cfg.get("tolerance", 0)
    if target is not None:
        return max(1, int(target) + max(0, int(tolerance)))
    return 256


def default_stop_ids(tokenizer, stop_at_eos: bool) -> tuple[int | None, set[int]]:
    eos_token_id = (
        int(tokenizer.eos_token_id)
        if tokenizer.eos_token_id is not None
        else None
    )
    if not stop_at_eos:
        return eos_token_id, set()

    stop_ids: set[int] = set()
    if eos_token_id is not None:
        stop_ids.add(eos_token_id)
    vocab = tokenizer.get_vocab()
    if "<|im_end|>" in vocab:
        stop_ids.add(int(vocab["<|im_end|>"]))
    return eos_token_id, stop_ids


def _number(value: Any, default: float) -> float:
    if isinstance(value, dict):
        return float(value.get("mean", default))
    if value is None:
        return float(default)
    return float(value)


def sample_rate_mbps(spec: Any, rng: random.Random, default: float) -> float:
    if isinstance(spec, dict):
        mean_value = float(spec.get("mean", default))
        jitter = float(spec.get("jitter", 0.0))
        lo = mean_value * (1.0 - jitter)
        hi = mean_value * (1.0 + jitter)
        if "min" in spec:
            lo = max(lo, float(spec["min"]))
        if "max" in spec:
            hi = min(hi, float(spec["max"]))
        if hi < lo:
            hi = lo
        value = rng.uniform(lo, hi) if jitter > 0 else mean_value
    else:
        value = _number(spec, default)
    return max(1e-6, float(value))


def sample_rtt_ms(spec: Any, rng: random.Random) -> float:
    if not isinstance(spec, dict):
        return max(0.0, float(spec or 0.0))

    mean_value = float(spec.get("mean", 0.0))
    jitter = float(spec.get("jitter", 0.0))
    base = rng.uniform(mean_value - jitter, mean_value + jitter) if jitter > 0 else mean_value
    base = max(0.0, base)

    burst_probability = float(spec.get("burst_probability", 0.0))
    if burst_probability > 0 and rng.random() < burst_probability:
        burst_cfg = as_dict(spec.get("burst_extra_ms"))
        burst_min = float(burst_cfg.get("min", 0.0))
        burst_max = float(burst_cfg.get("max", burst_min))
        if burst_max < burst_min:
            burst_max = burst_min
        base += rng.uniform(burst_min, burst_max)
    return float(base)


def transfer_ms(*, bytes_count: int, mbps: float) -> float:
    return float(bytes_count) * 8.0 / (float(mbps) * 1_000_000.0) * 1000.0


def sample_network_round(
    *,
    network_cfg: dict[str, Any],
    rng: random.Random,
    phase: str,
    wire_bytes: int,
    downlink_response_bytes: int,
) -> dict[str, Any]:
    uplink_mbps = sample_rate_mbps(network_cfg.get("uplink_mbps"), rng, default=15.0)
    downlink_mbps = sample_rate_mbps(network_cfg.get("downlink_mbps"), rng, default=80.0)
    rtt_ms = sample_rtt_ms(network_cfg.get("rtt_ms", {"mean": 0.0}), rng)
    uplink_ms = transfer_ms(bytes_count=int(wire_bytes), mbps=uplink_mbps)
    downlink_ms = transfer_ms(bytes_count=int(downlink_response_bytes), mbps=downlink_mbps)

    packet_loss = max(0.0, float(network_cfg.get("packet_loss", 0.0)))
    lost = packet_loss > 0 and rng.random() < packet_loss
    retransmit_ms = rtt_ms if lost else 0.0
    delay_ms = uplink_ms + downlink_ms + rtt_ms + retransmit_ms

    return {
        "phase": phase,
        "wire_bytes": int(wire_bytes),
        "downlink_response_bytes": int(downlink_response_bytes),
        "uplink_mbps": float(uplink_mbps),
        "downlink_mbps": float(downlink_mbps),
        "rtt_ms": float(rtt_ms),
        "uplink_ms": float(uplink_ms),
        "downlink_ms": float(downlink_ms),
        "packet_loss_event": bool(lost),
        "retransmit_ms": float(retransmit_ms),
        "delay_ms": float(delay_ms),
    }


def local_arrivals_ms(metric) -> list[float]:
    if metric.generated_tokens <= 0:
        return []
    arrivals = [float(metric.ttft_ms)]
    for step_ms in metric.decode_step_ms[: max(0, metric.generated_tokens - 1)]:
        arrivals.append(float(arrivals[-1] + step_ms))
    return arrivals


def simulate_network_timeline(
    *,
    metric,
    network_cfg: dict[str, Any],
    rng: random.Random,
    downlink_response_bytes: int,
) -> SampleTimeline:
    real = local_arrivals_ms(metric)
    if not real:
        return SampleTimeline(real_local_ms=[], simulated_ms=[], network_rounds=[])

    prefill_round = next((x for x in metric.codec_rounds if x.phase == "prefill"), None)
    decode_rounds = [x for x in metric.codec_rounds if x.phase == "decode"]
    network_rounds: list[dict[str, Any]] = []

    prefill_wire = int(prefill_round.wire_bytes) if prefill_round is not None else 0
    prefill_net = sample_network_round(
        network_cfg=network_cfg,
        rng=rng,
        phase="prefill",
        wire_bytes=prefill_wire,
        downlink_response_bytes=downlink_response_bytes,
    )
    network_rounds.append(prefill_net)
    simulated = [float(metric.ttft_ms + prefill_net["delay_ms"])]

    for i in range(1, metric.generated_tokens):
        round_metric = decode_rounds[i - 1] if (i - 1) < len(decode_rounds) else None
        wire_bytes = int(round_metric.wire_bytes) if round_metric is not None else 0
        net = sample_network_round(
            network_cfg=network_cfg,
            rng=rng,
            phase="decode",
            wire_bytes=wire_bytes,
            downlink_response_bytes=downlink_response_bytes,
        )
        network_rounds.append(net)
        step_ms = float(metric.decode_step_ms[i - 1]) if (i - 1) < len(metric.decode_step_ms) else 0.0
        simulated.append(float(simulated[-1] + step_ms + net["delay_ms"]))

    return SampleTimeline(
        real_local_ms=real,
        simulated_ms=simulated,
        network_rounds=network_rounds,
    )


def flatten(xs: list[list[float]]) -> list[float]:
    return [float(v) for row in xs for v in row]


def inter_token_latencies_ms(arrivals: list[list[float]]) -> list[float]:
    out: list[float] = []
    for row in arrivals:
        for i in range(1, len(row)):
            out.append(float(row[i] - row[i - 1]))
    return out


def last_values(arrivals: list[list[float]]) -> list[float]:
    return [float(row[-1]) for row in arrivals if row]


def first_values(arrivals: list[list[float]]) -> list[float]:
    return [float(row[0]) for row in arrivals if row]


def mean_by_token(arrivals: list[list[float]], token_index: int) -> float | None:
    vals = [row[token_index - 1] for row in arrivals if len(row) >= token_index]
    if not vals:
        return None
    return float(mean(vals))


def target_time_s(targets: dict[str, Any], token_index: int, max_new_tokens: int) -> float | None:
    ttft_s = targets.get("ttft_s")
    e2e_s = targets.get("e2e_s")
    mean_itl_ms = targets.get("mean_itl_ms")

    if ttft_s is None and e2e_s is None and mean_itl_ms is None:
        return None

    base = float(ttft_s or 0.0)
    if mean_itl_ms is not None:
        return float(base + max(0, token_index - 1) * float(mean_itl_ms) / 1000.0)

    if e2e_s is not None and max_new_tokens > 1:
        return float(base + (float(e2e_s) - base) * (token_index - 1) / (max_new_tokens - 1))

    return float(base)


def build_timeline_rows(
    *,
    targets: dict[str, Any],
    real_arrivals_ms: list[list[float]],
    simulated_arrivals_ms: list[list[float]],
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    measured_max = max([len(x) for x in real_arrivals_ms + simulated_arrivals_ms] or [0])
    max_token = max(int(max_new_tokens), int(measured_max))
    rows: list[dict[str, Any]] = []
    for token_index in range(1, max_token + 1):
        target_s = target_time_s(targets, token_index, max_new_tokens)
        real_ms = mean_by_token(real_arrivals_ms, token_index)
        sim_ms = mean_by_token(simulated_arrivals_ms, token_index)
        rows.append(
            {
                "token_index": int(token_index),
                "target_s": target_s,
                "real_local_s": None if real_ms is None else float(real_ms / 1000.0),
                "simulated_network_s": None if sim_ms is None else float(sim_ms / 1000.0),
            }
        )
    return rows


def write_timeline_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=["token_index", "target_s", "real_local_s", "simulated_network_s"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "" if row.get(key) is None else row.get(key)
                    for key in writer.fieldnames
                }
            )


def plot_timeline(path: Path, rows: list[dict[str, Any]], targets: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(key: str) -> tuple[list[float], list[int]]:
        xs: list[float] = [0.0]
        ys: list[int] = [0]
        for row in rows:
            value = row.get(key)
            if value is None:
                continue
            xs.append(float(value))
            ys.append(int(row["token_index"]))
        return xs, ys

    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    for key, label in (
        ("target_s", "target"),
        ("real_local_s", "real local"),
        ("simulated_network_s", "simulated network"),
    ):
        xs, ys = series(key)
        if len(xs) > 1:
            ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.0, label=label)

    markers = (
        ("ttft_s", "TTFT target"),
        ("e2e_s", "E2E target"),
        ("atat_s", "ATAT target"),
    )
    for key, label in markers:
        if targets.get(key) is not None:
            ax.axvline(float(targets[key]), linestyle="--", linewidth=1.0, alpha=0.65, label=label)

    ax.set_xlabel("time (s)")
    ax.set_ylabel("generated tokens")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def target_checks(
    *,
    targets: dict[str, Any],
    simulated_arrivals_ms: list[list[float]],
    max_new_tokens: int,
) -> dict[str, Any]:
    first_s = [x / 1000.0 for x in first_values(simulated_arrivals_ms)]
    last_s = [x / 1000.0 for x in last_values(simulated_arrivals_ms)]
    itl_ms = inter_token_latencies_ms(simulated_arrivals_ms)
    atat_s_values = [x / 1000.0 for x in flatten(simulated_arrivals_ms)]

    observed = {
        "ttft_s": float(mean(first_s)),
        "e2e_s": float(mean(last_s)),
        "mean_itl_ms": float(mean(itl_ms)),
        "atat_s": float(mean(atat_s_values)),
    }
    checks: dict[str, Any] = {}
    for key in ("ttft_s", "e2e_s", "mean_itl_ms", "atat_s"):
        target = targets.get(key)
        checks[key] = {
            "target": None if target is None else float(target),
            "observed": float(observed[key]),
            "pass": None if target is None else bool(observed[key] <= float(target)),
        }

    conflict = False
    if (
        targets.get("ttft_s") is not None
        and targets.get("mean_itl_ms") is not None
        and targets.get("e2e_s") is not None
    ):
        implied_e2e = float(targets["ttft_s"]) + float(targets["mean_itl_ms"]) * max_new_tokens / 1000.0
        conflict = implied_e2e > float(targets["e2e_s"])
        if conflict:
            print(
                "[warn] target conflict: "
                f"ttft_s + mean_itl_ms * max_new_tokens = {implied_e2e:.3f}s "
                f"> e2e_s={float(targets['e2e_s']):.3f}s"
            )
    checks["target_conflict"] = bool(conflict)
    return checks


def summarize_arrivals(arrivals: list[list[float]]) -> dict[str, Any]:
    first_ms = first_values(arrivals)
    last_ms = last_values(arrivals)
    itl_ms = inter_token_latencies_ms(arrivals)
    all_ms = flatten(arrivals)
    return {
        "ttft_ms": stat_block(first_ms),
        "e2e_ms": stat_block(last_ms),
        "mean_itl_ms": stat_block(itl_ms),
        "atat_ms": {
            "count": int(len(all_ms)),
            "mean": float(mean(all_ms)),
            "max": float(max_or_zero(all_ms)),
        },
    }


def summarize_network_rounds(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    prefill = [x for x in rounds if x["phase"] == "prefill"]
    decode = [x for x in rounds if x["phase"] == "decode"]

    def block(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "rounds": int(len(items)),
            "wire_bytes": stat_block([float(x["wire_bytes"]) for x in items]),
            "delay_ms": stat_block([float(x["delay_ms"]) for x in items]),
            "uplink_ms": stat_block([float(x["uplink_ms"]) for x in items]),
            "rtt_ms": stat_block([float(x["rtt_ms"]) for x in items]),
            "packet_loss_events": int(sum(1 for x in items if x["packet_loss_event"])),
        }

    return {
        "prefill": block(prefill),
        "decode": block(decode),
        "total_rounds": int(len(rounds)),
        "total_wire_bytes": int(sum(int(x["wire_bytes"]) for x in rounds)),
        "packet_loss_events": int(sum(1 for x in rounds if x["packet_loss_event"])),
    }


def run_latency(
    *,
    args: argparse.Namespace,
    scenario: dict[str, Any],
    runtime: LocalSplitRuntime,
    tokenizer,
    codec,
    codec_extras: dict[str, Any],
    generation: dict[str, Any],
    selected_prompts: list[PromptSample],
    targets: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    sampling = sampling_from_generation(generation)
    eos_token_id, stop_token_ids = default_stop_ids(tokenizer, bool(generation["stop_at_eos"]))
    max_new_tokens = int(generation["max_new_tokens"])
    max_prompt_length = max_prompt_length_from_config(scenario, args)
    network_cfg = as_dict(scenario.get("network"))
    network_seed = int(network_cfg.get("seed", int(args.seed or 42)))
    rng = random.Random(network_seed)
    amp_enabled = runtime.device.type == "cuda" and runtime.dtype in (
        torch.float16,
        torch.bfloat16,
    )

    print(f"[info] scenario latency samples={len(selected_prompts)} max_new_tokens={max_new_tokens}")
    print(f"[info] max_prompt_length={max_prompt_length} network_seed={network_seed}")

    metrics = []
    real_arrivals: list[list[float]] = []
    simulated_arrivals: list[list[float]] = []
    network_rounds: list[dict[str, Any]] = []
    prompt_tokens_after_truncation: list[int] = []
    finish_reason_count: dict[str, int] = {}
    progress_every = max(1, int(args.progress_every))
    t0 = time.perf_counter()

    for i, sample in enumerate(selected_prompts, start=1):
        enc = tokenizer(
            sample.text,
            max_length=max_prompt_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(runtime.device)
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(runtime.device)
        prompt_tokens_after_truncation.append(int(input_ids.shape[1]))

        metric = run_one_sample(
            runtime=runtime,
            codec=codec,
            input_ids=input_ids,
            attention_mask=attention_mask,
            eos_token_id=eos_token_id,
            stop_token_ids=stop_token_ids,
            sampling=sampling,
            max_new_tokens=max_new_tokens,
            codec_extras=codec_extras,
            amp_enabled=amp_enabled,
        )
        metrics.append(metric)
        finish_reason_count[metric.finish_reason] = finish_reason_count.get(metric.finish_reason, 0) + 1

        timeline = simulate_network_timeline(
            metric=metric,
            network_cfg=network_cfg,
            rng=rng,
            downlink_response_bytes=max(0, int(args.downlink_response_bytes)),
        )
        real_arrivals.append(timeline.real_local_ms)
        simulated_arrivals.append(timeline.simulated_ms)
        network_rounds.extend(timeline.network_rounds)

        if i % progress_every == 0 or i == len(selected_prompts):
            sim_e2e = mean([row[-1] for row in simulated_arrivals if row])
            print(
                f"[progress] scenario latency {i}/{len(selected_prompts)} "
                f"sim_e2e_mean_ms={sim_e2e:.3f}"
            )

    elapsed_sec = time.perf_counter() - t0
    timeline_rows = build_timeline_rows(
        targets=targets,
        real_arrivals_ms=real_arrivals,
        simulated_arrivals_ms=simulated_arrivals,
        max_new_tokens=max_new_tokens,
    )
    csv_path = out_dir / "timeline.csv"
    png_path = out_dir / "timeline.png"
    write_timeline_csv(csv_path, timeline_rows)
    plot_timeline(png_path, timeline_rows, targets)

    real_total_ms = [float(x.total_ms) for x in metrics]
    decode_step_ms = [float(v) for x in metrics for v in x.decode_step_ms]
    generated_tokens = [int(x.generated_tokens) for x in metrics]
    prefill_wire = [
        int(r.wire_bytes)
        for x in metrics
        for r in x.codec_rounds
        if r.phase == "prefill"
    ]
    decode_wire = [
        int(r.wire_bytes)
        for x in metrics
        for r in x.codec_rounds
        if r.phase == "decode"
    ]

    return {
        "samples": int(len(selected_prompts)),
        "elapsed_sec": float(elapsed_sec),
        "max_prompt_length": int(max_prompt_length),
        "max_new_tokens": int(max_new_tokens),
        "generation": generation,
        "finish_reason_count": finish_reason_count,
        "prompt_tokens": {
            "before_truncation": stat_block([float(x.token_count) for x in selected_prompts]),
            "after_truncation": stat_block([float(x) for x in prompt_tokens_after_truncation]),
        },
        "generated_tokens": stat_block([float(x) for x in generated_tokens]),
        "real_local": {
            "runtime_total_ms": stat_block(real_total_ms),
            "token_arrivals": summarize_arrivals(real_arrivals),
            "decode_step_latency_ms": stat_block(decode_step_ms),
        },
        "simulated_network": {
            "token_arrivals": summarize_arrivals(simulated_arrivals),
            "network_rounds": summarize_network_rounds(network_rounds),
            "target_checks": target_checks(
                targets=targets,
                simulated_arrivals_ms=simulated_arrivals,
                max_new_tokens=max_new_tokens,
            ),
        },
        "codec_transfer_bytes": {
            "prefill": stat_block([float(x) for x in prefill_wire]),
            "decode": stat_block([float(x) for x in decode_wire]),
        },
        "timeline_csv": str(csv_path),
        "timeline_png": str(png_path),
    }


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def quality_cfg(scenario: dict[str, Any], metric: str) -> dict[str, Any]:
    return as_dict(as_dict(scenario.get("quality")).get(metric))


def run_ppl_quality(
    *,
    args: argparse.Namespace,
    scenario: dict[str, Any],
    codec_spec: str,
    codec_extras: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    from bench import ppl as ppl_bench

    cfg = quality_cfg(scenario, "ppl")
    dataset_cfg = as_dict(cfg.get("dataset"))
    ns = argparse.Namespace(
        front_dir=args.front_dir,
        back_dir=args.back_dir,
        revision=args.revision,
        tokenizer_id=args.tokenizer_id,
        trust_remote_code=bool(args.trust_remote_code),
        device=args.device,
        dtype=args.dtype,
        front_quant=args.front_quant,
        back_quant=args.back_quant,
        codec=codec_spec,
        codec_extras_json=json.dumps(codec_extras, ensure_ascii=False),
        dataset_name=str(dataset_cfg.get("name", "wikitext")),
        dataset_config=str(dataset_cfg.get("config", "wikitext-2-raw-v1")),
        split=str(dataset_cfg.get("split", "test")),
        text_column=str(dataset_cfg.get("text_column", cfg.get("text_column", "text"))),
        samples=int(args.ppl_samples or cfg.get("samples", 128)),
        max_length=int(cfg.get("max_length", 256)),
        batch_size=int(cfg.get("batch_size", 4)),
        progress_every=max(1, int(args.progress_every)),
        seed=int(args.seed or cfg.get("seed", 42)),
        out_json=str(out_dir / "ppl_result.json"),
    )
    return ppl_bench.run(ns)


def run_mmlu_quality(
    *,
    args: argparse.Namespace,
    scenario: dict[str, Any],
    codec_spec: str,
    codec_extras: dict[str, Any],
    generation: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    from bench import mmlu as mmlu_bench

    cfg = quality_cfg(scenario, "mmlu")
    dataset_cfg = as_dict(cfg.get("dataset"))
    decoding = str(cfg.get("decoding", generation.get("decoding", "greedy")))
    temperature = float(cfg.get("temperature", generation.get("temperature", 1.0)))
    if decoding == "greedy" and temperature <= 0:
        temperature = 1.0

    ns = argparse.Namespace(
        front_dir=args.front_dir,
        back_dir=args.back_dir,
        revision=args.revision,
        tokenizer_id=args.tokenizer_id,
        trust_remote_code=bool(args.trust_remote_code),
        device=args.device,
        dtype=args.dtype,
        front_quant=args.front_quant,
        back_quant=args.back_quant,
        codec=codec_spec,
        codec_extras_json=json.dumps(codec_extras, ensure_ascii=False),
        dataset_name=str(dataset_cfg.get("name", "cais/mmlu")),
        dataset_config=str(dataset_cfg.get("config", "all")),
        fewshot_split=str(cfg.get("fewshot_split", "dev")),
        eval_split=str(dataset_cfg.get("split", cfg.get("eval_split", "test"))),
        question_column=str(cfg.get("question_column", "question")),
        choices_column=str(cfg.get("choices_column", "choices")),
        answer_column=str(cfg.get("answer_column", "answer")),
        subject_column=str(cfg.get("subject_column", "subject")),
        subjects=cfg.get("subjects"),
        n_shot=int(cfg.get("n_shot", 5)),
        samples=int(args.mmlu_samples or cfg.get("samples", 32)),
        max_samples_per_subject=int(cfg.get("max_samples_per_subject", -1)),
        max_length=int(cfg.get("max_length", 2048)),
        decoding=decoding,
        max_new_tokens=int(cfg.get("max_new_tokens", 1)),
        temperature=temperature,
        top_k=int(cfg.get("top_k", generation.get("top_k", 50))),
        top_p=float(cfg.get("top_p", generation.get("top_p", 0.9))),
        min_new_tokens=int(cfg.get("min_new_tokens", 0)),
        no_repeat_ngram_size=int(cfg.get("no_repeat_ngram_size", 0)),
        repetition_penalty=float(cfg.get("repetition_penalty", 1.0)),
        jsonl_num_workers=int(cfg.get("jsonl_num_workers", 1)),
        jsonl_max_in_flight=int(cfg.get("jsonl_max_in_flight", 0)),
        prompt_jsonl=cfg.get("prompt_jsonl"),
        pred_jsonl=cfg.get("pred_jsonl"),
        progress_every=max(1, int(args.progress_every)),
        seed=int(args.seed or cfg.get("seed", 42)),
        out_json=str(out_dir / "mmlu_result.json"),
    )
    return mmlu_bench.run(ns)


def dry_run(scenario: dict[str, Any], args: argparse.Namespace) -> None:
    network_cfg = as_dict(scenario.get("network"))
    seed = int(network_cfg.get("seed", int(args.seed or 42)))
    rng = random.Random(seed)
    generation = workload_generation(scenario, args)
    sample = sample_network_round(
        network_cfg=network_cfg,
        rng=rng,
        phase="prefill",
        wire_bytes=1024 * 1024,
        downlink_response_bytes=max(0, int(args.downlink_response_bytes)),
    )
    scenario_cfg = as_dict(scenario.get("scenario"))
    print(
        "[ok] dry run parsed scenario="
        f"{scenario_cfg.get('name', '<unnamed>')} "
        f"codec={codec_spec_from_config(scenario, args.codec)} "
        f"max_new_tokens={generation['max_new_tokens']} "
        f"sample_prefill_delay_ms={sample['delay_ms']:.3f}"
    )


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    scenario_path = Path(args.scenario).expanduser()
    scenario = load_scenario(scenario_path)
    if args.dry_run:
        dry_run(scenario, args)
        return None

    seed = int(args.seed if args.seed is not None else as_dict(scenario.get("network")).get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario_cfg = as_dict(scenario.get("scenario"))
    targets = as_dict(scenario.get("targets"))
    codec_spec = codec_spec_from_config(scenario, args.codec)
    codec_extras = merge_codec_extras(scenario, args.codec_extras_json)
    generation = workload_generation(scenario, args)

    print(
        f"[info] benchmark=scenario name={scenario_cfg.get('name', '<unnamed>')} "
        f"codec_spec={codec_spec}"
    )
    print(f"[info] output_dir={out_dir}")

    codec = build_codec(codec_spec)
    runtime = LocalSplitRuntime(
        front_dir=args.front_dir,
        back_dir=args.back_dir,
        device=args.device,
        dtype=args.dtype,
        front_quant=args.front_quant,
        back_quant=args.back_quant,
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

    selected_prompts, dataset_info = select_prompt_samples(
        scenario=scenario,
        tokenizer=tokenizer,
        samples=max(1, int(args.samples)),
        scan_limit=int(args.dataset_scan_limit),
        seed=seed,
    )

    latency_result = run_latency(
        args=args,
        scenario=scenario,
        runtime=runtime,
        tokenizer=tokenizer,
        codec=codec,
        codec_extras=codec_extras,
        generation=generation,
        selected_prompts=selected_prompts,
        targets=targets,
        out_dir=out_dir,
    )

    runtime_info = {
        "device": str(runtime.device),
        "dtype": str(runtime.dtype),
        "seed": int(seed),
        "mode": "local_split",
        "simulation": "post_hoc_network",
    }
    del runtime
    cleanup_cuda()

    quality: dict[str, Any] = {"ppl": None, "mmlu": None}
    if not args.skip_ppl:
        print("[info] running PPL quality metric")
        quality["ppl"] = run_ppl_quality(
            args=args,
            scenario=scenario,
            codec_spec=codec_spec,
            codec_extras=codec_extras,
            out_dir=out_dir,
        )
        cleanup_cuda()
    else:
        print("[info] skipping PPL quality metric")

    if not args.skip_mmlu:
        print("[info] running MMLU quality metric")
        quality["mmlu"] = run_mmlu_quality(
            args=args,
            scenario=scenario,
            codec_spec=codec_spec,
            codec_extras=codec_extras,
            generation=generation,
            out_dir=out_dir,
        )
        cleanup_cuda()
    else:
        print("[info] skipping MMLU quality metric")

    result = {
        "benchmark": {
            "name": "scenario",
            "mode": "local_split",
            "simulation": "post_hoc_network",
        },
        "scenario": {
            "name": scenario_cfg.get("name"),
            "description": scenario_cfg.get("description"),
            "path": str(scenario_path),
        },
        "model": {
            "front_dir": str(Path(args.front_dir).expanduser()),
            "back_dir": str(Path(args.back_dir).expanduser()),
            "tokenizer_id": args.tokenizer_id,
            "revision": args.revision,
        },
        "codec": {
            "name": codec.name,
            "spec": codec_spec,
            "extras": codec_extras,
        },
        "dataset": {
            "workload": dataset_info,
        },
        "runtime": runtime_info,
        "network": {
            **as_dict(scenario.get("network")),
            "downlink_response_bytes": int(args.downlink_response_bytes),
        },
        "targets": targets,
        "eval": {
            "latency": latency_result,
            "quality": quality,
        },
    }

    out_json = out_dir / "scenario_result.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote {out_json.resolve()}")
    print(f"[ok] wrote {(out_dir / 'timeline.csv').resolve()}")
    print(f"[ok] wrote {(out_dir / 'timeline.png').resolve()}")

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
