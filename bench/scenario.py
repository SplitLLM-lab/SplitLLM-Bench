"""Scenario benchmark for split LLM edge-cloud inference.

This module intentionally keeps the scenario specification simple:

    task: input_tokens, output_tokens, action_tokens
    network: a preset name such as 5g_weak / 5g_mid / 6g_edge, or a small dict
    codec: raw / automix8 / automix10 / custom profile
    experience: ttft_s, interaction_s, complete_s

It uses SimPy as a discrete-event simulation engine.  The goal is not to model
full TCP, routing, or radio scheduling.  Instead, it models the specific queue
of events relevant to split inference:

    edge/front prefill -> activation upload -> cloud/back prefill
    -> repeated decode steps with optional split activation transfer

Outputs are designed for thesis figures:

    1. local compute-only arrival curve
    2. raw split over network arrival curve
    3. codec split over network arrival curve
    4. user-experience target curve

Run examples:

    python -m bench.scenario --scenario scenarios/ar_wearable_2k.yaml --samples 500
    python -m bench.scenario --scenario_dir scenarios --codec automix8 --network 5g_weak
    python -m bench.scenario --scenario scenarios/ar_wearable_2k.yaml \
        --codec_profile ./calib/automix_8x_rms.summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import simpy  # type: ignore
except Exception as exc:  # pragma: no cover - this branch is for runtime help.
    raise RuntimeError(
        "bench.scenario v3 requires SimPy. Install it with: pip install simpy"
    ) from exc

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import yaml


# -----------------------------------------------------------------------------
# Presets
# -----------------------------------------------------------------------------

NETWORK_PRESETS: Dict[str, Dict[str, float]] = {
    # Conservative, presentation-friendly presets rather than formal 3GPP claims.
    # All bandwidths are uplink Mbps because split activation upload is the bottleneck.
    "wifi": {"uplink_mbps": 80.0, "downlink_mbps": 150.0, "rtt_ms": 20.0, "jitter": 0.10},
    "4g": {"uplink_mbps": 10.0, "downlink_mbps": 30.0, "rtt_ms": 70.0, "jitter": 0.25},
    "5g_weak": {"uplink_mbps": 20.0, "downlink_mbps": 80.0, "rtt_ms": 45.0, "jitter": 0.22},
    "5g_mid": {"uplink_mbps": 50.0, "downlink_mbps": 150.0, "rtt_ms": 30.0, "jitter": 0.15},
    "5g_good": {"uplink_mbps": 120.0, "downlink_mbps": 300.0, "rtt_ms": 18.0, "jitter": 0.10},
    "6g_edge": {"uplink_mbps": 500.0, "downlink_mbps": 1000.0, "rtt_ms": 6.0, "jitter": 0.06},
    "ethernet": {"uplink_mbps": 1000.0, "downlink_mbps": 1000.0, "rtt_ms": 2.0, "jitter": 0.02},
}

CODEC_PRESETS: Dict[str, Dict[str, float]] = {
    "raw": {
        "compression_ratio": 1.0,
        "encode_ms_per_token": 0.0,
        "decode_ms_per_token": 0.0,
        "metadata_bytes_per_token": 0.0,
    },
    "4x": {
        "compression_ratio": 4.0,
        "encode_ms_per_token": 0.004,
        "decode_ms_per_token": 0.003,
        "metadata_bytes_per_token": 8.0,
    },
    "8x": {
        "compression_ratio": 8.0,
        "encode_ms_per_token": 0.006,
        "decode_ms_per_token": 0.004,
        "metadata_bytes_per_token": 8.0,
    },
    "10x": {
        "compression_ratio": 10.0,
        "encode_ms_per_token": 0.007,
        "decode_ms_per_token": 0.005,
        "metadata_bytes_per_token": 8.0,
    },
    "12x": {
        "compression_ratio": 12.0,
        "encode_ms_per_token": 0.008,
        "decode_ms_per_token": 0.006,
        "metadata_bytes_per_token": 8.0,
    },
    "16x": {
        "compression_ratio": 16.0,
        "encode_ms_per_token": 0.010,
        "decode_ms_per_token": 0.008,
        "metadata_bytes_per_token": 8.0,
    },
    "automix8": {
        "compression_ratio": 8.0,
        "encode_ms_per_token": 0.006,
        "decode_ms_per_token": 0.004,
        "metadata_bytes_per_token": 8.0,
    },
    "automix10": {
        "compression_ratio": 10.0,
        "encode_ms_per_token": 0.007,
        "decode_ms_per_token": 0.005,
        "metadata_bytes_per_token": 8.0,
    },
}

MODEL_PRESETS: Dict[str, Dict[str, float]] = {
    # These are intentionally simple knobs.  Replace with measured local latencies
    # when available.
    "llama3_8b_split": {
        "hidden_size": 4096,
        "activation_bytes_per_value": 2.0,
        "return_token_bytes": 8.0,
        "local_prefill_ms_per_token": 0.35,
        "local_decode_ms_per_token": 32.0,
        "edge_prefill_ms_per_token": 0.10,
        "cloud_prefill_ms_per_token": 0.24,
        "edge_decode_ms_per_token": 3.0,
        "cloud_decode_ms_per_token": 21.0,
        "decode_context_slowdown": 0.000018,
        "decode_position_slowdown": 0.0009,
        "compute_jitter": 0.03,
    },
    "fast_demo": {
        "hidden_size": 4096,
        "activation_bytes_per_value": 2.0,
        "return_token_bytes": 8.0,
        "local_prefill_ms_per_token": 0.15,
        "local_decode_ms_per_token": 16.0,
        "edge_prefill_ms_per_token": 0.05,
        "cloud_prefill_ms_per_token": 0.10,
        "edge_decode_ms_per_token": 1.5,
        "cloud_decode_ms_per_token": 12.0,
        "decode_context_slowdown": 0.000012,
        "decode_position_slowdown": 0.0007,
        "compute_jitter": 0.025,
    },
}


# -----------------------------------------------------------------------------
# Data objects
# -----------------------------------------------------------------------------

@dataclass
class NetworkProfile:
    name: str
    uplink_mbps: float
    downlink_mbps: float
    rtt_ms: float
    jitter: float = 0.0
    loss_prob: float = 0.0


@dataclass
class CodecProfile:
    name: str
    compression_ratio: float
    encode_ms_per_token: float = 0.0
    decode_ms_per_token: float = 0.0
    metadata_bytes_per_token: float = 0.0


@dataclass
class ModelProfile:
    name: str
    hidden_size: int
    activation_bytes_per_value: float
    return_token_bytes: float
    local_prefill_ms_per_token: float
    local_decode_ms_per_token: float
    edge_prefill_ms_per_token: float
    cloud_prefill_ms_per_token: float
    edge_decode_ms_per_token: float
    cloud_decode_ms_per_token: float
    decode_context_slowdown: float
    decode_position_slowdown: float
    compute_jitter: float


@dataclass
class TaskSpec:
    input_tokens: int
    output_tokens: int
    action_tokens: int


@dataclass
class ExperienceTarget:
    ttft_s: float
    interaction_s: float
    complete_s: float


@dataclass
class ScenarioSpec:
    name: str
    task: TaskSpec
    network: NetworkProfile
    codec: CodecProfile
    model: ModelProfile
    experience: ExperienceTarget
    dataset: str = "synthetic"
    description: str = ""


@dataclass
class SimulationTrace:
    scenario: str
    method: str
    seed: int
    token_times: List[float]
    ttft_s: float
    action_s: float
    e2e_s: float
    mean_itl_s: float
    bytes_uploaded: float
    bytes_downloaded: float
    prefill_upload_s: float
    decode_network_s: float
    compute_s: float
    codec_s: float


# -----------------------------------------------------------------------------
# Loading configuration
# -----------------------------------------------------------------------------

def _as_dict(x: Any) -> Dict[str, Any]:
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    return {"name": str(x)}


def load_network(value: Any, override: Optional[str] = None) -> NetworkProfile:
    if override:
        value = override
    d = _as_dict(value)
    name = d.get("name", value if isinstance(value, str) else "custom")
    if isinstance(name, str) and name in NETWORK_PRESETS:
        base = dict(NETWORK_PRESETS[name])
        base.update({k: v for k, v in d.items() if k != "name"})
        return NetworkProfile(name=name, **base)
    return NetworkProfile(
        name=str(name),
        uplink_mbps=float(d.get("uplink_mbps", 50.0)),
        downlink_mbps=float(d.get("downlink_mbps", d.get("uplink_mbps", 50.0))),
        rtt_ms=float(d.get("rtt_ms", 30.0)),
        jitter=float(d.get("jitter", 0.10)),
        loss_prob=float(d.get("loss_prob", 0.0)),
    )


def _read_codec_profile_json(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Accept AutoMix summary or generic profile.
    ratio = None
    for key in ("estimated_compression_ratio", "compression_ratio", "actual_compression_ratio"):
        if key in data:
            ratio = float(data[key])
            break
    if ratio is None and "actual_bits_per_value" in data:
        ratio = 16.0 / float(data["actual_bits_per_value"])
    if ratio is None:
        raise ValueError(f"Cannot infer compression ratio from codec profile: {path}")
    return {
        "compression_ratio": ratio,
        "encode_ms_per_token": float(data.get("encode_ms_per_token", data.get("encode_ms_per_row", 0.006))),
        "decode_ms_per_token": float(data.get("decode_ms_per_token", data.get("decode_ms_per_row", 0.004))),
        "metadata_bytes_per_token": float(data.get("metadata_bytes_per_token", 8.0)),
    }


def load_codec(value: Any, override: Optional[str] = None, profile_path: Optional[str] = None) -> CodecProfile:
    if profile_path:
        data = _read_codec_profile_json(profile_path)
        return CodecProfile(name=Path(profile_path).stem, **data)
    if override:
        value = override
    d = _as_dict(value)
    name = d.get("name", value if isinstance(value, str) else "automix8")
    if isinstance(name, str) and name in CODEC_PRESETS:
        base = dict(CODEC_PRESETS[name])
        base.update({k: v for k, v in d.items() if k != "name"})
        return CodecProfile(name=name, **base)
    return CodecProfile(
        name=str(name),
        compression_ratio=float(d.get("compression_ratio", 8.0)),
        encode_ms_per_token=float(d.get("encode_ms_per_token", 0.006)),
        decode_ms_per_token=float(d.get("decode_ms_per_token", 0.004)),
        metadata_bytes_per_token=float(d.get("metadata_bytes_per_token", 8.0)),
    )


def load_model(value: Any) -> ModelProfile:
    d = _as_dict(value)
    name = d.get("name", value if isinstance(value, str) else "llama3_8b_split")
    if isinstance(name, str) and name in MODEL_PRESETS:
        base = dict(MODEL_PRESETS[name])
        base.update({k: v for k, v in d.items() if k != "name"})
        return ModelProfile(name=name, **base)
    base = dict(MODEL_PRESETS["llama3_8b_split"])
    base.update({k: v for k, v in d.items() if k != "name"})
    return ModelProfile(name=str(name), **base)


def load_scenario(path: Path, codec_override: Optional[str] = None, network_override: Optional[str] = None,
                  codec_profile: Optional[str] = None) -> ScenarioSpec:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    task_d = cfg.get("task", {})
    out_tokens = int(task_d.get("output_tokens", task_d.get("max_new_tokens", 128)))
    task = TaskSpec(
        input_tokens=int(task_d.get("input_tokens", task_d.get("prompt_tokens", 2048))),
        output_tokens=out_tokens,
        action_tokens=int(task_d.get("action_tokens", max(1, min(out_tokens, round(out_tokens * 0.35))))),
    )
    exp_d = cfg.get("experience", {})
    experience = ExperienceTarget(
        ttft_s=float(exp_d.get("ttft_s", 1.5)),
        interaction_s=float(exp_d.get("interaction_s", exp_d.get("response_s", 6.0))),
        complete_s=float(exp_d.get("complete_s", 12.0)),
    )
    return ScenarioSpec(
        name=str(cfg.get("name", path.stem)),
        description=str(cfg.get("description", "")),
        dataset=str(cfg.get("dataset", "synthetic")),
        task=task,
        network=load_network(cfg.get("network", "5g_mid"), override=network_override),
        codec=load_codec(cfg.get("codec", "automix8"), override=codec_override, profile_path=codec_profile),
        model=load_model(cfg.get("model", "llama3_8b_split")),
        experience=experience,
    )


# -----------------------------------------------------------------------------
# SimPy network model
# -----------------------------------------------------------------------------

class Link:
    """A simple half-duplex-ish link model with independent uplink/downlink queues.

    This is not a full TCP model. It is a discrete-event abstraction for payload
    transfer with bandwidth, propagation delay, and jitter. It is enough for
    split-activation transfer because the key question is how many seconds are
    spent waiting for bytes to cross the boundary.
    """

    def __init__(self, env: "simpy.Environment", profile: NetworkProfile, rng: random.Random):
        self.env = env
        self.profile = profile
        self.rng = rng
        self.uplink = simpy.Resource(env, capacity=1)
        self.downlink = simpy.Resource(env, capacity=1)
        self.upload_time_acc = 0.0
        self.download_time_acc = 0.0

    def _sample_mbps(self, mean_mbps: float) -> float:
        jitter = max(0.0, self.profile.jitter)
        if jitter <= 1e-9:
            return max(1e-9, mean_mbps)
        # Lognormal keeps bandwidth positive and creates realistic occasional slowdowns.
        sigma = jitter
        mu = math.log(max(mean_mbps, 1e-9)) - 0.5 * sigma * sigma
        return max(0.05, self.rng.lognormvariate(mu, sigma))

    def _sample_one_way_delay(self) -> float:
        rtt_s = self.profile.rtt_ms / 1000.0
        jitter = max(0.0, self.profile.jitter)
        if jitter <= 1e-9:
            return rtt_s / 2.0
        factor = max(0.3, self.rng.gauss(1.0, jitter * 0.35))
        return max(0.0, rtt_s * factor / 2.0)

    def transmit(self, nbytes: float, direction: str = "uplink"):
        if nbytes <= 0:
            yield self.env.timeout(self._sample_one_way_delay())
            return
        resource = self.uplink if direction == "uplink" else self.downlink
        mean_bw = self.profile.uplink_mbps if direction == "uplink" else self.profile.downlink_mbps
        with resource.request() as req:
            yield req
            bw = self._sample_mbps(mean_bw)
            tx_s = (nbytes * 8.0) / (bw * 1e6)
            delay_s = self._sample_one_way_delay()
            total = tx_s + delay_s
            if direction == "uplink":
                self.upload_time_acc += total
            else:
                self.download_time_acc += total
            yield self.env.timeout(total)


def activation_bytes(tokens: int, model: ModelProfile, codec: CodecProfile) -> float:
    raw = tokens * model.hidden_size * model.activation_bytes_per_value
    return raw / max(codec.compression_ratio, 1e-9) + tokens * codec.metadata_bytes_per_token


def compute_ms(ms: float, rng: random.Random, jitter: float) -> float:
    if jitter <= 1e-9:
        return max(0.0, ms)
    return max(0.0, ms * max(0.5, rng.gauss(1.0, jitter)))


def decode_ms_for_token(base_ms: float, prompt_tokens: int, idx: int, model: ModelProfile) -> float:
    # Compute gets gradually slower with longer KV length and generated position.
    # This is intentionally simple and monotone.
    ctx_factor = 1.0 + model.decode_context_slowdown * max(0, prompt_tokens + idx)
    pos_factor = 1.0 + model.decode_position_slowdown * max(0, idx)
    return base_ms * ctx_factor * pos_factor


def run_local(spec: ScenarioSpec, seed: int) -> SimulationTrace:
    rng = random.Random(seed)
    env = simpy.Environment()
    token_times: List[float] = []
    compute_s = 0.0

    def proc():
        nonlocal compute_s
        prefill_s = compute_ms(spec.model.local_prefill_ms_per_token * spec.task.input_tokens, rng, spec.model.compute_jitter) / 1000.0
        compute_s += prefill_s
        yield env.timeout(prefill_s)
        for i in range(spec.task.output_tokens):
            ms = decode_ms_for_token(spec.model.local_decode_ms_per_token, spec.task.input_tokens, i, spec.model)
            step_s = compute_ms(ms, rng, spec.model.compute_jitter) / 1000.0
            compute_s += step_s
            yield env.timeout(step_s)
            token_times.append(env.now)

    env.process(proc())
    env.run()
    return summarize_trace(spec, "local", seed, token_times, 0.0, 0.0, 0.0, 0.0, compute_s, 0.0)


def run_split(spec: ScenarioSpec, method: str, codec: CodecProfile, seed: int) -> SimulationTrace:
    rng = random.Random(seed)
    env = simpy.Environment()
    link = Link(env, spec.network, rng)
    token_times: List[float] = []
    bytes_up = 0.0
    bytes_down = 0.0
    compute_s = 0.0
    codec_s = 0.0
    prefill_upload_s = 0.0
    decode_network_s_before = 0.0

    def proc():
        nonlocal bytes_up, bytes_down, compute_s, codec_s, prefill_upload_s, decode_network_s_before

        # Edge/front prefill.
        edge_prefill_s = compute_ms(spec.model.edge_prefill_ms_per_token * spec.task.input_tokens, rng, spec.model.compute_jitter) / 1000.0
        compute_s += edge_prefill_s
        yield env.timeout(edge_prefill_s)

        # Codec encode and upload split activation.
        enc_s = codec.encode_ms_per_token * spec.task.input_tokens / 1000.0
        codec_s += enc_s
        yield env.timeout(enc_s)
        pre_upload_start = env.now
        b = activation_bytes(spec.task.input_tokens, spec.model, codec)
        bytes_up += b
        yield env.process(link.transmit(b, "uplink"))
        prefill_upload_s += env.now - pre_upload_start

        # Cloud/back prefill.
        cloud_prefill_s = compute_ms(spec.model.cloud_prefill_ms_per_token * spec.task.input_tokens, rng, spec.model.compute_jitter) / 1000.0
        compute_s += cloud_prefill_s
        yield env.timeout(cloud_prefill_s)

        dec_s = codec.decode_ms_per_token * spec.task.input_tokens / 1000.0
        codec_s += dec_s
        yield env.timeout(dec_s)

        # Decode loop.
        for i in range(spec.task.output_tokens):
            edge_decode_s = compute_ms(spec.model.edge_decode_ms_per_token, rng, spec.model.compute_jitter) / 1000.0
            compute_s += edge_decode_s
            yield env.timeout(edge_decode_s)

            enc_step_s = codec.encode_ms_per_token / 1000.0
            codec_s += enc_step_s
            yield env.timeout(enc_step_s)

            net_start = env.now
            step_bytes = activation_bytes(1, spec.model, codec)
            bytes_up += step_bytes
            yield env.process(link.transmit(step_bytes, "uplink"))

            cloud_ms = decode_ms_for_token(spec.model.cloud_decode_ms_per_token, spec.task.input_tokens, i, spec.model)
            cloud_decode_s = compute_ms(cloud_ms, rng, spec.model.compute_jitter) / 1000.0
            compute_s += cloud_decode_s
            yield env.timeout(cloud_decode_s)

            dec_step_s = codec.decode_ms_per_token / 1000.0
            codec_s += dec_step_s
            yield env.timeout(dec_step_s)

            # Return the token id / text fragment. This is tiny but pays one-way delay.
            token_bytes = spec.model.return_token_bytes
            bytes_down += token_bytes
            yield env.process(link.transmit(token_bytes, "downlink"))
            decode_network_s_before += env.now - net_start - cloud_decode_s - dec_step_s
            token_times.append(env.now)

    env.process(proc())
    env.run()
    return summarize_trace(
        spec,
        method,
        seed,
        token_times,
        bytes_up,
        bytes_down,
        prefill_upload_s,
        max(0.0, decode_network_s_before),
        compute_s,
        codec_s,
    )


def summarize_trace(spec: ScenarioSpec, method: str, seed: int, token_times: List[float], bytes_up: float, bytes_down: float,
                    prefill_upload_s: float, decode_network_s: float, compute_s: float, codec_s: float) -> SimulationTrace:
    if not token_times:
        ttft = action = e2e = float("nan")
        mean_itl = float("nan")
    else:
        ttft = token_times[0]
        action_idx = max(1, min(spec.task.action_tokens, len(token_times))) - 1
        action = token_times[action_idx]
        e2e = token_times[-1]
        if len(token_times) > 1:
            mean_itl = float(np.mean(np.diff(token_times)))
        else:
            mean_itl = 0.0
    return SimulationTrace(
        scenario=spec.name,
        method=method,
        seed=seed,
        token_times=[float(x) for x in token_times],
        ttft_s=float(ttft),
        action_s=float(action),
        e2e_s=float(e2e),
        mean_itl_s=float(mean_itl),
        bytes_uploaded=float(bytes_up),
        bytes_downloaded=float(bytes_down),
        prefill_upload_s=float(prefill_upload_s),
        decode_network_s=float(decode_network_s),
        compute_s=float(compute_s),
        codec_s=float(codec_s),
    )


# -----------------------------------------------------------------------------
# Aggregation and plotting
# -----------------------------------------------------------------------------

def target_curve(spec: ScenarioSpec) -> Tuple[np.ndarray, np.ndarray]:
    # Piecewise user-experience target curve:
    # first visible token by TTFT, actionable content by interaction_s, full answer by complete_s.
    xs = np.array([0.0, spec.experience.ttft_s, spec.experience.interaction_s, spec.experience.complete_s])
    ys = np.array([0.0, 1.0, float(spec.task.action_tokens), float(spec.task.output_tokens)])
    # Remove non-increasing x points if user sets aggressive targets.
    keep_x = [xs[0]]
    keep_y = [ys[0]]
    for x, y in zip(xs[1:], ys[1:]):
        if x > keep_x[-1]:
            keep_x.append(x)
            keep_y.append(y)
    return np.array(keep_x), np.array(keep_y)


def arrival_quantiles(traces: Sequence[SimulationTrace], output_tokens: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = np.full((len(traces), output_tokens), np.nan)
    for i, tr in enumerate(traces):
        n = min(output_tokens, len(tr.token_times))
        if n:
            arr[i, :n] = tr.token_times[:n]
    q10 = np.nanquantile(arr, 0.10, axis=0)
    q50 = np.nanquantile(arr, 0.50, axis=0)
    q90 = np.nanquantile(arr, 0.90, axis=0)
    tokens = np.arange(1, output_tokens + 1)
    return tokens, q10, q50, q90


def grouped(traces: Sequence[SimulationTrace]) -> Dict[str, List[SimulationTrace]]:
    out: Dict[str, List[SimulationTrace]] = {}
    for tr in traces:
        out.setdefault(tr.method, []).append(tr)
    return out


def plot_arrival(spec: ScenarioSpec, traces: Sequence[SimulationTrace], out_path: Path) -> None:
    by = grouped(traces)
    colors = {
        "local": "#6e6e6e",
        "raw": "#e45745",
        "codec": "#276bd6",
    }
    labels = {
        "local": "Local compute only",
        "raw": "Raw split over network",
        "codec": f"Network + codec ({spec.codec.name}, {spec.codec.compression_ratio:.1f}x)",
    }
    band_alpha = {"raw": 0.11, "codec": 0.14, "local": 0.06}
    plt.figure(figsize=(9.4, 5.6))
    for method in ["raw", "codec", "local"]:
        if method not in by:
            continue
        tokens, q10, q50, q90 = arrival_quantiles(by[method], spec.task.output_tokens)
        color = colors[method]
        plt.fill_betweenx(tokens, q10, q90, color=color, alpha=band_alpha.get(method, 0.12), linewidth=0)
        plt.plot(q50, tokens, color=color, linewidth=2.4, label=labels[method])

    tx, ty = target_curve(spec)
    plt.plot(tx, ty, "--", color="#222222", linewidth=1.8, alpha=0.48, label="User-experience target")

    deadline_specs = [
        (spec.experience.ttft_s, "TTFT deadline", "#9b59b6"),
        (spec.experience.interaction_s, "Interaction deadline", "#f39c12"),
        (spec.experience.complete_s, "Complete deadline", "#2e7d32"),
    ]
    for x, _, color in deadline_specs:
        plt.axvline(x, color=color, linestyle="--", linewidth=1.4, alpha=0.55)

    plt.title("Token arrival curve")
    plt.xlabel("time (s)")
    plt.ylabel("arrived/generated tokens")
    plt.grid(True, alpha=0.25)
    plt.xlim(left=0)
    plt.ylim(bottom=0, top=spec.task.output_tokens * 1.03)

    handles, labels_in_plot = plt.gca().get_legend_handles_labels()
    handles.extend([
        Line2D([0], [0], color=color, linestyle="--", linewidth=1.4, alpha=0.55, label=label)
        for _, label, color in deadline_specs
    ])
    labels_in_plot.extend([label for _, label, _ in deadline_specs])
    plt.legend(handles, labels_in_plot, loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def quantile_summary(traces: Sequence[SimulationTrace]) -> pd.DataFrame:
    rows = []
    for (scenario, method), group_df in pd.DataFrame([asdict(t) for t in traces]).groupby(["scenario", "method"]):
        row: Dict[str, Any] = {"scenario": scenario, "method": method, "trials": len(group_df)}
        for metric in ["ttft_s", "action_s", "e2e_s", "mean_itl_s", "bytes_uploaded", "bytes_downloaded", "prefill_upload_s", "decode_network_s", "compute_s", "codec_s"]:
            vals = group_df[metric].astype(float).to_numpy()
            row[f"{metric}_p10"] = float(np.nanquantile(vals, 0.10))
            row[f"{metric}_median"] = float(np.nanquantile(vals, 0.50))
            row[f"{metric}_p90"] = float(np.nanquantile(vals, 0.90))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_latency_bars(specs: Dict[str, ScenarioSpec], summary: pd.DataFrame, out_path: Path) -> None:
    # One compact grouped bar plot for median TTFT/action/E2E.
    rows = []
    for _, row in summary.iterrows():
        spec = specs[row["scenario"]]
        rows.append({"scenario": row["scenario"], "method": row["method"], "metric": "TTFT", "value": row["ttft_s_median"], "target": spec.experience.ttft_s})
        rows.append({"scenario": row["scenario"], "method": row["method"], "metric": "interaction", "value": row["action_s_median"], "target": spec.experience.interaction_s})
        rows.append({"scenario": row["scenario"], "method": row["method"], "metric": "complete", "value": row["e2e_s_median"], "target": spec.experience.complete_s})
    df = pd.DataFrame(rows)
    # Normalize by target so scenarios are comparable.
    df["normalized"] = df["value"] / df["target"]
    pivot = df.pivot_table(index=["scenario", "metric"], columns="method", values="normalized")
    pivot = pivot[[c for c in ["local", "raw", "codec"] if c in pivot.columns]]
    plt.figure(figsize=(11, max(4.8, 0.45 * len(pivot))))
    y = np.arange(len(pivot))
    h = 0.24
    colors = {"local": "#6e6e6e", "raw": "#e45745", "codec": "#276bd6"}
    for k, method in enumerate(pivot.columns):
        plt.barh(y + (k - (len(pivot.columns)-1)/2) * h, pivot[method], height=h, color=colors.get(method, None), label=method)
    plt.axvline(1.0, color="#222222", linestyle="--", linewidth=1.2, label="target")
    plt.yticks(y, [f"{idx[0]} / {idx[1]}" for idx in pivot.index], fontsize=8)
    plt.xlabel("median latency / user-experience target  (≤1 passes)")
    plt.title("Latency target attainment")
    plt.grid(True, axis="x", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_breakdown(summary: pd.DataFrame, out_path: Path) -> None:
    df = summary.copy()
    df = df[df["method"].isin(["raw", "codec"])]
    if df.empty:
        return
    method_order = {"codec": 0, "raw": 1}
    df["_method_order"] = df["method"].map(method_order).fillna(99)
    df = df.sort_values(["scenario", "_method_order"]).reset_index(drop=True)
    if df["scenario"].nunique() == 1:
        labels = list(df["method"])
        xtick_rotation = 0
        xtick_ha = "center"
        fig_width = 8.8
        fontsize = 12
    else:
        labels = [f"{row.scenario}\n{row.method}" for row in df.itertuples(index=False)]
        xtick_rotation = 0
        xtick_ha = "center"
        fig_width = max(9.5, 0.65 * len(df))
        fontsize = 8
    components = [
        ("compute_s_median", "compute"),
        ("prefill_upload_s_median", "prefill upload"),
        ("decode_network_s_median", "decode network"),
        ("codec_s_median", "codec"),
    ]
    x = np.arange(len(df))
    bottom = np.zeros(len(df))
    plt.figure(figsize=(fig_width, 5.6))
    for col, label in components:
        vals = df[col].astype(float).to_numpy()
        plt.bar(x, vals, bottom=bottom, label=label)
        bottom += vals
    plt.xticks(x, labels, rotation=xtick_rotation, ha=xtick_ha, fontsize=fontsize, fontstyle="normal")
    plt.ylabel("seconds")
    plt.title("Median latency decomposition")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_bytes(summary: pd.DataFrame, out_path: Path) -> None:
    df = summary[summary["method"].isin(["raw", "codec"])].copy()
    if df.empty:
        return
    df["uploaded_MB"] = df["bytes_uploaded_median"] / (1024 ** 2)
    pivot = df.pivot_table(index="scenario", columns="method", values="uploaded_MB")
    pivot = pivot[[c for c in ["raw", "codec"] if c in pivot.columns]]
    plt.figure(figsize=(10, max(4.5, 0.36 * len(pivot))))
    y = np.arange(len(pivot))
    h = 0.32
    colors = {"raw": "#e45745", "codec": "#276bd6"}
    for k, method in enumerate(pivot.columns):
        plt.barh(y + (k - (len(pivot.columns)-1)/2) * h, pivot[method], height=h, label=method, color=colors.get(method, None))
    plt.yticks(y, pivot.index, fontsize=8)
    plt.xlabel("median uploaded activation volume (MiB)")
    plt.title("Transmission volume")
    plt.grid(True, axis="x", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


# -----------------------------------------------------------------------------
# Running
# -----------------------------------------------------------------------------

def scenario_paths(args: argparse.Namespace) -> List[Path]:
    paths: List[Path] = []
    if args.scenario:
        paths.extend(Path(p) for p in args.scenario)
    if args.scenario_dir:
        for p in sorted(Path(args.scenario_dir).glob("*.yaml")):
            paths.append(p)
    if not paths:
        raise ValueError("Provide --scenario or --scenario_dir")
    return paths


def run_spec(spec: ScenarioSpec, samples: int, seed: int) -> List[SimulationTrace]:
    traces: List[SimulationTrace] = []
    for i in range(samples):
        s = seed + i * 1009
        traces.append(run_local(spec, s))
        traces.append(run_split(spec, "raw", load_codec("raw"), s + 17))
        traces.append(run_split(spec, "codec", spec.codec, s + 29))
    return traces


def write_trace_csv(traces: Sequence[SimulationTrace], path: Path) -> None:
    rows = []
    for tr in traces:
        for idx, t in enumerate(tr.token_times, start=1):
            rows.append({"scenario": tr.scenario, "method": tr.method, "seed": tr.seed, "token": idx, "time_s": t})
    pd.DataFrame(rows).to_csv(path, index=False)


# -----------------------------------------------------------------------------
# Live generation mode
# -----------------------------------------------------------------------------

@dataclass
class LiveRunResult:
    method: str
    token_times_compute_s: List[float]
    prefill_wire_bytes: int
    decode_wire_bytes: List[int]
    prefill_encode_ms: float
    prefill_decode_ms: float
    decode_encode_ms: List[float]
    decode_decode_ms: List[float]
    generated_tokens: int
    finish_reason: str
    prompt_tokens: int


def _network_transfer_sample_s(nbytes: float, profile: NetworkProfile, rng: random.Random, direction: str = "uplink") -> float:
    """Sample one payload transfer time for the live post-hoc network model."""
    mean_mbps = profile.uplink_mbps if direction == "uplink" else profile.downlink_mbps
    jitter = max(0.0, profile.jitter)
    if jitter <= 1e-9:
        bw = max(1e-9, mean_mbps)
    else:
        sigma = jitter
        mu = math.log(max(mean_mbps, 1e-9)) - 0.5 * sigma * sigma
        bw = max(0.05, rng.lognormvariate(mu, sigma))
    tx_s = (float(nbytes) * 8.0) / (bw * 1e6) if nbytes > 0 else 0.0
    rtt_s = profile.rtt_ms / 1000.0
    if jitter <= 1e-9:
        delay_s = rtt_s / 2.0
    else:
        delay_s = max(0.0, rtt_s * max(0.3, rng.gauss(1.0, jitter * 0.35)) / 2.0)
    return tx_s + delay_s


def _metric_to_live_result(metric: Any, method: str) -> LiveRunResult:
    # Actual compute-only token arrival times reconstructed from SplitLLM latency metric.
    times: List[float] = []
    if int(metric.generated_tokens) > 0:
        cur = float(metric.ttft_ms) / 1000.0
        times.append(cur)
        for step_ms in list(metric.decode_step_ms)[: max(0, int(metric.generated_tokens) - 1)]:
            cur += float(step_ms) / 1000.0
            times.append(cur)
    prefill_wire = 0
    prefill_enc = 0.0
    prefill_dec = 0.0
    dec_wire: List[int] = []
    dec_enc: List[float] = []
    dec_dec: List[float] = []
    for r in list(metric.codec_rounds):
        phase = str(getattr(r, "phase", ""))
        if phase == "prefill":
            prefill_wire = int(getattr(r, "wire_bytes", 0))
            prefill_enc += float(getattr(r, "encode_ms", 0.0))
            prefill_dec += float(getattr(r, "decode_ms", 0.0))
        elif phase == "decode":
            dec_wire.append(int(getattr(r, "wire_bytes", 0)))
            dec_enc.append(float(getattr(r, "encode_ms", 0.0)))
            dec_dec.append(float(getattr(r, "decode_ms", 0.0)))
    return LiveRunResult(
        method=method,
        token_times_compute_s=times,
        prefill_wire_bytes=prefill_wire,
        decode_wire_bytes=dec_wire,
        prefill_encode_ms=prefill_enc,
        prefill_decode_ms=prefill_dec,
        decode_encode_ms=dec_enc,
        decode_decode_ms=dec_dec,
        generated_tokens=int(metric.generated_tokens),
        finish_reason=str(metric.finish_reason),
        prompt_tokens=int(metric.prompt_tokens),
    )


def _live_result_to_local_trace(spec: ScenarioSpec, res: LiveRunResult, seed: int) -> SimulationTrace:
    times = list(res.token_times_compute_s)
    return summarize_trace(
        spec,
        res.method,
        seed,
        times,
        0.0,
        0.0,
        0.0,
        0.0,
        times[-1] if times else 0.0,
        (res.prefill_encode_ms + res.prefill_decode_ms + sum(res.decode_encode_ms) + sum(res.decode_decode_ms)) / 1000.0,
    )


def _live_result_to_network_trace(spec: ScenarioSpec, res: LiveRunResult, seed: int, rng: random.Random) -> SimulationTrace:
    """Add simulated network transfer delays to a real compute trace."""
    if not res.token_times_compute_s:
        return summarize_trace(spec, res.method, seed, [], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    output_n = len(res.token_times_compute_s)
    down_bytes = spec.model.return_token_bytes
    pre_net = _network_transfer_sample_s(res.prefill_wire_bytes, spec.network, rng, "uplink")
    first_down = _network_transfer_sample_s(down_bytes, spec.network, rng, "downlink")
    token_times: List[float] = [res.token_times_compute_s[0] + pre_net + first_down]
    decode_net_acc = first_down
    bytes_up = float(res.prefill_wire_bytes)
    bytes_down = float(down_bytes)
    # For tokens 2..N, add one activation upload and one tiny token return per step.
    for i in range(1, output_n):
        compute_delta = max(0.0, res.token_times_compute_s[i] - res.token_times_compute_s[i - 1])
        wire = int(res.decode_wire_bytes[i - 1]) if i - 1 < len(res.decode_wire_bytes) else 0
        up = _network_transfer_sample_s(wire, spec.network, rng, "uplink")
        down = _network_transfer_sample_s(down_bytes, spec.network, rng, "downlink")
        token_times.append(token_times[-1] + compute_delta + up + down)
        decode_net_acc += up + down
        bytes_up += float(wire)
        bytes_down += float(down_bytes)
    codec_s = (res.prefill_encode_ms + res.prefill_decode_ms + sum(res.decode_encode_ms) + sum(res.decode_decode_ms)) / 1000.0
    # compute_s is approximate here: the measured compute-only timeline already includes codec overhead.
    compute_s = max(0.0, res.token_times_compute_s[-1] - codec_s)
    return summarize_trace(
        spec,
        res.method,
        seed,
        token_times,
        bytes_up,
        bytes_down,
        pre_net,
        max(0.0, decode_net_acc),
        compute_s,
        codec_s,
    )


def _make_synthetic_input_ids(tokenizer: Any, target_tokens: int, device: Any) -> Tuple[Any, Any, str]:
    base = (
        "You are evaluating a split edge-cloud language model. "
        "Answer concisely and continue the response naturally. "
    )
    text = base
    # Grow until tokenized length is enough; keep this deterministic.
    ids = tokenizer(text, add_special_tokens=True, return_tensors="pt")["input_ids"][0].tolist()
    while len(ids) < target_tokens:
        text += base
        ids = tokenizer(text, add_special_tokens=True, return_tensors="pt")["input_ids"][0].tolist()
    ids = ids[:target_tokens]
    import torch  # local import so simulation-only mode does not require torch at import time
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    return input_ids, attention_mask, tokenizer.decode(ids, skip_special_tokens=False)


def _load_live_prompt(args: argparse.Namespace, spec: ScenarioSpec, tokenizer: Any, device: Any) -> Tuple[Any, Any, str]:
    import torch
    text: Optional[str] = None
    if getattr(args, "prompt", None):
        text = str(args.prompt)
    elif getattr(args, "prompt_file", None):
        text = Path(args.prompt_file).read_text(encoding="utf-8")
    elif getattr(args, "dataset_name", None):
        try:
            from bench.utils import load_texts  # type: ignore
            texts = load_texts(
                dataset_name=args.dataset_name,
                dataset_config=args.dataset_config,
                split=args.split,
                text_column=args.text_column,
                samples=1,
            )
            if texts:
                text = str(texts[0])
        except Exception as exc:
            print(f"[warn] failed to load dataset prompt; falling back to synthetic prompt: {exc}")
            text = None
    if text is None:
        return _make_synthetic_input_ids(tokenizer, spec.task.input_tokens, device)
    max_len = int(args.max_prompt_length or spec.task.input_tokens)
    enc = tokenizer(text, max_length=max_len, truncation=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
    return input_ids, attention_mask, text


def _run_one_actual_generation(
    *,
    args: argparse.Namespace,
    codec_name: str,
    method: str,
    input_ids: Any,
    attention_mask: Any,
    tokenizer: Any,
    max_new_tokens: int,
) -> LiveRunResult:
    import torch
    from bench.latency import build_runtime, run_one_sample, run_one_sample_runtime  # type: ignore
    from bench.utils import build_codec, parse_codec_extras  # type: ignore
    from model import SamplingConfig  # type: ignore

    codec = build_codec(codec_name)
    rt_args = SimpleNamespace(
        runtime_mode="local_split",
        front_dir=args.front_dir,
        back_dir=args.back_dir,
        server_url=None,
        timeout_sec=float(args.timeout_sec),
        revision=args.revision,
        tokenizer_id=args.tokenizer_id,
        trust_remote_code=bool(args.trust_remote_code),
        device=args.device,
        dtype=args.dtype,
        front_quant=args.front_quant,
        back_quant=args.back_quant,
        generation_mode=str(args.generation_mode),
    )
    runtime = build_runtime(rt_args, codec)
    codec = runtime.codec
    eos_token_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None
    stop_token_ids = set()
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
        self_speculative=(str(args.generation_mode) == "self_speculative"),
        assistant_early_exit=(int(args.assistant_early_exit) if str(args.generation_mode) == "self_speculative" and args.assistant_early_exit is not None else None),
        num_speculations=int(args.num_speculations) if str(args.generation_mode) == "self_speculative" else 0,
    )
    sampling.validate()
    codec_extras = parse_codec_extras(args.codec_extras_json)
    amp_enabled = runtime.device.type == "cuda" and runtime.dtype in (torch.float16, torch.bfloat16)
    print(
        f"[live] loading/running method={method} codec={codec_name} "
        f"generation_mode={args.generation_mode} device={runtime.device} dtype={runtime.dtype}"
    )
    if sampling.self_speculative or sampling.assistant_early_exit is not None:
        # LayerSkip self-speculative decoding is implemented in the runtime path.
        # This mirrors bench.latency: use runtime.generate_from_ids() so draft/verify
        # metrics and codec wire bytes are recorded correctly.
        metric = run_one_sample_runtime(
            runtime=runtime,
            input_ids=input_ids.to(runtime.device),
            attention_mask=attention_mask.to(runtime.device),
            eos_token_id=eos_token_id,
            stop_token_ids=stop_token_ids,
            sampling=sampling,
            max_new_tokens=max_new_tokens,
            codec_extras=codec_extras,
        )
    else:
        metric = run_one_sample(
            runtime=runtime,
            codec=codec,
            input_ids=input_ids.to(runtime.device),
            attention_mask=attention_mask.to(runtime.device),
            eos_token_id=eos_token_id,
            stop_token_ids=stop_token_ids,
            sampling=sampling,
            max_new_tokens=max_new_tokens,
            codec_extras=codec_extras,
            amp_enabled=amp_enabled,
        )
    # Free as much as possible before next method.
    try:
        del runtime
        del codec
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return _metric_to_live_result(metric, method)


def run_live_spec(spec: ScenarioSpec, args: argparse.Namespace) -> Tuple[List[SimulationTrace], Dict[str, Any]]:
    """Run the model once for local/raw/codec, then Monte-Carlo sample network delays."""
    import torch
    from transformers import AutoTokenizer

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_id,
        use_fast=True,
        trust_remote_code=bool(args.trust_remote_code),
        revision=args.revision,
        local_files_only=bool(getattr(args, "local_files_only", False)),
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Need a provisional device for prompt tensor. build_runtime will move it if needed.
    provisional_device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    input_ids, attention_mask, prompt_text = _load_live_prompt(args, spec, tokenizer, provisional_device)
    max_new_tokens = int(args.max_new_tokens or spec.task.output_tokens)
    print(f"[live] prompt_tokens={int(input_ids.shape[1])}, max_new_tokens={max_new_tokens}, network_trials={args.samples}")

    local_codec = str(args.local_codec)
    raw_codec = str(args.raw_codec)
    method_codec = str(args.codec)
    local_res = _run_one_actual_generation(
        args=args, codec_name=local_codec, method="local", input_ids=input_ids, attention_mask=attention_mask,
        tokenizer=tokenizer, max_new_tokens=max_new_tokens,
    )
    raw_res = _run_one_actual_generation(
        args=args, codec_name=raw_codec, method="raw", input_ids=input_ids, attention_mask=attention_mask,
        tokenizer=tokenizer, max_new_tokens=max_new_tokens,
    )
    codec_res = _run_one_actual_generation(
        args=args, codec_name=method_codec, method="codec", input_ids=input_ids, attention_mask=attention_mask,
        tokenizer=tokenizer, max_new_tokens=max_new_tokens,
    )

    traces: List[SimulationTrace] = []
    # Local has no network; repeat it so quantile code can reuse the same path.
    for i in range(int(args.samples)):
        traces.append(_live_result_to_local_trace(spec, local_res, int(args.seed) + i))
        rng_raw = random.Random(int(args.seed) + i * 1009 + 17)
        rng_codec = random.Random(int(args.seed) + i * 1009 + 29)
        traces.append(_live_result_to_network_trace(spec, raw_res, int(args.seed) + i, rng_raw))
        traces.append(_live_result_to_network_trace(spec, codec_res, int(args.seed) + i, rng_codec))
    details = {
        "prompt_preview": prompt_text[:1000],
        "actual_runs": {
            "local": asdict(local_res),
            "raw": asdict(raw_res),
            "codec": asdict(codec_res),
        },
        "note": "Model generation is executed once per method. Network delay is Monte-Carlo simulated from measured codec wire bytes.",
    }
    return traces, details


def run_live_main(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    specs: Dict[str, ScenarioSpec] = {}
    all_traces: List[SimulationTrace] = []
    live_details: Dict[str, Any] = {}
    for path in scenario_paths(args):
        # In live mode --codec is a real codec module.  Do not use it to infer compression profile;
        # use YAML/profile only for plot label unless a --codec_profile is passed.
        spec = load_scenario(path, codec_override=None, network_override=args.network, codec_profile=args.codec_profile)
        if args.codec_label:
            spec.codec.name = str(args.codec_label)
        else:
            spec.codec.name = str(args.codec).split(":")[0].split(".")[-1]
        specs[spec.name] = spec
        traces, details = run_live_spec(spec, args)
        all_traces.extend(traces)
        live_details[spec.name] = details
        plot_arrival(spec, traces, plot_dir / f"arrival_{spec.name}.png")

    with open(out_dir / "scenario_config_resolved.json", "w", encoding="utf-8") as f:
        json.dump({k: asdict(v) for k, v in specs.items()}, f, indent=2)
    with open(out_dir / "live_generation_details.json", "w", encoding="utf-8") as f:
        json.dump(live_details, f, indent=2)
    with open(out_dir / "traces.json", "w", encoding="utf-8") as f:
        json.dump([asdict(t) for t in all_traces], f, indent=2)
    write_trace_csv(all_traces, out_dir / "token_arrivals.csv")
    summary = quantile_summary(all_traces)
    summary.to_csv(out_dir / "summary.csv", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(json.loads(summary.to_json(orient="records")), f, indent=2)
    plot_latency_bars(specs, summary, plot_dir / "latency_targets_normalized.png")
    plot_breakdown(summary, plot_dir / "latency_decomposition.png")
    plot_bytes(summary, plot_dir / "transmission_volume.png")
    print(f"Wrote LIVE scenario outputs to {out_dir}")
    print(f"Figures are in {plot_dir}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="SimPy-based scenario benchmark for split LLM transmission")
    ap.add_argument("--scenario", action="append", help="Path to a simple scenario YAML. Can be repeated.")
    ap.add_argument("--scenario_dir", help="Directory containing scenario YAML files.")
    ap.add_argument("--codec", help="Override codec preset, e.g. 8x, automix8, raw, custom name.")
    ap.add_argument("--codec_profile", help="JSON profile/AutoMix summary with compression ratio.")
    ap.add_argument("--network", help="Override network preset, e.g. 5g_weak, 5g_mid, 6g_edge.")
    ap.add_argument("--samples", type=int, default=300, help="Monte-Carlo trials per scenario. In --live mode, the model is run once per method and --samples controls network trials.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out_dir", default="./bench_out/scenario_simpy")

    # Live generation mode: actually load the split model once per method, then simulate network delay
    # from measured codec wire bytes.
    ap.add_argument("--live", action="store_true", help="Run actual split generation once for local/raw/codec before network simulation.")
    ap.add_argument("--front_dir", type=str, default="./split_out/front")
    ap.add_argument("--back_dir", type=str, default="./split_out/back")
    ap.add_argument("--revision", type=str, default=None)
    ap.add_argument("--tokenizer_id", type=str, default="Qwen/Qwen3-1.7B")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--local_files_only", action="store_true", help="Use local tokenizer/model files when supported.")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="auto")
    ap.add_argument("--front_quant", type=str, default="none")
    ap.add_argument("--back_quant", type=str, default="none")
    ap.add_argument("--timeout_sec", type=float, default=120.0)
    ap.add_argument("--raw_codec", type=str, default="default", help="Codec used for raw network split in --live mode.")
    ap.add_argument("--local_codec", type=str, default="default", help="Codec used for local compute-only line in --live mode.")
    ap.add_argument("--codec_label", type=str, default=None, help="Pretty label for the live codec curve.")
    ap.add_argument("--codec_extras_json", type=str, default=None)
    ap.add_argument(
        "--generation_mode",
        type=str,
        default="autoregressive",
        choices=["autoregressive", "self_speculative"],
        help="Generation mode passed to the SplitLLM runtime. Use self_speculative for LayerSkip self-speculative decoding.",
    )
    ap.add_argument("--num_speculations", type=int, default=3, help="Number of draft tokens for self-speculative decoding.")
    ap.add_argument(
        "--assistant_early_exit",
        type=int,
        default=None,
        help="Optional assistant early-exit override for self-speculative decoding.",
    )
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--prompt_file", type=str, default=None)
    ap.add_argument("--dataset_name", type=str, default=None)
    ap.add_argument("--dataset_config", type=str, default=None)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--text_column", type=str, default="text")
    ap.add_argument("--max_prompt_length", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=0)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--min_new_tokens", type=int, default=0)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    args = ap.parse_args(argv)

    if args.live:
        run_live_main(args)
        return

    out_dir = Path(args.out_dir)
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    specs: Dict[str, ScenarioSpec] = {}
    all_traces: List[SimulationTrace] = []
    for path in scenario_paths(args):
        spec = load_scenario(path, codec_override=args.codec, network_override=args.network, codec_profile=args.codec_profile)
        specs[spec.name] = spec
        traces = run_spec(spec, samples=args.samples, seed=args.seed)
        all_traces.extend(traces)
        plot_arrival(spec, traces, plot_dir / f"arrival_{spec.name}.png")

    # Save raw outputs.
    with open(out_dir / "scenario_config_resolved.json", "w", encoding="utf-8") as f:
        json.dump({k: asdict(v) for k, v in specs.items()}, f, indent=2)
    with open(out_dir / "traces.json", "w", encoding="utf-8") as f:
        json.dump([asdict(t) for t in all_traces], f, indent=2)
    write_trace_csv(all_traces, out_dir / "token_arrivals.csv")

    summary = quantile_summary(all_traces)
    summary.to_csv(out_dir / "summary.csv", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(json.loads(summary.to_json(orient="records")), f, indent=2)

    plot_latency_bars(specs, summary, plot_dir / "latency_targets_normalized.png")
    plot_breakdown(summary, plot_dir / "latency_decomposition.png")
    plot_bytes(summary, plot_dir / "transmission_volume.png")

    print(f"Wrote scenario outputs to {out_dir}")
    print(f"Figures are in {plot_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
