from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import sys
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from transformers import AutoTokenizer

from model.cli_common import build_codec
from model.codec import ActivationCodec, CodecContext, EncodedActivation
from runtime.common import (
    encoded_payload_size_bytes,
    load_front_model,
    pick_device_and_dtype,
    resolve_dir,
)
from viz.activation import (
    build_token_labels,
    choose_font,
    compute_color_limits,
    vector_to_grid,
)


@dataclass
class PlaygroundConfig:
    front_dir: str
    tokenizer_id: str
    revision: str | None
    trust_remote_code: bool
    local_files_only: bool
    device: str
    dtype: str
    front_quant: str
    back_dir: str
    back_quant: str
    out_dir: Path
    bench_out_dir: Path
    custom_dir: Path
    font_name: str | None
    font_path: str | None
    cmap: str
    dpi: int
    max_tokens: int
    tokens_per_figure: int
    topk_percent: float
    save_images: bool
    add_special_tokens: bool
    codec: str | None
    codec_extras_json: str


@dataclass
class PlaygroundState:
    config: PlaygroundConfig
    tokenizer: Any
    front: Any
    device: torch.device
    dtype: torch.dtype
    codec_choices: list[tuple[str, str]]
    lock: Lock


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Launch a Gradio playground for split-front activation heatmaps.",
    )
    p.add_argument("--front_dir", type=str, default="./split_out/front")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--tokenizer_id", type=str, default="Qwen/Qwen3-1.7B")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument("--front_quant", type=str, default="none")
    p.add_argument("--back_dir", type=str, default="./split_out/back")
    p.add_argument("--back_quant", type=str, default="none")
    p.add_argument("--custom_dir", type=str, default="./custom")
    p.add_argument("--out_dir", type=str, default="./playground_out/activation")
    p.add_argument("--bench_out_dir", type=str, default="./playground_out/bench")
    p.add_argument("--font_name", type=str, default=None)
    p.add_argument("--font_path", type=str, default=None)
    p.add_argument("--cmap", type=str, default="RdBu_r")
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--max_tokens", type=int, default=64)
    p.add_argument("--tokens_per_figure", type=int, default=5)
    p.add_argument(
        "--topk_percent",
        type=float,
        default=1.0,
        help="Percent of largest-magnitude hidden dims to zero before summary averaging.",
    )
    p.add_argument("--save_images", action="store_true")
    p.add_argument("--add_special_tokens", action="store_true")
    p.add_argument(
        "--codec",
        type=str,
        default=None,
        help="Default codec spec selected in the UI. Example: custom.topk_4bit_codec",
    )
    p.add_argument("--codec_extras_json", type=str, default="{}")
    p.add_argument("--server_name", type=str, default="127.0.0.1")
    p.add_argument("--server_port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    return p.parse_args()


def parse_codec_extras(raw: str | None) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("codec extras JSON must decode to an object")
    return dict(obj)


def discover_custom_codecs(custom_dir: Path) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    root = custom_dir.expanduser()
    if not root.is_dir():
        print(f"[warn] custom_dir not found: {root}")
        return [("identity_fp32 (builtin fallback)", "identity_fp32")]

    for path in sorted(root.glob("*.py")):
        if path.name == "__init__.py" or path.stem.startswith("_"):
            continue
        spec = f"{root.name}.{path.stem}"
        try:
            codec = build_codec(spec)
        except Exception as exc:
            print(f"[warn] skip codec file={path} error={exc}")
            continue
        choices.append((f"{codec.name} ({spec})", spec))

    if choices:
        print(f"[ok] discovered {len(choices)} custom codec(s)")
    else:
        print("[warn] no valid custom codecs found; fallback to builtin identity_fp32")
        choices.append(("identity_fp32 (builtin fallback)", "identity_fp32"))
    return choices


def topk_removed_average(activations: np.ndarray, topk_percent: float) -> np.ndarray:
    if activations.ndim != 2:
        raise ValueError(f"expected activations [tokens, hidden], got {activations.shape}")
    if topk_percent < 0.0 or topk_percent >= 100.0:
        raise ValueError("topk_percent must be >= 0 and < 100")

    hidden_size = int(activations.shape[1])
    k = int(np.ceil(hidden_size * float(topk_percent) / 100.0))
    if k <= 0:
        return activations.mean(axis=0)
    k = min(k, hidden_size - 1)

    masked = activations.copy()
    idx = np.argpartition(np.abs(masked), -k, axis=1)[:, -k:]
    rows = np.arange(masked.shape[0])[:, None]
    masked[rows, idx] = 0.0
    return masked.mean(axis=0)


def vector_to_grid_with_indices(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grid = vector_to_grid(vec)
    indices = np.full(grid.shape, -1, dtype=np.int64)
    indices.flat[: int(vec.shape[0])] = np.arange(int(vec.shape[0]), dtype=np.int64)
    return grid, indices


def load_playground(args: argparse.Namespace) -> PlaygroundState:
    cfg = PlaygroundConfig(
        front_dir=args.front_dir,
        tokenizer_id=args.tokenizer_id,
        revision=args.revision,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
        device=args.device,
        dtype=args.dtype,
        front_quant=args.front_quant,
        back_dir=args.back_dir,
        back_quant=args.back_quant,
        out_dir=Path(args.out_dir).expanduser().resolve(),
        bench_out_dir=Path(args.bench_out_dir).expanduser().resolve(),
        custom_dir=Path(args.custom_dir).expanduser(),
        font_name=args.font_name,
        font_path=args.font_path,
        cmap=args.cmap,
        dpi=int(args.dpi),
        max_tokens=int(args.max_tokens),
        tokens_per_figure=max(1, int(args.tokens_per_figure)),
        topk_percent=float(args.topk_percent),
        save_images=bool(args.save_images),
        add_special_tokens=bool(args.add_special_tokens),
        codec=args.codec,
        codec_extras_json=args.codec_extras_json,
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.bench_out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[info] loading tokenizer={cfg.tokenizer_id} front_dir={cfg.front_dir} "
        f"revision={cfg.revision}"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer_id,
        use_fast=True,
        trust_remote_code=cfg.trust_remote_code,
        local_files_only=cfg.local_files_only,
    )

    device, dtype = pick_device_and_dtype(cfg.device, cfg.dtype)
    front_local = resolve_dir(cfg.front_dir, revision=cfg.revision)
    front = load_front_model(
        front_local,
        device,
        dtype,
        quant_mode=cfg.front_quant,
    )
    choices = discover_custom_codecs(cfg.custom_dir)
    if cfg.codec:
        known_specs = {spec for _, spec in choices}
        if cfg.codec not in known_specs:
            try:
                codec = build_codec(cfg.codec)
            except Exception as exc:
                print(f"[warn] requested codec={cfg.codec} is invalid: {exc}")
            else:
                choices.append((f"{codec.name} ({cfg.codec})", cfg.codec))
    print(f"[ok] playground ready device={device} dtype={dtype}")

    return PlaygroundState(
        config=cfg,
        tokenizer=tokenizer,
        front=front,
        device=device,
        dtype=dtype,
        codec_choices=choices,
        lock=Lock(),
    )


def safe_wire_bytes(codec: ActivationCodec, encoded: EncodedActivation) -> int | None:
    try:
        return encoded_payload_size_bytes(codec, encoded)
    except Exception as exc:
        print(f"[warn] cannot measure codec wire bytes: {exc}")
        return None


def save_comparison_heatmaps(
    *,
    out_dir: Path,
    raw_activations: np.ndarray,
    codec_activations: np.ndarray,
    token_ids: list[int],
    token_labels: list[str],
    tokens_per_figure: int,
    codec_name: str,
    cmap: str,
    dpi: int,
    vmin: float,
    vmax: float,
) -> list[Path]:
    if raw_activations.shape != codec_activations.shape:
        raise ValueError(
            "raw and codec activations must have the same shape: "
            f"{raw_activations.shape} vs {codec_activations.shape}"
        )
    if raw_activations.ndim != 2:
        raise ValueError(
            "expected activations with shape [tokens, hidden], "
            f"got {raw_activations.shape}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    total_tokens = int(raw_activations.shape[0])
    per_fig = max(1, int(tokens_per_figure))
    saved: list[Path] = []

    fig_idx = 0
    for start in range(0, total_tokens, per_fig):
        end = min(start + per_fig, total_tokens)
        width = 3.2 * per_fig
        fig, axes = plt.subplots(
            nrows=2,
            ncols=per_fig,
            figsize=(width, 6.2),
            constrained_layout=True,
            squeeze=False,
        )
        mappable = None
        used_axes = []

        for slot in range(per_fig):
            token_index = start + slot
            raw_ax = axes[0, slot]
            codec_ax = axes[1, slot]
            if token_index >= end:
                raw_ax.axis("off")
                codec_ax.axis("off")
                continue

            raw_grid = vector_to_grid(raw_activations[token_index])
            codec_grid = vector_to_grid(codec_activations[token_index])
            mappable = raw_ax.imshow(
                raw_grid,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )
            codec_ax.imshow(
                codec_grid,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )
            used_axes.extend([raw_ax, codec_ax])

            for ax in (raw_ax, codec_ax):
                ax.set_xticks([])
                ax.set_yticks([])

            token_id = int(token_ids[token_index])
            label = token_labels[token_index]
            raw_ax.set_title(f"{token_index + 1}. id={token_id}\n{label}", fontsize=9)
            codec_ax.set_title(f"{token_index + 1}' after codec", fontsize=9)

        axes[0, 0].set_ylabel("raw", rotation=0, labelpad=20, va="center")
        axes[1, 0].set_ylabel("codec", rotation=0, labelpad=20, va="center")
        if mappable is not None:
            fig.colorbar(mappable, ax=used_axes, fraction=0.02, pad=0.01)

        fig.suptitle(
            f"Activation comparison ({start + 1} to {end}) - {codec_name}",
            fontsize=12,
        )
        out_path = out_dir / f"activation_compare_{fig_idx + 1:04d}.png"
        fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
        print(f"[progress] saved comparison heatmap: {out_path}")
        saved.append(out_path)
        fig_idx += 1

    return saved


def save_two_row_summary_heatmap(
    *,
    out_dir: Path,
    raw_vec: np.ndarray,
    codec_vec: np.ndarray,
    title: str,
    filename: str,
    codec_name: str,
    cmap: str,
    dpi: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = np.stack([raw_vec, codec_vec], axis=0)
    vmin, vmax = compute_color_limits(summary)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(8.0, 7.0),
        constrained_layout=True,
        squeeze=False,
    )
    used_axes = []
    mappable = None
    for ax, row_title, vec in (
        (axes[0, 0], "raw", raw_vec),
        (axes[1, 0], "codec", codec_vec),
    ):
        mappable = ax.imshow(
            vector_to_grid(vec),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(row_title, fontsize=11)
        used_axes.append(ax)

    if mappable is not None:
        fig.colorbar(mappable, ax=used_axes, fraction=0.03, pad=0.02)
    fig.suptitle(f"{title} - {codec_name}", fontsize=12)

    out_path = out_dir / filename
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    print(f"[progress] saved summary heatmap: {out_path}")
    return out_path


def save_summary_heatmaps(
    *,
    out_dir: Path,
    raw_activations: np.ndarray,
    codec_activations: np.ndarray,
    topk_percent: float,
    codec_name: str,
    cmap: str,
    dpi: int,
) -> list[Path]:
    raw_mean = raw_activations.mean(axis=0)
    codec_mean = codec_activations.mean(axis=0)
    raw_topk_mean = topk_removed_average(raw_activations, topk_percent)
    codec_topk_mean = topk_removed_average(codec_activations, topk_percent)

    average_path = save_two_row_summary_heatmap(
        out_dir=out_dir,
        raw_vec=raw_mean,
        codec_vec=codec_mean,
        title="Average activation",
        filename="activation_average.png",
        codec_name=codec_name,
        cmap=cmap,
        dpi=dpi,
    )
    topk_path = save_two_row_summary_heatmap(
        out_dir=out_dir,
        raw_vec=raw_topk_mean,
        codec_vec=codec_topk_mean,
        title=f"Average after removing top {topk_percent:g}%",
        filename="activation_average_without_topk.png",
        codec_name=codec_name,
        cmap=cmap,
        dpi=dpi,
    )
    return [average_path, topk_path]


def _add_activation_trace(
    *,
    fig: go.Figure,
    vec: np.ndarray,
    row: int,
    col: int,
    token_index: int | None,
    token_id: int | None,
    token_label: str | None,
    layer_name: str,
    colorscale: str,
    zmin: float,
    zmax: float,
    showscale: bool,
) -> None:
    grid, hidden_indices = vector_to_grid_with_indices(vec)
    yy, xx = np.indices(grid.shape)
    custom = np.stack([hidden_indices, yy, xx], axis=-1)
    if token_index is None:
        hover = (
            f"{layer_name}<br>"
            "hidden_dim=%{customdata[0]}<br>"
            "grid=(%{customdata[1]}, %{customdata[2]})<br>"
            "value=%{z:.8f}<extra></extra>"
        )
    else:
        hover = (
            f"{layer_name}<br>"
            f"token_index={token_index} id={token_id}<br>"
            f"token={html.escape(str(token_label))}<br>"
            "hidden_dim=%{customdata[0]}<br>"
            "grid=(%{customdata[1]}, %{customdata[2]})<br>"
            "value=%{z:.8f}<extra></extra>"
        )
    fig.add_trace(
        go.Heatmap(
            z=grid,
            customdata=custom,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            hovertemplate=hover,
            hoverongaps=False,
            showscale=showscale,
            colorbar={"title": "activation"} if showscale else None,
        ),
        row=row,
        col=col,
    )


def keep_heatmap_cells_square(fig: go.Figure) -> None:
    fig.update_xaxes(visible=False, constrain="domain")
    fig.update_yaxes(visible=False, autorange="reversed", constrain="domain")
    for axis in fig.select_yaxes():
        if axis.anchor:
            axis.update(scaleanchor=axis.anchor, scaleratio=1)


def build_interactive_token_figure(
    *,
    raw_activations: np.ndarray,
    codec_activations: np.ndarray,
    token_ids: list[int],
    token_labels: list[str],
    token_start: int,
    tokens_per_figure: int,
    codec_name: str,
    colorscale: str,
    vmin: float,
    vmax: float,
) -> go.Figure:
    token_count = int(raw_activations.shape[0])
    start = max(0, min(int(token_start), token_count - 1))
    end = min(start + max(1, int(tokens_per_figure)), token_count)
    cols = end - start
    subplot_titles: list[str] = []
    for row_name in ("raw", "codec"):
        for token_index in range(start, end):
            token_id = int(token_ids[token_index])
            label = html.escape(str(token_labels[token_index]))
            suffix = "'" if row_name == "codec" else ""
            subplot_titles.append(f"{token_index + 1}{suffix}. id={token_id}<br>{label}")

    fig = make_subplots(
        rows=2,
        cols=cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.12,
    )
    for col in range(cols):
        token_index = start + col
        _add_activation_trace(
            fig=fig,
            vec=raw_activations[token_index],
            row=1,
            col=col + 1,
            token_index=token_index,
            token_id=int(token_ids[token_index]),
            token_label=token_labels[token_index],
            layer_name="raw",
            colorscale=colorscale,
            zmin=vmin,
            zmax=vmax,
            showscale=(col == cols - 1),
        )
        _add_activation_trace(
            fig=fig,
            vec=codec_activations[token_index],
            row=2,
            col=col + 1,
            token_index=token_index,
            token_id=int(token_ids[token_index]),
            token_label=token_labels[token_index],
            layer_name="codec",
            colorscale=colorscale,
            zmin=vmin,
            zmax=vmax,
            showscale=False,
        )

    keep_heatmap_cells_square(fig)
    fig.update_layout(
        title=f"Interactive token activations ({start + 1} to {end}) - {codec_name}",
        height=620,
        margin={"l": 30, "r": 30, "t": 90, "b": 30},
    )
    return fig


def build_interactive_summary_figure(
    *,
    raw_activations: np.ndarray,
    codec_activations: np.ndarray,
    topk_percent: float,
    codec_name: str,
    colorscale: str,
) -> go.Figure:
    raw_mean = raw_activations.mean(axis=0)
    codec_mean = codec_activations.mean(axis=0)
    raw_topk_mean = topk_removed_average(raw_activations, topk_percent)
    codec_topk_mean = topk_removed_average(codec_activations, topk_percent)
    summary = np.stack([raw_mean, codec_mean, raw_topk_mean, codec_topk_mean], axis=0)
    vmin, vmax = compute_color_limits(summary)

    titles = [
        "raw average",
        "codec average",
        f"raw average without top {topk_percent:g}%",
        f"codec average without top {topk_percent:g}%",
    ]
    vecs = [raw_mean, codec_mean, raw_topk_mean, codec_topk_mean]
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.14,
    )
    for idx, vec in enumerate(vecs):
        row = idx // 2 + 1
        col = idx % 2 + 1
        _add_activation_trace(
            fig=fig,
            vec=vec,
            row=row,
            col=col,
            token_index=None,
            token_id=None,
            token_label=None,
            layer_name=titles[idx],
            colorscale=colorscale,
            zmin=vmin,
            zmax=vmax,
            showscale=(idx == 1),
        )

    keep_heatmap_cells_square(fig)
    fig.update_layout(
        title=f"Interactive average summaries - {codec_name}",
        height=740,
        margin={"l": 30, "r": 30, "t": 90, "b": 30},
    )
    return fig


def benchmark_output_path(cfg: PlaygroundConfig, name: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    cfg.bench_out_dir.mkdir(parents=True, exist_ok=True)
    return cfg.bench_out_dir / f"{name}_{stamp}.json"


def append_common_benchmark_args(
    cmd: list[str],
    *,
    cfg: PlaygroundConfig,
    codec_spec: str,
    codec_extras_json: str,
    out_json: Path,
) -> None:
    cmd.extend(
        [
            "--front_dir",
            cfg.front_dir,
            "--back_dir",
            cfg.back_dir,
            "--tokenizer_id",
            cfg.tokenizer_id,
            "--device",
            cfg.device,
            "--dtype",
            cfg.dtype,
            "--front_quant",
            cfg.front_quant,
            "--back_quant",
            cfg.back_quant,
            "--codec",
            codec_spec,
            "--out_json",
            str(out_json),
        ]
    )
    if cfg.revision:
        cmd.extend(["--revision", cfg.revision])
    if cfg.trust_remote_code:
        cmd.append("--trust_remote_code")
    if cfg.local_files_only:
        cmd.append("--local_files_only")
    if codec_extras_json.strip():
        cmd.extend(["--codec_extras_json", codec_extras_json])


def benchmark_progress_fraction(kind: str, line: str, total_hint: int) -> float | None:
    if kind == "ppl":
        m = re.search(r"processed\s+(\d+)/(\d+)", line)
        if m:
            done = int(m.group(1))
            total = max(1, int(m.group(2)))
            return min(0.98, done / total)
        return None

    m = re.search(r"prepared\s+(\d+)/(\d+)", line)
    if m:
        done = int(m.group(1))
        total = max(1, int(m.group(2)))
        return min(0.20, 0.20 * done / total)

    m = re.search(r"generated=(\d+)", line)
    if m and total_hint > 0:
        done = int(m.group(1))
        total = max(1, int(total_hint))
        return min(0.82, 0.20 + 0.60 * done / total)

    m = re.search(r"scored\s+(\d+)/(\d+)", line)
    if m:
        done = int(m.group(1))
        total = max(1, int(m.group(2)))
        return min(0.98, 0.82 + 0.16 * done / total)

    return None


def run_benchmark_command(
    *,
    cmd: list[str],
    out_json: Path,
    kind: str,
    total_hint: int,
    progress: gr.Progress,
) -> tuple[str, dict[str, Any] | None]:
    log_lines: list[str] = []
    progress(0.0, desc=f"starting {kind}")
    print(f"[info] benchmark command: {' '.join(cmd)}")

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.rstrip()
        if not line:
            continue
        log_lines.append(line)
        print(line)
        fraction = benchmark_progress_fraction(kind, line, total_hint)
        if fraction is not None:
            progress(fraction, desc=line[:120])
        elif line.startswith("[info]") or line.startswith("[ok]") or line.startswith("[result]"):
            progress(None, desc=line[:120])

    return_code = proc.wait()
    if return_code != 0:
        progress(1.0, desc=f"{kind} failed")
        log_lines.append(f"[error] benchmark process exited with code {return_code}")
        return "\n".join(log_lines[-400:]), None

    result: dict[str, Any] | None = None
    if out_json.is_file():
        result = json.loads(out_json.read_text(encoding="utf-8"))
    progress(1.0, desc=f"{kind} finished")
    log_lines.append(f"[ok] result_json={out_json}")
    return "\n".join(log_lines[-400:]), result


def run_ppl_benchmark(
    *,
    state: PlaygroundState,
    codec_spec: str,
    codec_extras_json: str,
    dataset_name: str,
    dataset_config: str,
    split: str,
    text_column: str,
    samples: int,
    max_length: int,
    batch_size: int,
    progress_every: int,
    progress: gr.Progress,
) -> tuple[str, dict[str, Any] | None]:
    cfg = state.config
    out_json = benchmark_output_path(cfg, "ppl")
    cmd = [sys.executable, "-u", "-m", "bench.ppl"]
    append_common_benchmark_args(
        cmd,
        cfg=cfg,
        codec_spec=codec_spec,
        codec_extras_json=codec_extras_json,
        out_json=out_json,
    )
    cmd.extend(
        [
            "--dataset_name",
            str(dataset_name),
            "--dataset_config",
            str(dataset_config),
            "--split",
            str(split),
            "--text_column",
            str(text_column),
            "--samples",
            str(int(samples)),
            "--max_length",
            str(int(max_length)),
            "--batch_size",
            str(int(batch_size)),
            "--progress_every",
            str(max(1, int(progress_every))),
        ]
    )
    return run_benchmark_command(
        cmd=cmd,
        out_json=out_json,
        kind="ppl",
        total_hint=max(1, int(samples)),
        progress=progress,
    )


def run_mmlu_benchmark(
    *,
    state: PlaygroundState,
    codec_spec: str,
    codec_extras_json: str,
    dataset_name: str,
    dataset_config: str,
    fewshot_split: str,
    eval_split: str,
    subjects: str,
    n_shot: int,
    samples: int,
    max_samples_per_subject: int,
    max_length: int,
    max_new_tokens: int,
    progress_every: int,
    progress: gr.Progress,
) -> tuple[str, dict[str, Any] | None]:
    cfg = state.config
    out_json = benchmark_output_path(cfg, "mmlu")
    cmd = [sys.executable, "-u", "-m", "bench.mmlu"]
    append_common_benchmark_args(
        cmd,
        cfg=cfg,
        codec_spec=codec_spec,
        codec_extras_json=codec_extras_json,
        out_json=out_json,
    )
    cmd.extend(
        [
            "--dataset_name",
            str(dataset_name),
            "--dataset_config",
            str(dataset_config),
            "--fewshot_split",
            str(fewshot_split),
            "--eval_split",
            str(eval_split),
            "--n_shot",
            str(int(n_shot)),
            "--samples",
            str(int(samples)),
            "--max_samples_per_subject",
            str(int(max_samples_per_subject)),
            "--max_length",
            str(int(max_length)),
            "--max_new_tokens",
            str(int(max_new_tokens)),
            "--progress_every",
            str(max(1, int(progress_every))),
        ]
    )
    if str(subjects).strip():
        cmd.extend(["--subjects", str(subjects).strip()])
    total_hint = int(samples) if int(samples) > 0 else 0
    return run_benchmark_command(
        cmd=cmd,
        out_json=out_json,
        kind="mmlu",
        total_hint=total_hint,
        progress=progress,
    )


@torch.no_grad()
def render_activation_gallery(
    *,
    state: PlaygroundState,
    text: str,
    codec_spec: str,
    codec_extras_json: str,
    max_tokens: int,
    token_start: int,
    tokens_per_figure: int,
    topk_percent: float,
    save_images: bool,
    add_special_tokens: bool,
) -> tuple[list[str], go.Figure, go.Figure, str]:
    if not text:
        raise gr.Error("Input text is empty.")

    cfg = state.config
    with state.lock:
        t0 = time.perf_counter()
        try:
            topk_percent = float(topk_percent)
        except (TypeError, ValueError) as exc:
            raise gr.Error("Top-k percent must be a number.") from exc
        if not np.isfinite(topk_percent) or topk_percent < 0.0 or topk_percent >= 100.0:
            raise gr.Error("Top-k percent must be >= 0 and < 100.")

        font_choice = choose_font(
            input_text=text,
            font_name=cfg.font_name,
            font_path=cfg.font_path,
        )
        if not font_choice.supports_input_text:
            raise gr.Error(
                "Selected font cannot render every non-ASCII input character. "
                "Pass --font_path with a CJK-capable font."
            )

        enc = state.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=bool(add_special_tokens),
        )
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))
        if input_ids.numel() == 0:
            raise gr.Error("Tokenization produced zero tokens.")

        token_limit = int(max_tokens)
        if token_limit > 0 and input_ids.shape[1] > token_limit:
            input_ids = input_ids[:, :token_limit]
            attention_mask = attention_mask[:, :token_limit]

        token_ids = [int(x) for x in input_ids[0].tolist()]
        token_labels = build_token_labels(state.tokenizer, token_ids)
        codec_extras = parse_codec_extras(codec_extras_json)
        codec = build_codec(codec_spec)

        amp_enabled = (
            state.device.type == "cuda"
            and state.dtype in (torch.float16, torch.bfloat16)
        )
        with (
            torch.autocast(device_type="cuda", dtype=state.dtype)
            if amp_enabled
            else nullcontext()
        ):
            out = state.front.model(
                input_ids=input_ids.to(state.device),
                attention_mask=attention_mask.to(state.device),
                use_cache=False,
                return_dict=True,
            )

        raw_hidden = out.last_hidden_state.detach()
        raw_np = raw_hidden[0].float().cpu().numpy()

        session_id = str(uuid.uuid4())
        context = CodecContext(
            phase="prefill",
            side="playground",
            session_id=session_id,
            seq_len=len(token_ids),
            step=0,
            extras=codec_extras,
        )
        codec.start_session(session_id)
        try:
            encoded = codec.encode(raw_hidden.clone(), context=context)
            wire_bytes = safe_wire_bytes(codec, encoded)
            decoded = codec.decode(
                encoded,
                context=context,
                device=state.device,
                dtype=state.dtype,
            )
        finally:
            codec.end_session(session_id)

        if tuple(decoded.shape) != tuple(raw_hidden.shape):
            raise gr.Error(
                "Codec output shape does not match raw activation shape: "
                f"{tuple(decoded.shape)} vs {tuple(raw_hidden.shape)}"
            )

        codec_np = decoded[0].detach().float().cpu().numpy()
        combined = np.concatenate([raw_np, codec_np], axis=0)
        vmin, vmax = compute_color_limits(combined)

        token_fig = build_interactive_token_figure(
            raw_activations=raw_np,
            codec_activations=codec_np,
            token_ids=token_ids,
            token_labels=token_labels,
            token_start=int(token_start),
            tokens_per_figure=int(tokens_per_figure),
            codec_name=codec.name,
            colorscale="RdBu",
            vmin=vmin,
            vmax=vmax,
        )
        summary_fig = build_interactive_summary_figure(
            raw_activations=raw_np,
            codec_activations=codec_np,
            topk_percent=float(topk_percent),
            codec_name=codec.name,
            colorscale="RdBu",
        )

        paths: list[Path] = []
        run_dir: Path | None = None
        if bool(save_images):
            run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
            run_dir = cfg.out_dir / run_id
            paths = save_comparison_heatmaps(
                out_dir=run_dir,
                raw_activations=raw_np,
                codec_activations=codec_np,
                token_ids=token_ids,
                token_labels=token_labels,
                tokens_per_figure=int(tokens_per_figure),
                codec_name=codec.name,
                cmap=cfg.cmap,
                dpi=cfg.dpi,
                vmin=vmin,
                vmax=vmax,
            )
            summary_paths = save_summary_heatmaps(
                out_dir=run_dir,
                raw_activations=raw_np,
                codec_activations=codec_np,
                topk_percent=float(topk_percent),
                codec_name=codec.name,
                cmap=cfg.cmap,
                dpi=cfg.dpi,
            )
            paths.extend(summary_paths)

        diff = raw_np - codec_np
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff * diff)))
        max_abs = float(np.max(np.abs(diff)))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        wire_text = "unknown" if wire_bytes is None else str(int(wire_bytes))
        meta_text = ""
        meta_fields = []
        for key in ("mode", "topk_ratio", "topk_count", "drop_ratio", "drop_count", "payload_nbytes"):
            if key in encoded.meta:
                meta_fields.append(f"{key}={encoded.meta[key]}")
        if meta_fields:
            meta_text = " " + " ".join(meta_fields)
        status = (
            f"[ok] token_count={len(token_ids)} hidden_size={raw_np.shape[1]} "
            f"codec={codec.name} wire_bytes={wire_text} "
            f"token_start={int(token_start)} topk_percent={float(topk_percent):g} "
            f"mae={mae:.6g} rmse={rmse:.6g} max_abs={max_abs:.6g}{meta_text} "
            f"images={len(paths)} elapsed_ms={elapsed_ms:.1f}"
        )
        if run_dir is not None:
            status += f" out_dir={run_dir}"
        return [str(path) for path in paths], token_fig, summary_fig, status


def build_demo(state: PlaygroundState) -> gr.Blocks:
    cfg = state.config
    available_specs = {spec for _, spec in state.codec_choices}
    default_codec = cfg.codec if cfg.codec in available_specs else state.codec_choices[0][1]

    with gr.Blocks(title="SplitLLM Activation Playground") as demo:
        gr.Markdown("# SplitLLM Activation Playground")
        gr.Markdown(
            f"front_dir: `{cfg.front_dir}` | tokenizer: `{cfg.tokenizer_id}` | "
            f"device: `{state.device}` | dtype: `{state.dtype}`"
        )

        with gr.Row():
            text = gr.Textbox(
                label="Input text",
                lines=6,
                placeholder="Type text and render split-front activations.",
            )
            with gr.Column():
                codec = gr.Dropdown(
                    label="Codec",
                    choices=state.codec_choices,
                    value=default_codec,
                )
                codec_extras = gr.Textbox(
                    label="Codec extras JSON",
                    value=cfg.codec_extras_json,
                    lines=4,
                )
                max_tokens = gr.Slider(
                    label="Max tokens",
                    minimum=0,
                    maximum=512,
                    step=1,
                    value=max(0, cfg.max_tokens),
                )
                token_start = gr.Slider(
                    label="Token start index",
                    minimum=0,
                    maximum=512,
                    step=1,
                    value=0,
                )
                tokens_per_figure = gr.Slider(
                    label="Tokens per view",
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=max(1, cfg.tokens_per_figure),
                )
                topk_percent = gr.Number(
                    label="Top-k remove percent",
                    value=cfg.topk_percent,
                    precision=2,
                )
                save_images = gr.Checkbox(
                    label="Save PNG gallery",
                    value=cfg.save_images,
                )
                add_special_tokens = gr.Checkbox(
                    label="Add special tokens",
                    value=cfg.add_special_tokens,
                )
                render = gr.Button("Render", variant="primary")

        token_plot = gr.Plot(label="Interactive token activations")
        summary_plot = gr.Plot(label="Interactive average summaries")
        with gr.Accordion("Saved PNG gallery", open=False):
            gallery = gr.Gallery(
                label="Raw / codec activation heatmaps",
                columns=1,
                object_fit="contain",
                height="auto",
            )
        status = gr.Textbox(label="Status", lines=3)

        render.click(
            fn=lambda user_text, codec_spec, extras, limit, start_idx, per_fig, topk_pct, save_png, add_special: (
                render_activation_gallery(
                    state=state,
                    text=user_text,
                    codec_spec=codec_spec,
                    codec_extras_json=extras,
                    max_tokens=int(limit),
                    token_start=int(start_idx),
                    tokens_per_figure=int(per_fig),
                    topk_percent=topk_pct,
                    save_images=bool(save_png),
                    add_special_tokens=bool(add_special),
                )
            ),
            inputs=[
                text,
                codec,
                codec_extras,
                max_tokens,
                token_start,
                tokens_per_figure,
                topk_percent,
                save_images,
                add_special_tokens,
            ],
            outputs=[gallery, token_plot, summary_plot, status],
        )

        gr.Markdown("## Benchmarks")
        gr.Markdown(
            f"back_dir: `{cfg.back_dir}` | back_quant: `{cfg.back_quant}` | "
            f"outputs: `{cfg.bench_out_dir}`"
        )
        benchmark_log = gr.Textbox(label="Benchmark log", lines=16)
        benchmark_result = gr.JSON(label="Benchmark result JSON")

        def run_ppl_ui(
            codec_spec,
            extras,
            dataset_name,
            dataset_config,
            split,
            text_column,
            samples,
            max_len,
            batch_size,
            progress_every,
            progress=gr.Progress(),
        ):
            return run_ppl_benchmark(
                state=state,
                codec_spec=codec_spec,
                codec_extras_json=extras,
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                split=split,
                text_column=text_column,
                samples=int(samples),
                max_length=int(max_len),
                batch_size=int(batch_size),
                progress_every=int(progress_every),
                progress=progress,
            )

        def run_mmlu_ui(
            codec_spec,
            extras,
            dataset_name,
            dataset_config,
            fewshot_split,
            eval_split,
            subjects,
            n_shot,
            samples,
            max_samples_per_subject,
            max_len,
            max_new_tokens,
            progress_every,
            progress=gr.Progress(),
        ):
            return run_mmlu_benchmark(
                state=state,
                codec_spec=codec_spec,
                codec_extras_json=extras,
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                fewshot_split=fewshot_split,
                eval_split=eval_split,
                subjects=subjects,
                n_shot=int(n_shot),
                samples=int(samples),
                max_samples_per_subject=int(max_samples_per_subject),
                max_length=int(max_len),
                max_new_tokens=int(max_new_tokens),
                progress_every=int(progress_every),
                progress=progress,
            )

        with gr.Accordion("PPL", open=True):
            with gr.Row():
                ppl_dataset_name = gr.Textbox(label="Dataset", value="wikitext")
                ppl_dataset_config = gr.Textbox(label="Config", value="wikitext-2-raw-v1")
                ppl_split = gr.Textbox(label="Split", value="test")
                ppl_text_column = gr.Textbox(label="Text column", value="text")
            with gr.Row():
                ppl_samples = gr.Number(label="Samples", value=32, precision=0)
                ppl_max_length = gr.Number(label="Max length", value=256, precision=0)
                ppl_batch_size = gr.Number(label="Batch size", value=4, precision=0)
                ppl_progress_every = gr.Number(label="Progress every", value=1, precision=0)
            run_ppl = gr.Button("Run PPL", variant="secondary")

        with gr.Accordion("MMLU", open=False):
            with gr.Row():
                mmlu_dataset_name = gr.Textbox(label="Dataset", value="cais/mmlu")
                mmlu_dataset_config = gr.Textbox(label="Config", value="all")
                mmlu_fewshot_split = gr.Textbox(label="Few-shot split", value="dev")
                mmlu_eval_split = gr.Textbox(label="Eval split", value="test")
            mmlu_subjects = gr.Textbox(
                label="Subjects",
                value="",
                placeholder="Optional comma-separated subjects",
            )
            with gr.Row():
                mmlu_n_shot = gr.Number(label="n-shot", value=5, precision=0)
                mmlu_samples = gr.Number(label="Samples", value=20, precision=0)
                mmlu_max_samples_per_subject = gr.Number(
                    label="Max samples / subject",
                    value=-1,
                    precision=0,
                )
                mmlu_max_length = gr.Number(label="Max length", value=2048, precision=0)
            with gr.Row():
                mmlu_max_new_tokens = gr.Number(label="Max new tokens", value=1, precision=0)
                mmlu_progress_every = gr.Number(label="Progress every", value=1, precision=0)
            run_mmlu = gr.Button("Run MMLU", variant="secondary")

        run_ppl.click(
            fn=run_ppl_ui,
            inputs=[
                codec,
                codec_extras,
                ppl_dataset_name,
                ppl_dataset_config,
                ppl_split,
                ppl_text_column,
                ppl_samples,
                ppl_max_length,
                ppl_batch_size,
                ppl_progress_every,
            ],
            outputs=[benchmark_log, benchmark_result],
        )
        run_mmlu.click(
            fn=run_mmlu_ui,
            inputs=[
                codec,
                codec_extras,
                mmlu_dataset_name,
                mmlu_dataset_config,
                mmlu_fewshot_split,
                mmlu_eval_split,
                mmlu_subjects,
                mmlu_n_shot,
                mmlu_samples,
                mmlu_max_samples_per_subject,
                mmlu_max_length,
                mmlu_max_new_tokens,
                mmlu_progress_every,
            ],
            outputs=[benchmark_log, benchmark_result],
        )

    return demo


def main() -> None:
    args = parse_args()
    state = load_playground(args)
    demo = build_demo(state)
    demo.queue()
    demo.launch(
        server_name=args.server_name,
        server_port=int(args.server_port),
        share=bool(args.share),
    )


if __name__ == "__main__":
    main()
