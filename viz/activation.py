from __future__ import annotations

import argparse
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager, ft2font
from transformers import AutoTokenizer

from model.runtime import load_front_model, pick_device_and_dtype, resolve_dir


@dataclass
class FontChoice:
    name: str
    path: str | None
    supports_input_text: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Visualize split-front token activations as heatmaps. "
            "Outputs 6 tokens per image and one final average image."
        ),
    )
    p.add_argument("--front_dir", type=str, default="./split_out/front")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--tokenizer_id", type=str, default="Qwen/Qwen3-1.7B")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--text", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./viz_out/activation")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument("--add_special_tokens", action="store_true")
    p.add_argument(
        "--max_tokens",
        type=int,
        default=0,
        help="0 means use all tokens.",
    )
    p.add_argument("--tokens_per_figure", type=int, default=6)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--cmap", type=str, default="RdBu_r")
    p.add_argument("--font_name", type=str, default=None)
    p.add_argument(
        "--font_path",
        type=str,
        default=None,
        help="Optional path to a TTF/TTC/OTF font file for CJK text.",
    )
    return p.parse_args()


def _collect_non_ascii_chars(text: str) -> str:
    return "".join(ch for ch in text if ord(ch) > 127 and not ch.isspace())


def _font_supports_text(font_path: str, text: str) -> bool:
    if not text:
        return True
    try:
        ft = ft2font.FT2Font(font_path)
        cmap = ft.get_charmap()
    except Exception:
        return False

    for ch in text:
        if ord(ch) not in cmap:
            return False
    return True


def _apply_font(name: str) -> None:
    plt.rcParams["font.family"] = [name, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def choose_font(
    *,
    input_text: str,
    font_name: str | None,
    font_path: str | None,
) -> FontChoice:
    target_chars = _collect_non_ascii_chars(input_text)
    fonts = list(font_manager.fontManager.ttflist)

    if font_path:
        p = Path(font_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"font_path not found: {p}")
        font_manager.fontManager.addfont(str(p))
        prop = font_manager.FontProperties(fname=str(p))
        used_name = prop.get_name()
        _apply_font(used_name)
        return FontChoice(
            name=used_name,
            path=str(p),
            supports_input_text=_font_supports_text(str(p), target_chars),
        )

    if font_name:
        named = [f for f in fonts if f.name == font_name]
        if not named:
            raise ValueError(f"font_name not found in matplotlib font list: {font_name}")
        for entry in named:
            if _font_supports_text(entry.fname, target_chars):
                _apply_font(entry.name)
                return FontChoice(
                    name=entry.name,
                    path=entry.fname,
                    supports_input_text=True,
                )
        entry = named[0]
        _apply_font(entry.name)
        return FontChoice(
            name=entry.name,
            path=entry.fname,
            supports_input_text=_font_supports_text(entry.fname, target_chars),
        )

    preferred = [
        "Noto Sans CJK TC",
        "Noto Serif CJK TC",
        "Noto Sans TC",
        "Source Han Sans TC",
        "Source Han Sans TW",
        "Microsoft JhengHei",
        "PingFang TC",
        "Heiti TC",
        "WenQuanYi Zen Hei",
        "AR PL UMing TW",
        "AR PL UKai TW",
        "SimHei",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
    ]

    for name in preferred:
        matched = [f for f in fonts if f.name == name]
        for entry in matched:
            if _font_supports_text(entry.fname, target_chars):
                _apply_font(entry.name)
                return FontChoice(
                    name=entry.name,
                    path=entry.fname,
                    supports_input_text=True,
                )

    if target_chars:
        for entry in fonts:
            if _font_supports_text(entry.fname, target_chars):
                _apply_font(entry.name)
                return FontChoice(
                    name=entry.name,
                    path=entry.fname,
                    supports_input_text=True,
                )

    fallback = "DejaVu Sans"
    _apply_font(fallback)
    return FontChoice(
        name=fallback,
        path=None,
        supports_input_text=(not bool(target_chars)),
    )


def clean_label(text: str, max_len: int = 24) -> str:
    out = text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    if out == " ":
        out = "<space>"
    if out == "":
        out = "<empty>"
    if len(out) > max_len:
        out = out[: max_len - 3] + "..."
    return out


def build_token_labels(tokenizer, token_ids: list[int]) -> list[str]:
    labels: list[str] = []
    for token_id in token_ids:
        decoded = tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if decoded == "":
            decoded = tokenizer.convert_ids_to_tokens([int(token_id)])[0]
        labels.append(clean_label(decoded))
    return labels


def vector_to_grid(vec: np.ndarray) -> np.ndarray:
    dim = int(vec.shape[0])
    rows = max(1, int(math.sqrt(dim)))
    cols = int(math.ceil(dim / rows))
    grid = np.full((rows, cols), np.nan, dtype=np.float32)
    grid.flat[:dim] = vec.astype(np.float32, copy=False)
    return grid


def compute_color_limits(activations: np.ndarray) -> tuple[float, float]:
    flat = activations.reshape(-1).astype(np.float32, copy=False)
    lo = float(np.percentile(flat, 1.0))
    hi = float(np.percentile(flat, 99.0))
    bound = max(abs(lo), abs(hi))
    if not np.isfinite(bound) or bound <= 0.0:
        bound = 1.0
    return -bound, bound


def save_group_heatmaps(
    *,
    out_dir: Path,
    activations: np.ndarray,
    token_ids: list[int],
    token_labels: list[str],
    tokens_per_figure: int,
    cmap: str,
    dpi: int,
    vmin: float,
    vmax: float,
) -> list[Path]:
    if activations.ndim != 2:
        raise ValueError(f"expected activations with shape [tokens, hidden], got {activations.shape}")

    total_tokens = int(activations.shape[0])
    per_fig = max(1, int(tokens_per_figure))
    ncols = 3
    nrows = int(math.ceil(per_fig / ncols))
    saved: list[Path] = []

    fig_idx = 0
    for start in range(0, total_tokens, per_fig):
        end = min(start + per_fig, total_tokens)
        fig_idx += 1

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(4.8 * ncols, 3.2 * nrows),
            constrained_layout=True,
        )
        axes_arr = np.asarray(axes).reshape(-1)
        used_axes = []
        mappable = None

        for slot in range(len(axes_arr)):
            ax = axes_arr[slot]
            token_index = start + slot
            if token_index >= end:
                ax.axis("off")
                continue

            grid = vector_to_grid(activations[token_index])
            mappable = ax.imshow(
                grid,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )
            used_axes.append(ax)
            ax.set_xticks([])
            ax.set_yticks([])
            token_id = int(token_ids[token_index])
            token_label = token_labels[token_index]
            ax.set_title(f"token[{token_index}] id={token_id}\n{token_label}", fontsize=10)

        if mappable is not None:
            fig.colorbar(mappable, ax=used_axes, fraction=0.02, pad=0.02)

        fig.suptitle(f"Token activation heatmaps ({start} to {end - 1})", fontsize=12)
        out_path = out_dir / f"activation_{fig_idx:04d}.png"
        fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
        print(f"[progress] saved token heatmaps: {out_path}")
        saved.append(out_path)

    return saved


def save_average_heatmap(
    *,
    out_dir: Path,
    activations: np.ndarray,
    cmap: str,
    dpi: int,
    vmin: float,
    vmax: float,
    image_index: int,
) -> Path:
    mean_vec = activations.mean(axis=0)
    grid = vector_to_grid(mean_vec)

    fig, ax = plt.subplots(figsize=(8.0, 6.0), constrained_layout=True)
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Average activation over {activations.shape[0]} tokens")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)

    out_path = out_dir / f"activation_{image_index:04d}_average.png"
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    print(f"[progress] saved average heatmap: {out_path}")
    return out_path


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    font_choice = choose_font(
        input_text=args.text,
        font_name=args.font_name,
        font_path=args.font_path,
    )
    print(f"[info] font={font_choice.name} path={font_choice.path}")
    if not font_choice.supports_input_text:
        raise RuntimeError(
            "selected font cannot render all non-ASCII chars in --text. "
            "please pass --font_path to a Traditional Chinese capable font file"
        )

    print(
        f"[info] loading tokenizer={args.tokenizer_id} "
        f"front_dir={args.front_dir} revision={args.revision}"
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_id,
            use_fast=True,
            trust_remote_code=bool(args.trust_remote_code),
            local_files_only=bool(args.local_files_only),
        )
    except Exception as exc:
        raise RuntimeError(
            "failed to load tokenizer. If you are offline, set --tokenizer_id to a local "
            "tokenizer directory and pass --local_files_only."
        ) from exc

    device, dtype = pick_device_and_dtype(args.device, args.dtype)
    front_local = resolve_dir(args.front_dir, revision=args.revision)
    front = load_front_model(front_local, device, dtype)

    enc = tokenizer(
        args.text,
        return_tensors="pt",
        add_special_tokens=bool(args.add_special_tokens),
    )
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))

    if input_ids.numel() == 0:
        raise ValueError("tokenization produced zero tokens")

    if int(args.max_tokens) > 0:
        max_tokens = int(args.max_tokens)
        if input_ids.shape[1] > max_tokens:
            input_ids = input_ids[:, :max_tokens]
            attention_mask = attention_mask[:, :max_tokens]

    token_ids = [int(x) for x in input_ids[0].tolist()]
    token_labels = build_token_labels(tokenizer, token_ids)
    print(
        f"[info] token_count={len(token_ids)} "
        f"hidden_size={int(front.config.hidden_size)}"
    )

    amp_enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    with (
        torch.autocast(device_type="cuda", dtype=dtype)
        if amp_enabled
        else nullcontext()
    ):
        out = front.model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            use_cache=False,
            return_dict=True,
        )

    activations = out.last_hidden_state[0].detach().float().cpu().numpy()
    vmin, vmax = compute_color_limits(activations)

    saved_groups = save_group_heatmaps(
        out_dir=out_dir,
        activations=activations,
        token_ids=token_ids,
        token_labels=token_labels,
        tokens_per_figure=int(args.tokens_per_figure),
        cmap=args.cmap,
        dpi=int(args.dpi),
        vmin=vmin,
        vmax=vmax,
    )
    avg_path = save_average_heatmap(
        out_dir=out_dir,
        activations=activations,
        cmap=args.cmap,
        dpi=int(args.dpi),
        vmin=vmin,
        vmax=vmax,
        image_index=len(saved_groups) + 1,
    )

    print(f"[ok] saved {len(saved_groups)} token heatmap images")
    print(f"[ok] average image: {avg_path}")
    print(f"[ok] output_dir={out_dir}")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
