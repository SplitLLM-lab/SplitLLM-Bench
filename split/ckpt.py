from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class SplitAdapter:
    """Key routing rules for a decoder-only checkpoint."""

    name: str
    embed_prefixes: tuple[str, ...]
    back_prefixes: tuple[str, ...]
    layer_patterns: tuple[re.Pattern[str], ...]


@dataclass
class SplitStats:
    scanned_keys: int
    front_keys: int
    back_keys: int
    other_keys: int
    max_layer_index: int


def compile_patterns(*patterns: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p) for p in patterns)


DECODER_STANDARD = SplitAdapter(
    name="decoder_standard",
    embed_prefixes=(
        "model.embed_tokens.",
    ),
    back_prefixes=(
        "model.norm.",
        "lm_head.",
    ),
    layer_patterns=compile_patterns(
        r"^(?P<prefix>model\.layers\.)(?P<idx>\d+)(?P<suffix>\..+)$",
    ),
)


ADAPTERS: dict[str, SplitAdapter] = {
    "decoder_standard": DECODER_STANDARD,
    "qwen": DECODER_STANDARD,
    "llama": DECODER_STANDARD,
    "gemma": DECODER_STANDARD,
}


MODEL_TYPE_TO_ADAPTER: dict[str, str] = {
    "qwen2": "qwen",
    "qwen3": "qwen",
    "llama": "llama",
    "gemma": "gemma",
    "gemma2": "gemma",
    "gemma3": "gemma",
}


MODEL_HINT_TO_ADAPTER: dict[str, str] = {
    "qwen": "qwen",
    "llama": "llama",
    "gemma": "gemma",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a decoder-only HF checkpoint into front/back safetensors.",
    )
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--local_repo", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--cut", type=int, required=True)
    parser.add_argument("--out_dir", type=str, default="split_out")
    parser.add_argument(
        "--adapter",
        type=str,
        default="auto",
        choices=["auto", *sorted(ADAPTERS.keys())],
    )
    parser.add_argument(
        "--keep_other",
        type=str,
        default="drop",
        choices=["drop", "front", "back", "both"],
        help="How to route unmatched keys.",
    )
    return parser.parse_args()


def resolve_repo_dir(args: argparse.Namespace) -> Path:
    if args.local_repo is not None:
        repo_dir = Path(args.local_repo).expanduser().resolve()
        if not repo_dir.is_dir():
            raise FileNotFoundError(f"local_repo is not a directory: {repo_dir}")
        return repo_dir

    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "huggingface_hub is required when --local_repo is not provided."
        ) from exc

    kwargs: dict[str, Any] = {"repo_id": args.model_id}
    if args.revision:
        kwargs["revision"] = args.revision
    if args.cache_dir:
        kwargs["cache_dir"] = str(Path(args.cache_dir).expanduser().resolve())
    local_path = snapshot_download(**kwargs)
    return Path(local_path).resolve()


def load_config(repo_dir: Path) -> dict[str, Any]:
    cfg_path = repo_dir / "config.json"
    if not cfg_path.exists():
        return {}
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def pick_adapter(
    adapter_name: str,
    config: dict[str, Any],
    model_id: str,
    repo_dir: Path,
) -> SplitAdapter:
    if adapter_name != "auto":
        return ADAPTERS[adapter_name]

    model_type = str(config.get("model_type", "")).lower()
    if model_type in MODEL_TYPE_TO_ADAPTER:
        return ADAPTERS[MODEL_TYPE_TO_ADAPTER[model_type]]

    haystack = f"{model_id} {repo_dir.name}".lower()
    for hint, mapped in MODEL_HINT_TO_ADAPTER.items():
        if hint in haystack:
            return ADAPTERS[mapped]

    return ADAPTERS["decoder_standard"]


def list_shards(repo_dir: Path) -> list[Path]:
    shards = sorted(repo_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors found under {repo_dir}")
    return shards


def match_layer_key(key: str, adapter: SplitAdapter) -> re.Match[str] | None:
    for pattern in adapter.layer_patterns:
        m = pattern.match(key)
        if m is not None:
            return m
    return None


def route_other_key(
    key: str,
    tensor: Any,
    keep_other: str,
    front_sd: dict[str, Any],
    back_sd: dict[str, Any],
    other_keys: list[str],
) -> None:
    if keep_other == "front":
        if key in front_sd:
            raise ValueError(f"duplicate key in front checkpoint: {key}")
        front_sd[key] = tensor
    elif keep_other == "back":
        if key in back_sd:
            raise ValueError(f"duplicate key in back checkpoint: {key}")
        back_sd[key] = tensor
    elif keep_other == "both":
        if key in front_sd:
            raise ValueError(f"duplicate key in front checkpoint: {key}")
        if key in back_sd:
            raise ValueError(f"duplicate key in back checkpoint: {key}")
        front_sd[key] = tensor
        back_sd[key] = tensor
    else:
        other_keys.append(key)


def ensure_back_lm_head_weight(
    front_sd: dict[str, Any],
    back_sd: dict[str, Any],
) -> bool:
    head_key = "lm_head.weight"
    embed_key = "model.embed_tokens.weight"
    if head_key in back_sd:
        return False
    if embed_key not in front_sd:
        raise KeyError(
            "missing lm_head.weight in source checkpoint and cannot recover because "
            "model.embed_tokens.weight was not routed to front split"
        )
    back_sd[head_key] = front_sd[embed_key].clone()
    return True


def split_checkpoint(
    shards: list[Path],
    adapter: SplitAdapter,
    cut: int,
    keep_other: str,
) -> tuple[dict[str, Any], dict[str, Any], list[str], SplitStats]:
    try:
        from safetensors import safe_open
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("safetensors is required to read checkpoint shards.") from exc

    front_sd: dict[str, Any] = {}
    back_sd: dict[str, Any] = {}
    other_keys: list[str] = []

    scanned = 0
    max_layer_idx = -1

    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as reader:
            for key in reader.keys():
                scanned += 1

                if key.startswith(adapter.embed_prefixes):
                    if key in front_sd:
                        raise ValueError(f"duplicate key in front checkpoint: {key}")
                    front_sd[key] = reader.get_tensor(key)
                    continue

                if key.startswith(adapter.back_prefixes):
                    if key in back_sd:
                        raise ValueError(f"duplicate key in back checkpoint: {key}")
                    back_sd[key] = reader.get_tensor(key)
                    continue

                matched = match_layer_key(key, adapter)
                if matched is not None:
                    layer_idx = int(matched.group("idx"))
                    max_layer_idx = max(max_layer_idx, layer_idx)
                    tensor = reader.get_tensor(key)

                    if layer_idx < cut:
                        if key in front_sd:
                            raise ValueError(f"duplicate key in front checkpoint: {key}")
                        front_sd[key] = tensor
                    else:
                        new_idx = layer_idx - cut
                        remapped_key = (
                            f"{matched.group('prefix')}{new_idx}{matched.group('suffix')}"
                        )
                        if remapped_key in back_sd:
                            raise ValueError(
                                f"duplicate remapped key in back checkpoint: {remapped_key}"
                            )
                        back_sd[remapped_key] = tensor
                    continue

                tensor = reader.get_tensor(key) if keep_other != "drop" else None
                route_other_key(key, tensor, keep_other, front_sd, back_sd, other_keys)

    injected = ensure_back_lm_head_weight(front_sd, back_sd)
    if injected:
        print(
            "[warn] lm_head.weight missing in source shards; copied from "
            "model.embed_tokens.weight into back checkpoint"
        )

    stats = SplitStats(
        scanned_keys=scanned,
        front_keys=len(front_sd),
        back_keys=len(back_sd),
        other_keys=len(other_keys),
        max_layer_index=max_layer_idx,
    )
    return front_sd, back_sd, other_keys, stats


def infer_total_layers(config: dict[str, Any], max_layer_index: int) -> int:
    cfg_layers = int(config.get("num_hidden_layers", 0) or 0)
    inferred_layers = max_layer_index + 1 if max_layer_index >= 0 else 0
    return cfg_layers if cfg_layers > 0 else inferred_layers


def validate_cut(cut: int, total_layers: int) -> None:
    if cut < 0:
        raise ValueError(f"cut must be >= 0, got {cut}")
    if total_layers > 0 and cut > total_layers:
        raise ValueError(
            f"cut ({cut}) exceeds total layers ({total_layers}); check --cut or adapter.",
        )


def build_front_config(config: dict[str, Any], cut: int) -> dict[str, Any]:
    front_cfg = dict(config)
    front_cfg["num_hidden_layers"] = int(cut)
    return front_cfg


def build_back_config(
    config: dict[str, Any],
    cut: int,
    total_layers: int,
) -> dict[str, Any]:
    back_cfg = dict(config)
    back_cfg["num_hidden_layers"] = int(max(0, total_layers - cut))
    back_cfg["tie_word_embeddings"] = False
    return back_cfg


def write_split_artifacts(
    out_dir: Path,
    front_sd: dict[str, Any],
    back_sd: dict[str, Any],
    front_cfg: dict[str, Any],
    back_cfg: dict[str, Any],
    plan: dict[str, Any],
    other_keys: list[str],
) -> None:
    try:
        from safetensors.torch import save_file
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("safetensors is required to write split weights.") from exc

    front_dir = out_dir / "front"
    back_dir = out_dir / "back"
    front_dir.mkdir(parents=True, exist_ok=True)
    back_dir.mkdir(parents=True, exist_ok=True)

    save_file(front_sd, str(front_dir / "model.safetensors"))
    save_file(back_sd, str(back_dir / "model.safetensors"))
    (front_dir / "config.json").write_text(
        json.dumps(front_cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (back_dir / "config.json").write_text(
        json.dumps(back_cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "split_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if other_keys:
        (out_dir / "other_keys.txt").write_text(
            "\n".join(other_keys) + "\n",
            encoding="utf-8",
        )


def main() -> int:
    args = parse_args()
    repo_dir = resolve_repo_dir(args)
    out_dir = Path(args.out_dir).expanduser().resolve()
    config = load_config(repo_dir)
    adapter = pick_adapter(args.adapter, config, args.model_id, repo_dir)
    shards = list_shards(repo_dir)

    print(f"[info] repo_dir={repo_dir}")
    print(f"[info] adapter={adapter.name}")
    print(f"[info] cut={args.cut}")
    print(f"[info] shards={len(shards)}")

    front_sd, back_sd, other_keys, stats = split_checkpoint(
        shards=shards,
        adapter=adapter,
        cut=args.cut,
        keep_other=args.keep_other,
    )

    total_layers = infer_total_layers(config, stats.max_layer_index)
    if total_layers <= 0:
        raise ValueError(
            "unable to infer total layers; check config.json or choose a different --adapter."
        )
    validate_cut(args.cut, total_layers)

    front_cfg = build_front_config(config, args.cut)
    back_cfg = build_back_config(config, args.cut, total_layers)

    plan = {
        "source": {
            "model_id": args.model_id,
            "local_repo": str(repo_dir),
            "revision": args.revision,
        },
        "split": {
            "cut": args.cut,
            "adapter": adapter.name,
            "keep_other": args.keep_other,
        },
        "layers": {
            "from_config": int(config.get("num_hidden_layers", 0) or 0),
            "inferred_from_keys": int(stats.max_layer_index + 1)
            if stats.max_layer_index >= 0
            else 0,
            "used_total": int(total_layers),
            "front": int(args.cut),
            "back": int(max(0, total_layers - args.cut)),
        },
        "stats": {
            "scanned_keys": int(stats.scanned_keys),
            "front_keys": int(stats.front_keys),
            "back_keys": int(stats.back_keys),
            "other_keys": int(stats.other_keys),
        },
        "artifacts": {
            "front_weights": "front/model.safetensors",
            "back_weights": "back/model.safetensors",
            "front_config": "front/config.json",
            "back_config": "back/config.json",
        },
    }
    if other_keys:
        plan["other_keys_preview"] = other_keys[:100]

    write_split_artifacts(
        out_dir=out_dir,
        front_sd=front_sd,
        back_sd=back_sd,
        front_cfg=front_cfg,
        back_cfg=back_cfg,
        plan=plan,
        other_keys=other_keys,
    )

    print(
        "[ok] scanned={s} front={f} back={b} other={o}".format(
            s=stats.scanned_keys,
            f=stats.front_keys,
            b=stats.back_keys,
            o=stats.other_keys,
        )
    )
    print(f"[ok] wrote {out_dir / 'front' / 'model.safetensors'}")
    print(f"[ok] wrote {out_dir / 'back' / 'model.safetensors'}")
    print(f"[ok] wrote {out_dir / 'split_plan.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
