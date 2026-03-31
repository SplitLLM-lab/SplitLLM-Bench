# SplitLLM

Research codebase for split LLM edge-cloud inference experiments.

## What It Does

- Split checkpoints into `front` and `back`.
- Run collaborative inference in `local_split` or `remote_back` mode.
- Benchmark quality and latency with reproducible scripts.
- Evaluate activation codec variants (default is no transport-style conversion).

## Typical Workflow

1. Split a checkpoint with `split/ckpt.py`.
2. Run local or remote split inference via `model/` + `runtime/`.
3. Run benchmarks in `bench/`.
4. Use `tests/` as reference experiment scripts and `viz/` for plots.

## Repo Map

- `split/`: checkpoint split tools ([split/README.md](split/README.md))
- `model/`: model API + activation codec modules ([model/README.md](model/README.md))
- `runtime/`: local/remote runtime + server/client CLI ([runtime/README.md](runtime/README.md))
- `bench/`: benchmark entry scripts and helpers ([bench/README.md](bench/README.md))
- `viz/`: activation visualization scripts ([viz/README.md](viz/README.md))
- `custom/`: custom experimental codec/module examples
- `tests/`: reference/legacy experiment scripts
- `docs/`: design notes

## Setup

```bash
uv venv .venv
# Linux/macOS
source .venv/bin/activate
# PowerShell
# .\\.venv\\Scripts\\Activate.ps1
uv pip install -r requirement.txt
```

## Commands (from repo root)

```bash
python -m split.ckpt --help
python -m runtime.server --help
python -m runtime.client --help
python -m bench.ppl --help
python -m bench.latency --help
python -m bench.mmlu --help
python -m bench.generate_jsonl --help
```
