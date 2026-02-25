# SplitLLM

Research codebase for split LLM edge-cloud inference experiments.

## Purpose

- split checkpoints into `front` and `back`
- run collaborative inference in local or remote mode
- evaluate quality and latency with reproducible benchmarks
- test activation codec variants

## Files

- `split/`: checkpoint splitting tools (`split/README.md`)
- `model/`: model API and activation codec modules (`model/README.md`)
- `runtime/`: local/remote runtime components and server/client CLI (`runtime/README.md`)
- `bench/`: benchmark scripts and JSON schema (`bench/README.md`)
- `viz/`: visualization scripts (`viz/README.md`)
- `custom/`: custom experimental codecs/modules
- `tests/`: reference experiment scripts
- `docs/`: design notes

## Usage

### Setup with uv

```bash
# 1) create virtual environment
uv venv .venv

# 2) activate venv
source .venv/bin/activate

# 3) install dependencies
uv pip install -r requirement.txt
```

### Run with python (activated venv)

```bash
python -m split.ckpt --help
python -m runtime.server --help
python -m runtime.client --help
python -m bench.ppl --help
python -m bench.latency --help
```
