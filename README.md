# SplitLLM

Research codebase for split LLM edge-cloud inference experiments.

## Purpose

- split checkpoints into `front` and `back`
- run collaborative inference in local or remote mode
- evaluate quality and latency with reproducible benchmarks
- test activation codec variants

## Files

- `split/`: checkpoint splitting tools (`split/README.md`)
- `model/`: runtime and model API (`model/README.md`)
- `bench/`: benchmark scripts and JSON schema (`bench/README.md`)
- `viz/`: visualization scripts (`viz/README.md`)
- `custom/`: custom experimental codecs/modules
- `tests/`: reference experiment scripts
- `docs/`: design notes

## Usage

```bash
python -m split.ckpt --help
python -m bench.ppl --help
python -m bench.latency --help
```
