# SplitLLM
SplitLLM is a research-oriented codebase for split LLM inference experiments.

## Scope

This repository is mainly for:
- checkpoint splitting (front/back)
- edge-cloud collaborative inference
- codec experiments on transmitted activations
- benchmark and reproducibility workflows

## Repository Guide

- `split/`: checkpoint split tools  
  See details: `split/README.md`
- `model/`: split runtime and model-facing API  
  See details: `model/README.md`
- `bench/`: benchmark scripts and result format  
  See details: `bench/README.md`
- `custom/`: user-defined experimental codecs/modules
- `tests/`: reference experiment scripts
- `docs/`: notes and design documents
