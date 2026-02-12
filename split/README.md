# Split

Checkpoint splitting tools for edge-cloud inference experiments.

## Purpose

- split model checkpoints into `front` and `back`
- remap back-layer indices after the cut
- save reproducible split artifacts and split plans

## Files

- `split/ckpt.py`: split CLI and model-specific adapter rules

## Usage

Use local checkpoint:

```bash
python -m split.ckpt \
  --local_repo /path/to/model_repo \
  --cut 12 \
  --out_dir split_out
```

Download from HuggingFace:

```bash
python -m split.ckpt \
  --model_id Qwen/Qwen3-1.7B \
  --cut 12 \
  --out_dir split_out
```

## Notes

- Current adapters target Qwen, Llama3/Llama3.2, and Gemma.
- Output files: `split_out/front/model.safetensors`, `split_out/back/model.safetensors`, `split_out/front/config.json`, `split_out/back/config.json`, `split_out/split_plan.json`.
- To add a new model family, extend `SplitAdapter` rules in `split/ckpt.py` and set `embed_prefixes`, `back_prefixes`, and `layer_patterns` named groups (`prefix`, `idx`, `suffix`).
