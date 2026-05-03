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

For self-speculative decoding on LayerSkip-style models, copy the draft head
into the front checkpoint:

```bash
python -m split.ckpt \
  --model_id facebook/layerskip-llama3-8B \
  --cut 12 \
  --front_draft_head \
  --out_dir split_out_layerskip
```

## Notes

- Current adapters target Qwen, Llama3/Llama3.2, and Gemma.
- `--front_draft_head` duplicates `model.norm.*` and `lm_head.*` into the front checkpoint so the edge can produce early-exit draft logits.
- Output files: `split_out/front/model.safetensors`, `split_out/back/model.safetensors`, `split_out/front/config.json`, `split_out/back/config.json`, `split_out/split_plan.json`.
- To add a new model family, extend `SplitAdapter` rules in `split/ckpt.py` and set `embed_prefixes`, `back_prefixes`, and `layer_patterns` named groups (`prefix`, `idx`, `suffix`).
