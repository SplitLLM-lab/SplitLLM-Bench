# Split Tools

Minimal split tool for edge-cloud collaborative inference.

Current target models:
- Qwen
- Llama3 / Llama3.2
- Gemma

## Design

Three layers only:
1. Read shards / list keys / load tensors
2. Route each key to `front` or `back`, and remap back layer index
3. Write `front/back` safetensors + configs + `split_plan.json`

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

Outputs:
- `split_out/front/model.safetensors`
- `split_out/back/model.safetensors`
- `split_out/front/config.json`
- `split_out/back/config.json`
- `split_out/split_plan.json`

## Add New Model (Minimal Change)

If a new model uses different key names, update adapter rules in `split/ckpt.py`:
- add a new `SplitAdapter`
- set `embed_prefixes`
- set `back_prefixes`
- add `layer_patterns` with named groups:
  - `prefix`
  - `idx`
  - `suffix`

Usually you only need to add a small regex and one mapping entry.
