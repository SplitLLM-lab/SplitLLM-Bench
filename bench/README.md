# Benchmarks

This folder contains benchmark scripts for split inference experiments.

Current benchmark:
- `ppl.py`: perplexity (PPL) benchmark for split front/back checkpoints.
- `utils.py`: shared benchmark helpers (codec loading, dataset loading, math/utils)

Codec argument:
- Builtin: `--codec default`, `--codec identity_fp32`
- Custom:
  - `--codec custom.my_codec` (module path)
  - `--codec custom.my_codec:build_codec` (explicit attribute)
  - `--codec custom.registry.my_codec` (module.attribute)

## Design Notes

To keep experiments comparable across codec/bench/mode, each benchmark should:
- use explicit CLI args for model, dataset, runtime, and codec settings
- print `[info]`, `[progress]`, `[result]`, `[ok]`, `[error]` logs
- optionally write one JSON result file with stable top-level fields:
  - `benchmark`
  - `model`
  - `codec`
  - `dataset`
  - `runtime`
  - `eval`

`ppl.py` already follows this structure.

## PPL Benchmark

Example:

```bash
python3 -m bench.ppl \
  --front_dir ./split_out/front \
  --back_dir ./split_out/back \
  --tokenizer_id Qwen/Qwen3-1.7B \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --split test \
  --samples 500 \
  --max_length 256 \
  --batch_size 8 \
  --device auto \
  --dtype auto \
  --codec default \
  --out_json bench/results/ppl_local_default.json
```

Codec experiment example:

```bash
python3 -m bench.ppl \
  --codec identity_fp32 \
  --out_json bench/results/ppl_local_identity.json
```

Custom codec example:

```bash
python3 -m bench.ppl \
  --codec custom.my_codec \
  --out_json bench/results/ppl_custom.json
```

For `--codec custom.my_codec` (module mode), benchmark resolves codec by:
1. `build_codec()` in module
2. `codec` variable
3. `CODEC` variable
4. exactly one `ActivationCodec` subclass in module
