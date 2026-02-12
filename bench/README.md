# Bench

Benchmark scripts for split inference quality and latency.

## Purpose

- measure perplexity (PPL) and generation latency
- compare codec and runtime settings with reproducible CLI runs
- output stable JSON results for later analysis

## Files

- `bench/ppl.py`: PPL benchmark (local split runtime)
- `bench/latency.py`: latency benchmark (local split or remote back)
- `bench/utils.py`: shared helpers (codec loading, dataset loading, math)

## Usage

PPL (local split):

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
  --codec default
```

Latency (local split):

```bash
python3 -m bench.latency \
  --runtime_mode local_split \
  --front_dir ./split_out/front \
  --back_dir ./split_out/back \
  --tokenizer_id Qwen/Qwen3-1.7B \
  --samples 100 \
  --max_prompt_length 256 \
  --max_new_tokens 64 \
  --codec default
```

Latency (remote back):

```bash
python3 -m bench.latency \
  --runtime_mode remote_back \
  --front_dir ./split_out/front \
  --server_url http://127.0.0.1:8000 \
  --samples 100 \
  --max_prompt_length 256 \
  --max_new_tokens 64 \
  --codec default
```

## Notes

- `--codec` supports builtin codecs (`default`, `identity_fp32`).
- `--codec` custom module forms: `custom.my_codec`, `custom.my_codec:build_codec`, `custom.registry.my_codec`.
- `ppl.py` is local-only; `latency.py` supports `local_split` and `remote_back`.
- Result JSON keeps stable top-level fields: `benchmark`, `model`, `codec`, `dataset`, `runtime`, `eval`.
- Main latency outputs include TTFT, system latency, decode-step latency, codec latency, and transfer bytes.
