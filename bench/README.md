# Bench

Benchmark scripts for split inference quality and latency.

## Purpose

- measure perplexity (PPL), MMLU accuracy, and generation latency
- compare codec and runtime settings with reproducible CLI runs
- output stable JSON results for later analysis

## Files

- `bench/ppl.py`: PPL benchmark (local split runtime)
- `bench/mmlu.py`: MMLU multiple-choice accuracy benchmark (local split runtime)
- `bench/latency.py`: latency benchmark (local split or remote back)
- `bench/generate_jsonl.py`: generic JSONL generation CLI (local split runtime)
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

MMLU (local split, single-letter answer over A/B/C/D):

```bash
python3 -m bench.mmlu \
  --front_dir ./split_out/front \
  --back_dir ./split_out/back \
  --tokenizer_id Qwen/Qwen3-1.7B \
  --dataset_name cais/mmlu \
  --dataset_config all \
  --fewshot_split dev \
  --eval_split test \
  --n_shot 5 \
  --max_samples_per_subject 100 \
  --max_length 2048 \
  --decoding greedy \
  --codec default
```

Direct JSONL generation (local split):

```bash
python3 -m bench.generate_jsonl \
  --front_dir ./split_out/front \
  --back_dir ./split_out/back \
  --tokenizer_id Qwen/Qwen3-1.7B \
  --input_jsonl ./data/prompts.jsonl \
  --output_jsonl ./out/generated.jsonl \
  --prompt_key prompt \
  --result_key result \
  --index_key index \
  --decoding top_p \
  --top_p 0.9 \
  --max_new_tokens 64 \
  --codec default
```

## Notes

- `--codec` supports builtin codecs (`default`, `identity_fp32`).
- `--codec` custom module forms: `custom.my_codec`, `custom.my_codec:build_codec`, `custom.registry.my_codec`.
- `ppl.py` and `mmlu.py` are local-only; `latency.py` supports `local_split` and `remote_back`.
- `latency.py --runtime_mode remote_back` requires a running server (`python3 -m runtime.server ...`).
- In `latency.py`, `ttft_ms`, `system_latency_ms`, and `decode_step_latency_ms` are end-to-end in both local and remote modes.
- Remote latency also reports HTTP RTT and server compute time under `eval.remote`.
- Remote codec metrics include client-side codec encode time, server-side codec decode time, and hidden-payload wire bytes. Local codec metrics measure the same split boundary inside one process.
- `mmlu.py` now uses a two-stage flow: prepare prompt JSONL -> generate with `generate_jsonl_with_model` -> score by single-letter parse. It supports `--decoding` (`greedy`, `top_k`, `top_p`) and related generation args.
- `generate_jsonl.py` is the direct CLI wrapper for line-by-line JSONL generation; output rows are written incrementally and reordered by index at the end to match input order.
- `generate_jsonl_with_model` is suitable for generation-first benchmarks (current fit: `mmlu.py`). It is not used in `ppl.py` (forward NLL) or `latency.py` (TTFT/transport latency metrics) because those require different measurement paths.
- Result JSON keeps stable top-level fields: `benchmark`, `model`, `codec`, `dataset`, `runtime`, `eval`.
- Main latency outputs include TTFT, system latency, decode-step latency, codec latency, and transfer bytes.
