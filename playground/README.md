# Playground

Interactive scripts for quick SplitLLM experiments.

## Activation Gradio

Launch a Gradio UI that renders split-front activations for typed text.

```bash
source .venv/bin/activate
python -m playground.activation_gradio \
  --front_dir ./split_out/front \
  --tokenizer_id /path/to/local/tokenizer \
  --local_files_only
```

The app scans `custom/*.py` and builds one dropdown option per valid custom
`ActivationCodec`. The default output is Plotly heatmap components with hover
values for token index, hidden dim, grid position, and exact activation value.

The token plot uses two rows:

- top row: raw front activations
- bottom row: activations after codec encode/decode

By default the token plot contains five tokens, matching the `1..5 / 1'..5'`
comparison layout.
Use `Token start index` to inspect later token groups.

The summary plot includes two average views:

- average activation
- average activation after zeroing the largest-magnitude top-k% hidden dims per token

Set the top-k percentage in the UI, or pass `--topk_percent 1` at launch time.

Set the default codec from the command line with `--codec`:

```bash
TOPK4BIT_DROP_RATIO=0.1 python -m playground.activation_gradio \
  --front_dir ./split_out/front \
  --tokenizer_id /path/to/local/tokenizer \
  --codec custom.topk_4bit_codec \
  --local_files_only
```

Turn on `Save PNG gallery` in the UI, or pass `--save_images`, if static PNG
files are also needed.

## Benchmarks

The playground can run the existing benchmark entrypoints with the selected
codec:

- `bench.ppl`
- `bench.mmlu`

Launch with both split checkpoint paths:

```bash
source .venv/bin/activate
python -m playground.activation_gradio \
  --front_dir ./split_out/front \
  --back_dir ./split_out/back \
  --tokenizer_id /path/to/local/tokenizer \
  --local_files_only
```

Benchmark buttons stream stdout into the UI, update a Gradio progress bar from
`[progress]` lines, and save result JSON files under `./playground_out/bench`.
