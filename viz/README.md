# Viz

Visualization scripts for SplitLLM experiments.

## Purpose

- render token-level activation heatmaps from front model outputs
- save per-token heatmaps and one average-activation figure

## Files

- `viz/activation.py`: activation heatmap renderer

## Usage

```bash
python -m viz.activation \
  --front_dir ./split_out/front \
  --tokenizer_id /path/to/local/tokenizer \
  --local_files_only \
  --text "這是一個繁體中文測試句子。" \
  --out_dir ./viz_out/activation
```

## Notes

- Default is `6` tokens per figure.
- The script auto-selects a CJK-capable font when available.
- In offline environments, use a local tokenizer path with `--local_files_only`.
