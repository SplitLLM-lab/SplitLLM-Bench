# viz

Visualization scripts for SplitLLM experiments.

## `activation.py`

Render token-level activation heatmaps from the split `front` model output.

Features:

- Takes one input sentence (`--text`)
- Saves token activation heatmaps
- Uses `6` tokens per figure by default
- Saves one final average-activation image as the last file
- Auto-selects a CJK-capable font for Traditional Chinese when available

Example:

```bash
python -m viz.activation \
  --front_dir ./split_out/front \
  --tokenizer_id /path/to/local/tokenizer \
  --local_files_only \
  --text "這是一個繁體中文測試句子。" \
  --out_dir ./viz_out/activation
```

If the environment is offline and tokenizer files are not cached, use a local tokenizer path with `--local_files_only`.

