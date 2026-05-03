# Model

Model API and activation codec modules for split LLM inference.

## Purpose

- expose a HuggingFace-like load/generate interface (`SplitLLMModel`)
- define activation codec interfaces and builtin codecs
- provide codec loader helpers used by runtime CLI entrypoints

## Files

- `model/__init__.py`: model package exports (includes compatibility re-exports)
- `model/model.py`: `SplitLLMModel.from_pretrained(...)` and `generate(...)`
- `model/codec.py`: `ActivationCodec` interface and builtin codecs
- `model/cli_common.py`: codec loader helpers for CLI entrypoints

## Usage

Local split:

```python
from model import SplitLLMModel

m = SplitLLMModel.from_pretrained(
    mode="local_split",
    tokenizer_id="Qwen/Qwen3-1.7B",
    front_dir="./split_out/front",
    back_dir="./split_out/back",
    device="auto",
    dtype="auto",
    front_quant="none",
    back_quant="none",
)

res = m.generate(prompt="Explain split inference in two sentences.", max_new_tokens=64)
print(res.text)
```

Self-speculative local generation:

```python
m = SplitLLMModel.from_pretrained(
    mode="local_split",
    tokenizer_id="facebook/layerskip-llama3-8B",
    front_dir="./split_out_layerskip/front",
    back_dir="./split_out_layerskip/back",
    enable_self_speculative=True,
)

res = m.generate(
    prompt="Explain LayerSkip briefly.",
    max_new_tokens=64,
    self_speculative=True,
    num_speculations=3,
)
```

## Notes

- Modes: `local_split` and `remote_back`.
- Self-speculative mode is greedy-only, uses all front layers as the draft model by default, and requires a front checkpoint created with `split.ckpt --front_draft_head`.
- `assistant_early_exit` is optional and only overrides the front layer count for ablation runs.
- Default codec is `DefaultCodec` (local split avoids transport-style conversion by default).
- You can pass custom codecs via `codec=...` (`ActivationCodec` or `FunctionalActivationCodec`).
- Remote server/client setup and CLI examples are documented in `runtime/README.md`.
- In `remote_back` mode, edge and server must use compatible codec logic.
- Quant modes are `none` and `bnb_8bit` (aliases: `hf_int8`, `int8`, `bnb`).
- In `remote_back` mode, configure front quant via model/client and back quant via server.
- `generate(...)` returns `GenerateResult` with text, token IDs, finish reason, and timing fields.
