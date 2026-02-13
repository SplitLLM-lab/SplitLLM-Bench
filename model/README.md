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
)

res = m.generate(prompt="Explain split inference in two sentences.", max_new_tokens=64)
print(res.text)
```

## Notes

- Modes: `local_split` and `remote_back`.
- Default codec is `DefaultCodec` (local split avoids transport-style conversion by default).
- You can pass custom codecs via `codec=...` (`ActivationCodec` or `FunctionalActivationCodec`).
- Remote server/client setup and CLI examples are documented in `runtime/README.md`.
- In `remote_back` mode, edge and server must use compatible codec logic.
- `generate(...)` returns `GenerateResult` with text, token IDs, finish reason, and timing fields.
