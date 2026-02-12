# Model

Runtime and model API for split LLM inference.

## Purpose

- provide local and remote split runtime paths
- expose a HuggingFace-like load/generate interface
- support pluggable activation codecs

## Files

- `model/runtime.py`: local runtime, remote edge runtime, and remote back server
- `model/model.py`: `SplitLLMModel.from_pretrained(...)` and `generate(...)`
- `model/codec.py`: `ActivationCodec` interface and default codec

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

Remote back server:

```python
from model import RemoteBackServer

server = RemoteBackServer(
    back_dir="./split_out/back",
    device="auto",
    dtype="auto",
)
server.run(host="0.0.0.0", port=8000)
```

Remote edge inference:

```python
from model import SplitLLMModel

m = SplitLLMModel.from_pretrained(
    mode="remote_back",
    tokenizer_id="Qwen/Qwen3-1.7B",
    front_dir="./split_out/front",
    server_url="http://127.0.0.1:8000",
    device="auto",
    dtype="auto",
)

res = m.generate(prompt="Briefly compare TTFT and token RTT.", max_new_tokens=64)
print(res.text)
```

## Notes

- Modes: `local_split` and `remote_back`.
- Default codec is `DefaultCodec` (local split avoids transport-style conversion by default).
- You can pass custom codecs via `codec=...` (`ActivationCodec` or `FunctionalActivationCodec`).
- In `remote_back` mode, edge and server must use compatible codec logic.
- `generate(...)` returns `GenerateResult` with text, token IDs, finish reason, and timing fields.
