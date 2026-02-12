# Model Runtime

Minimal runtime + model wrapper for split LLM inference.

Current modes:
- `local_split`: front/back both loaded on one machine
- `remote_back`: front on edge, back on server over HTTP

## Design

Three layers only:
1. Runtime primitives in `model/runtime.py` (local runtime, remote edge runtime, remote back server)
2. HuggingFace-like interface in `model/model.py` (`SplitLLMModel.from_pretrained`, `generate`)
3. Generation loop with HF logits processors/warpers (rules + sampling)

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

res = m.generate(
    prompt="請用兩句話解釋 split inference 的概念。",
    max_new_tokens=64,
)
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

res = m.generate(
    prompt="請簡述 TTFT 與每 token RTT 的差異。",
    max_new_tokens=64,
)
print(res.text)
```

## Outputs

`generate(...)` returns `GenerateResult`:
- `text`
- `generated_token_ids`
- `prompt_token_ids`
- `full_token_ids`
- `finish_reason`
- `timing` (`ttft_ms`, `per_token_rtt_ms`, `server_ms`, `total_ms`)
