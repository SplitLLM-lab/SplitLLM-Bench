# Runtime

Runtime modules and CLI entrypoints for split LLM inference.

## Purpose

- provide local split runtime
- provide remote split runtime (edge client + cloud server)
- provide CLI entrypoints for server/client experiments

## Files

- `runtime/__init__.py`: runtime export facade
- `runtime/common.py`: shared runtime helpers (model load, sampling, decode utils)
- `runtime/local.py`: local split runtime implementation
- `runtime/remote.py`: remote client/runtime implementation
- `runtime/server.py`: remote back server + CLI (`python3 -m runtime.server`)
- `runtime/client.py`: remote client generation CLI (`python3 -m runtime.client`)

## Usage

Remote back server (Python API):

```python
from runtime import RemoteBackServer

server = RemoteBackServer(
    back_dir="./split_out/back",
    device="auto",
    dtype="auto",
    back_quant="none",
)
server.run(host="0.0.0.0", port=8000)
```

Start remote back server (CLI):

```bash
python3 -m runtime.server \
  --back_dir ./split_out/back \
  --host 0.0.0.0 \
  --port 8000 \
  --back_quant none \
  --codec default
```

Remote edge inference (model API, with runtime server running):

```python
from model import SplitLLMModel

m = SplitLLMModel.from_pretrained(
    mode="remote_back",
    tokenizer_id="Qwen/Qwen3-1.7B",
    front_dir="./split_out/front",
    server_url="http://127.0.0.1:8000",
    device="auto",
    dtype="auto",
    front_quant="none",
)

res = m.generate(prompt="Briefly compare TTFT and token RTT.", max_new_tokens=64)
print(res.text)
```

Self-speculative generation uses the same API after loading the front with
draft-head weights:

```python
m = SplitLLMModel.from_pretrained(
    mode="remote_back",
    tokenizer_id="facebook/layerskip-llama3-8B",
    front_dir="./split_out_layerskip/front",
    server_url="http://127.0.0.1:8000",
    enable_self_speculative=True,
)

res = m.generate(
    prompt="Explain LayerSkip briefly.",
    max_new_tokens=64,
    assistant_early_exit=4,
    num_speculations=3,
)
```

Run remote client generation (CLI):

```bash
python3 -m runtime.client \
  --front_dir ./split_out/front \
  --tokenizer_id Qwen/Qwen3-1.7B \
  --server_url http://127.0.0.1:8000 \
  --front_quant none \
  --prompt "Briefly compare TTFT and token RTT." \
  --max_new_tokens 64 \
  --codec default
```

## Notes

- `runtime.client` internally uses `SplitLLMModel(mode="remote_back")`.
- Builtin codecs are `default` and `identity_fp32`; custom module specs are supported.
- Quant modes are `none` and `bnb_8bit` (aliases: `hf_int8`, `int8`, `bnb`).
- `bnb_8bit` requires CUDA + `bitsandbytes`.
- Local split runtime class is `runtime.local.LocalSplitRuntime` (no dedicated CLI entrypoint).
- Self-speculative mode is greedy-only in this prototype and requires a front checkpoint split with `--front_draft_head`.
