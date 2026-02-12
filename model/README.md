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
4. Pluggable activation codec in `model/codec.py` (`ActivationCodec`)

Default codec:
- `DefaultCodec` (default)
- For local split, it avoids base64/CPU wire conversion by default.
- For remote mode, wire serialization still happens when sending HTTP payload.

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

## Custom Codec

You can inject your own activation codec into both edge and server:

```python
import base64
import numpy as np
import torch
from model import (
    ActivationCodec,
    CodecContext,
    EncodedActivation,
    SplitLLMModel,
    RemoteBackServer,
)

class MyCodec(ActivationCodec):
    name = "my_codec"

    def encode(self, hidden: torch.Tensor, *, context: CodecContext) -> EncodedActivation:
        del context
        x = hidden.detach().to(torch.float32).cpu().numpy()
        q = np.clip(np.round(x * 128.0), -128, 127).astype(np.int8)
        b64 = base64.b64encode(q.tobytes()).decode("ascii")
        return EncodedActivation(
            data=b64,
            meta={"shape": list(q.shape), "dtype": "int8", "scale": 128.0},
        )

    def decode(
        self,
        payload: EncodedActivation,
        *,
        context: CodecContext,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        del context
        shape = payload.meta["shape"]
        raw = base64.b64decode(payload.data.encode("ascii"))
        q = np.frombuffer(raw, dtype=np.int8).reshape(shape)
        x = torch.from_numpy(q.astype(np.float32) / float(payload.meta["scale"]))
        x = x.to(device=device)
        if dtype in (torch.float16, torch.bfloat16):
            x = x.to(dtype)
        return x

codec = MyCodec()

server = RemoteBackServer(
    back_dir="./split_out/back",
    codec=codec,
)

m = SplitLLMModel.from_pretrained(
    mode="remote_back",
    tokenizer_id="Qwen/Qwen3-1.7B",
    front_dir="./split_out/front",
    server_url="http://127.0.0.1:8000",
    codec=codec,
)

res = m.generate(
    prompt="test",
    max_new_tokens=32,
    codec_extras={"quality": "aggressive", "level": 3},
)
```

If you prefer function-style codec, use `FunctionalActivationCodec`.
For `remote_back`, edge and server must use compatible codec logic.
In real deployment, create one codec instance on edge and another on server with same config.

## Outputs

`generate(...)` returns `GenerateResult`:
- `text`
- `generated_token_ids`
- `prompt_token_ids`
- `full_token_ids`
- `finish_reason`
- `timing` (`ttft_ms`, `per_token_rtt_ms`, `server_ms`, `total_ms`)
