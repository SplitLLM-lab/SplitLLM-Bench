from __future__ import annotations

import time
from typing import Any, Optional

import torch
from transformers.cache_utils import DynamicCache

from model.codec import ActivationCodec, CodecContext, ensure_codec
from .common import (
    RuntimeGenerateResult,
    SamplingConfig,
    build_logits_processors,
    build_logits_warpers,
    load_back_model,
    load_front_model,
    pick_device_and_dtype,
    resolve_dir,
    select_next_token,
)


class LocalSplitRuntime:
    def __init__(
        self,
        *,
        front_dir: str,
        back_dir: str,
        device: str | None = "auto",
        dtype: str = "auto",
        revision: str | None = None,
        codec: ActivationCodec | None = None,
    ):
        self.device, self.dtype = pick_device_and_dtype(device, dtype)
        self.codec = ensure_codec(codec)
        front_local = resolve_dir(front_dir, revision=revision)
        back_local = resolve_dir(back_dir, revision=revision)

        print(
            f"[info] loading local_split front={front_local} back={back_local} "
            f"device={self.device} dtype={self.dtype} codec={self.codec.name}"
        )
        self.front = load_front_model(front_local, self.device, self.dtype)
        self.back = load_back_model(back_local, self.device, self.dtype)
        print("[ok] local_split runtime ready")

    @torch.no_grad()
    def generate_from_ids(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None,
        stop_token_ids: set[int],
        sampling: SamplingConfig,
        codec_extras: Optional[dict[str, Any]] = None,
    ) -> RuntimeGenerateResult:
        sampling.validate()
        if max_new_tokens <= 0:
            return RuntimeGenerateResult(
                prompt_token_ids=input_ids[0].tolist(),
                generated_token_ids=[],
                finish_reason="length",
                ttft_ms=0.0,
                total_ms=0.0,
            )

        t_total0 = time.perf_counter()
        prompt_len = int(input_ids.shape[1])
        processors = build_logits_processors(
            prompt_length=prompt_len,
            eos_token_id=eos_token_id,
            cfg=sampling,
        )
        warpers = build_logits_warpers(sampling)

        front_cache = DynamicCache()
        back_cache = DynamicCache()

        cache_position = torch.arange(0, prompt_len, device=self.device)
        codec_extras = dict(codec_extras or {})

        t_prefill0 = time.perf_counter()

        mid = self.front.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=front_cache,
            use_cache=True,
            return_dict=True,
        )
        prefill_ctx = CodecContext(
            phase="prefill",
            side="local",
            session_id=None,
            seq_len=prompt_len,
            step=0,
            extras=codec_extras,
        )
        encoded_prefill = self.codec.encode(mid.last_hidden_state, context=prefill_ctx)
        prefill_hidden = self.codec.decode(
            encoded_prefill,
            context=prefill_ctx,
            device=self.device,
            dtype=self.dtype,
        )
        out = self.back.model(
            inputs_embeds=prefill_hidden,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=back_cache,
            use_cache=True,
            return_dict=True,
        )
        logits = self.back.lm_head(out.last_hidden_state)[:, -1, :]

        all_ids = input_ids
        next_token = select_next_token(
            input_ids=all_ids,
            logits=logits,
            processors=processors,
            warpers=warpers,
            do_sample=sampling.do_sample,
        )

        t_prefill1 = time.perf_counter()
        generated_ids = [int(next_token.item())]
        per_token_rtt_ms: list[float] = []
        finish_reason = "length"

        all_ids = torch.cat([all_ids, next_token], dim=1)
        if generated_ids[-1] in stop_token_ids:
            finish_reason = "stop"

        for _ in range(max(0, max_new_tokens - 1)):
            if finish_reason == "stop":
                break

            t_step0 = time.perf_counter()
            cache_position = cache_position[-1:] + 1

            hid = self.front.model(
                input_ids=next_token,
                cache_position=cache_position,
                past_key_values=front_cache,
                use_cache=True,
                return_dict=True,
            ).last_hidden_state
            decode_ctx = CodecContext(
                phase="decode",
                side="local",
                session_id=None,
                seq_len=int(all_ids.shape[1]),
                step=len(generated_ids),
                extras=codec_extras,
            )
            encoded_decode = self.codec.encode(hid, context=decode_ctx)
            decode_hidden = self.codec.decode(
                encoded_decode,
                context=decode_ctx,
                device=self.device,
                dtype=self.dtype,
            )

            out = self.back.model(
                inputs_embeds=decode_hidden,
                cache_position=cache_position,
                past_key_values=back_cache,
                use_cache=True,
                return_dict=True,
            )
            logits = self.back.lm_head(out.last_hidden_state)[:, -1, :]

            next_token = select_next_token(
                input_ids=all_ids,
                logits=logits,
                processors=processors,
                warpers=warpers,
                do_sample=sampling.do_sample,
            )

            t_step1 = time.perf_counter()
            per_token_rtt_ms.append((t_step1 - t_step0) * 1000.0)

            generated_ids.append(int(next_token.item()))
            all_ids = torch.cat([all_ids, next_token], dim=1)
            if generated_ids[-1] in stop_token_ids:
                finish_reason = "stop"

        t_total1 = time.perf_counter()
        return RuntimeGenerateResult(
            prompt_token_ids=input_ids[0].tolist(),
            generated_token_ids=generated_ids,
            finish_reason=finish_reason,
            ttft_ms=(t_prefill1 - t_prefill0) * 1000.0,
            per_token_rtt_ms=per_token_rtt_ms,
            total_ms=(t_total1 - t_total0) * 1000.0,
        )


__all__ = ["LocalSplitRuntime"]
