from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from typing import Any, Optional

import torch
from transformers.cache_utils import DynamicCache

from model.codec import ActivationCodec, CodecContext, ensure_codec
from .common import (
    RuntimeGenerateResult,
    SamplingConfig,
    build_logits_processors,
    build_logits_warpers,
    encoded_payload_size_bytes,
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
        front_quant: str = "none",
        back_quant: str = "none",
        enable_self_speculative: bool = False,
    ):
        self.device, self.dtype = pick_device_and_dtype(device, dtype)
        self.codec = ensure_codec(codec)
        front_local = resolve_dir(front_dir, revision=revision)
        back_local = resolve_dir(back_dir, revision=revision)

        print(
            f"[info] loading local_split front={front_local} back={back_local} "
            f"device={self.device} dtype={self.dtype} codec={self.codec.name} "
            f"front_quant={front_quant} back_quant={back_quant} "
            f"self_speculative={bool(enable_self_speculative)}"
        )
        self.front = load_front_model(
            front_local,
            self.device,
            self.dtype,
            quant_mode=front_quant,
            draft_head=bool(enable_self_speculative),
        )
        self.back = load_back_model(
            back_local,
            self.device,
            self.dtype,
            quant_mode=back_quant,
        )
        print("[ok] local_split runtime ready")

    @contextmanager
    def _front_early_exit(self, early_exit: int):
        old_layers = int(self.front.model.config.num_hidden_layers)
        self.front.model.config.num_hidden_layers = int(early_exit)
        try:
            yield
        finally:
            self.front.model.config.num_hidden_layers = old_layers

    def _ensure_self_speculative_ready(
        self,
        *,
        batch_size: int,
        early_exit: int,
    ) -> None:
        if batch_size != 1:
            raise ValueError("self-speculative decoding only supports batch_size=1")
        if not hasattr(self.front, "_splitllm_draft_norm") or not hasattr(
            self.front,
            "_splitllm_draft_lm_head",
        ):
            raise ValueError(
                "front runtime was not loaded with draft-head weights. "
                "Construct LocalSplitRuntime with enable_self_speculative=True."
            )
        layers = getattr(self.front.model, "layers", None)
        if layers is not None and int(early_exit) > len(layers):
            raise ValueError(
                f"assistant_early_exit ({early_exit}) exceeds front layers ({len(layers)})"
            )

    def _draft_logits(
        self,
        *,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: DynamicCache,
        early_exit: int,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with self._front_early_exit(early_exit):
            out = self.front.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
        hidden = self.front._splitllm_draft_norm(out.last_hidden_state)
        return self.front._splitllm_draft_lm_head(hidden)

    def _codec_roundtrip(
        self,
        *,
        hidden: torch.Tensor,
        context: CodecContext,
    ) -> tuple[torch.Tensor, float, float, int]:
        t0 = time.perf_counter()
        encoded = self.codec.encode(hidden, context=context)
        encode_ms = (time.perf_counter() - t0) * 1000.0
        wire_bytes = encoded_payload_size_bytes(self.codec, encoded)

        t1 = time.perf_counter()
        decoded = self.codec.decode(
            encoded,
            context=context,
            device=self.device,
            dtype=self.dtype,
        )
        decode_ms = (time.perf_counter() - t1) * 1000.0
        return decoded, float(encode_ms), float(decode_ms), int(wire_bytes)

    def _pick_token(
        self,
        *,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
        processors,
        warpers,
        sampling: SamplingConfig,
    ) -> torch.Tensor:
        return select_next_token(
            input_ids=input_ids,
            logits=logits,
            processors=processors,
            warpers=warpers,
            do_sample=sampling.do_sample,
        )

    @torch.no_grad()
    def _generate_self_speculative(
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
        early_exit = int(sampling.assistant_early_exit or 0)
        self._ensure_self_speculative_ready(
            batch_size=int(input_ids.shape[0]),
            early_exit=early_exit,
        )

        t_total0 = time.perf_counter()
        prompt_len = int(input_ids.shape[1])
        num_speculations = max(1, int(sampling.num_speculations))
        processors = build_logits_processors(
            prompt_length=prompt_len,
            eos_token_id=eos_token_id,
            cfg=sampling,
        )
        warpers = build_logits_warpers(sampling)

        front_cache = DynamicCache()
        back_cache = DynamicCache()
        draft_cache = DynamicCache()
        cache_position = torch.arange(0, prompt_len, device=self.device)
        codec_extras = dict(codec_extras or {})
        session_id = str(uuid.uuid4())

        prefill_codec_encode_ms = 0.0
        prefill_codec_decode_ms = 0.0
        prefill_codec_wire_bytes = 0
        decode_codec_encode_ms: list[float] = []
        decode_codec_decode_ms: list[float] = []
        decode_codec_wire_bytes: list[int] = []
        decode_step_ms: list[float] = []

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
            side="local_self_spec",
            session_id=session_id,
            seq_len=prompt_len,
            step=0,
            extras=codec_extras,
        )
        prefill_hidden, enc_ms, dec_ms, wire_bytes = self._codec_roundtrip(
            hidden=mid.last_hidden_state,
            context=prefill_ctx,
        )
        prefill_codec_encode_ms = enc_ms
        prefill_codec_decode_ms = dec_ms
        prefill_codec_wire_bytes = wire_bytes
        out = self.back.model(
            inputs_embeds=prefill_hidden,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=back_cache,
            use_cache=True,
            return_dict=True,
        )
        target_next_logits = self.back.lm_head(out.last_hidden_state)[:, -1, :]

        draft_prefill_logits = self._draft_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=draft_cache,
            early_exit=early_exit,
        )
        draft_next_logits = draft_prefill_logits[:, -1, :]

        all_ids = input_ids
        generated_ids: list[int] = []
        finish_reason = "length"
        ttft_ms = 0.0
        draft_tokens = 0
        accepted_tokens = 0
        verify_rounds = 0
        fallback_tokens = 0

        while len(generated_ids) < max_new_tokens:
            round_t0 = time.perf_counter()
            current_len = int(all_ids.shape[1])
            remaining = int(max_new_tokens - len(generated_ids))
            draft_count = min(num_speculations, remaining)

            candidates: list[int] = []
            draft_context_ids = all_ids
            for draft_idx in range(draft_count):
                next_candidate = self._pick_token(
                    input_ids=draft_context_ids,
                    logits=draft_next_logits,
                    processors=processors,
                    warpers=warpers,
                    sampling=sampling,
                )
                candidates.append(int(next_candidate.item()))
                draft_tokens += 1

                draft_pos = torch.tensor(
                    [current_len + draft_idx],
                    device=self.device,
                    dtype=torch.long,
                )
                draft_logits = self._draft_logits(
                    input_ids=next_candidate,
                    cache_position=draft_pos,
                    past_key_values=draft_cache,
                    early_exit=early_exit,
                )
                draft_next_logits = draft_logits[:, -1, :]
                draft_context_ids = torch.cat([draft_context_ids, next_candidate], dim=1)

            candidate_ids = torch.tensor([candidates], device=self.device, dtype=torch.long)
            first_target = self._pick_token(
                input_ids=all_ids,
                logits=target_next_logits,
                processors=processors,
                warpers=warpers,
                sampling=sampling,
            )

            accepted = 0
            fallback_id: int | None = None
            block_logits: torch.Tensor | None = None

            if int(first_target.item()) != candidates[0]:
                fallback_id = int(first_target.item())
                draft_cache.crop(current_len)
            else:
                target_pos = torch.arange(
                    current_len,
                    current_len + draft_count,
                    device=self.device,
                    dtype=torch.long,
                )
                mid = self.front.model(
                    input_ids=candidate_ids,
                    cache_position=target_pos,
                    past_key_values=front_cache,
                    use_cache=True,
                    return_dict=True,
                )
                decode_ctx = CodecContext(
                    phase="decode",
                    side="local_self_spec",
                    session_id=session_id,
                    seq_len=current_len + draft_count,
                    step=len(generated_ids),
                    extras=codec_extras,
                )
                verify_hidden, enc_ms, dec_ms, wire_bytes = self._codec_roundtrip(
                    hidden=mid.last_hidden_state,
                    context=decode_ctx,
                )
                decode_codec_encode_ms.append(enc_ms)
                decode_codec_decode_ms.append(dec_ms)
                decode_codec_wire_bytes.append(wire_bytes)

                out = self.back.model(
                    inputs_embeds=verify_hidden,
                    cache_position=target_pos,
                    past_key_values=back_cache,
                    use_cache=True,
                    return_dict=True,
                )
                block_logits = self.back.lm_head(out.last_hidden_state)

                accepted = 1
                for idx in range(1, draft_count):
                    context_ids = torch.cat([all_ids, candidate_ids[:, :idx]], dim=1)
                    target_token = self._pick_token(
                        input_ids=context_ids,
                        logits=block_logits[:, idx - 1, :],
                        processors=processors,
                        warpers=warpers,
                        sampling=sampling,
                    )
                    if int(target_token.item()) == candidates[idx]:
                        accepted += 1
                    else:
                        fallback_id = int(target_token.item())
                        break

                if accepted < draft_count:
                    front_cache.crop(current_len + accepted)
                    back_cache.crop(current_len + accepted)
                    draft_cache.crop(current_len + accepted)

            emitted: list[int] = candidates[:accepted]
            stop_in_emitted = next(
                (idx for idx, token_id in enumerate(emitted) if token_id in stop_token_ids),
                None,
            )
            if stop_in_emitted is not None:
                emitted = emitted[: stop_in_emitted + 1]
                crop_len = current_len + len(emitted)
                front_cache.crop(crop_len)
                back_cache.crop(crop_len)
                draft_cache.crop(crop_len)
                fallback_id = None
                finish_reason = "stop"

            if emitted:
                emitted_tensor = torch.tensor(
                    [emitted],
                    device=self.device,
                    dtype=torch.long,
                )
                all_ids = torch.cat([all_ids, emitted_tensor], dim=1)
                generated_ids.extend(emitted)
                accepted_tokens += len(emitted)
                if block_logits is not None and finish_reason != "stop":
                    target_next_logits = block_logits[:, len(emitted) - 1, :]

            if finish_reason == "stop":
                if ttft_ms == 0.0 and generated_ids:
                    ttft_ms = (time.perf_counter() - t_total0) * 1000.0
                decode_step_ms.append((time.perf_counter() - round_t0) * 1000.0)
                verify_rounds += 1
                break

            if fallback_id is not None and len(generated_ids) < max_new_tokens:
                fallback_tokens += 1
                fallback_tensor = torch.tensor(
                    [[fallback_id]],
                    device=self.device,
                    dtype=torch.long,
                )
                all_ids = torch.cat([all_ids, fallback_tensor], dim=1)
                generated_ids.append(fallback_id)

                if fallback_id in stop_token_ids:
                    finish_reason = "stop"
                else:
                    fallback_pos = torch.tensor(
                        [current_len + accepted],
                        device=self.device,
                        dtype=torch.long,
                    )
                    mid = self.front.model(
                        input_ids=fallback_tensor,
                        cache_position=fallback_pos,
                        past_key_values=front_cache,
                        use_cache=True,
                        return_dict=True,
                    )
                    decode_ctx = CodecContext(
                        phase="decode",
                        side="local_self_spec",
                        session_id=session_id,
                        seq_len=int(all_ids.shape[1]),
                        step=len(generated_ids) - 1,
                        extras=codec_extras,
                    )
                    fallback_hidden, enc_ms, dec_ms, wire_bytes = self._codec_roundtrip(
                        hidden=mid.last_hidden_state,
                        context=decode_ctx,
                    )
                    decode_codec_encode_ms.append(enc_ms)
                    decode_codec_decode_ms.append(dec_ms)
                    decode_codec_wire_bytes.append(wire_bytes)

                    out = self.back.model(
                        inputs_embeds=fallback_hidden,
                        cache_position=fallback_pos,
                        past_key_values=back_cache,
                        use_cache=True,
                        return_dict=True,
                    )
                    target_next_logits = self.back.lm_head(out.last_hidden_state)[:, -1, :]

                    draft_logits = self._draft_logits(
                        input_ids=fallback_tensor,
                        cache_position=fallback_pos,
                        past_key_values=draft_cache,
                        early_exit=early_exit,
                    )
                    draft_next_logits = draft_logits[:, -1, :]

            if accepted == draft_count and block_logits is not None:
                target_next_logits = block_logits[:, -1, :]

            if ttft_ms == 0.0 and generated_ids:
                ttft_ms = (time.perf_counter() - t_total0) * 1000.0
            decode_step_ms.append((time.perf_counter() - round_t0) * 1000.0)
            verify_rounds += 1

            if finish_reason == "stop":
                break

        total_ms = (time.perf_counter() - t_total0) * 1000.0
        return RuntimeGenerateResult(
            prompt_token_ids=input_ids[0].detach().cpu().tolist(),
            generated_token_ids=generated_ids,
            finish_reason=finish_reason,
            ttft_ms=ttft_ms,
            decode_step_ms=decode_step_ms,
            per_token_rtt_ms=[],
            prefill_codec_encode_ms=prefill_codec_encode_ms,
            prefill_codec_decode_ms=prefill_codec_decode_ms,
            prefill_codec_wire_bytes=prefill_codec_wire_bytes,
            decode_codec_encode_ms=decode_codec_encode_ms,
            decode_codec_decode_ms=decode_codec_decode_ms,
            decode_codec_wire_bytes=decode_codec_wire_bytes,
            self_speculative=True,
            draft_tokens=int(draft_tokens),
            accepted_tokens=int(accepted_tokens),
            verify_rounds=int(verify_rounds),
            fallback_tokens=int(fallback_tokens),
            total_ms=total_ms,
        )

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
        if sampling.assistant_early_exit is not None:
            return self._generate_self_speculative(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                stop_token_ids=stop_token_ids,
                sampling=sampling,
                codec_extras=codec_extras,
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
            decode_step_ms=per_token_rtt_ms,
            per_token_rtt_ms=per_token_rtt_ms,
            total_ms=(t_total1 - t_total0) * 1000.0,
        )


__all__ = ["LocalSplitRuntime"]
