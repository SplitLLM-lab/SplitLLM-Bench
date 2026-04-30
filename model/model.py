from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
from transformers import AutoTokenizer

from .codec import ActivationCodec
from runtime import LocalSplitRuntime, RemoteSplitRuntime, RuntimeGenerateResult, SamplingConfig


@dataclass
class GenerateResult:
    text: str
    generated_token_ids: list[int]
    prompt_token_ids: list[int]
    full_token_ids: list[int]
    finish_reason: str
    timing: dict[str, Any]


def build_chat_prompt(
    tokenizer,
    prompt: str,
    *,
    use_chat_template: bool,
    add_generation_prompt: bool,
) -> str:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            pass
    return prompt


def encode_bad_words(tokenizer, bad_words: Optional[Sequence[str]]) -> Optional[list[list[int]]]:
    if not bad_words:
        return None

    out: list[list[int]] = []
    for w in bad_words:
        ids = tokenizer(w, add_special_tokens=False).input_ids
        if ids:
            out.append(ids)
    return out or None


class SplitLLMModel:
    def __init__(
        self,
        *,
        mode: str,
        tokenizer,
        runtime,
    ):
        self.mode = mode
        self.tokenizer = tokenizer
        self.runtime = runtime

    @classmethod
    def from_pretrained(
        cls,
        *,
        mode: str,
        tokenizer_id: str,
        front_dir: Optional[str] = None,
        back_dir: Optional[str] = None,
        server_url: Optional[str] = None,
        device: str | None = "auto",
        dtype: str = "auto",
        front_quant: str = "none",
        back_quant: str = "none",
        timeout_sec: float = 120.0,
        revision: str | None = None,
        codec: ActivationCodec | None = None,
    ) -> "SplitLLMModel":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        if mode == "local_split":
            if not front_dir or not back_dir:
                raise ValueError("local_split mode requires front_dir and back_dir")
            runtime = LocalSplitRuntime(
                front_dir=front_dir,
                back_dir=back_dir,
                device=device,
                dtype=dtype,
                front_quant=front_quant,
                back_quant=back_quant,
                revision=revision,
                codec=codec,
            )
        elif mode == "remote_back":
            if not front_dir or not server_url:
                raise ValueError("remote_back mode requires front_dir and server_url")
            runtime = RemoteSplitRuntime(
                front_dir=front_dir,
                server_url=server_url,
                device=device,
                dtype=dtype,
                front_quant=front_quant,
                timeout_sec=timeout_sec,
                revision=revision,
                codec=codec,
            )
        else:
            raise ValueError(
                f"unsupported mode: {mode}. use one of ['local_split', 'remote_back']"
            )

        return cls(mode=mode, tokenizer=tokenizer, runtime=runtime)

    def _default_stop_ids(self) -> set[int]:
        stop_ids: set[int] = set()

        if self.tokenizer.eos_token_id is not None:
            stop_ids.add(int(self.tokenizer.eos_token_id))

        vocab = self.tokenizer.get_vocab()
        if "<|im_end|>" in vocab:
            stop_ids.add(int(vocab["<|im_end|>"]))

        return stop_ids

    @torch.no_grad()
    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        min_new_tokens: int = 0,
        no_repeat_ngram_size: int = 0,
        repetition_penalty: float = 1.0,
        bad_words: Optional[Sequence[str]] = None,
        bad_words_ids: Optional[list[list[int]]] = None,
        stop_token_ids: Optional[Sequence[int]] = None,
        codec_extras: Optional[dict[str, Any]] = None,
        use_chat_template: bool = True,
        add_generation_prompt: bool = True,
        skip_special_tokens: bool = True,
    ) -> GenerateResult:
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")

        prompt_text = build_chat_prompt(
            self.tokenizer,
            prompt,
            use_chat_template=use_chat_template,
            add_generation_prompt=add_generation_prompt,
        )

        enc = self.tokenizer(prompt_text, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.runtime.device)
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(
            self.runtime.device
        )

        if bad_words_ids is None:
            bad_words_ids = encode_bad_words(self.tokenizer, bad_words)

        sampling = SamplingConfig(
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_new_tokens=min_new_tokens,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            bad_words_ids=bad_words_ids,
        )

        eos_token_id = (
            int(self.tokenizer.eos_token_id)
            if self.tokenizer.eos_token_id is not None
            else None
        )

        default_stop_ids = self._default_stop_ids()
        if stop_token_ids is None:
            used_stop_ids = default_stop_ids
        else:
            used_stop_ids = {int(x) for x in stop_token_ids}

        runtime_result: RuntimeGenerateResult = self.runtime.generate_from_ids(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            stop_token_ids=used_stop_ids,
            sampling=sampling,
            codec_extras=codec_extras,
        )

        generated = runtime_result.generated_token_ids
        full_ids = runtime_result.prompt_token_ids + generated

        text = self.tokenizer.decode(
            generated,
            skip_special_tokens=skip_special_tokens,
        )

        timing = {
            "ttft_ms": runtime_result.ttft_ms,
            "decode_step_ms": runtime_result.decode_step_ms,
            "per_token_rtt_ms": runtime_result.per_token_rtt_ms,
            "prefill_rtt_ms": runtime_result.prefill_rtt_ms,
            "decode_rtt_ms": runtime_result.decode_rtt_ms,
            "prefill_server_ms": runtime_result.prefill_server_ms,
            "decode_server_ms": runtime_result.decode_server_ms,
            "server_ms": runtime_result.server_ms,
            "total_ms": runtime_result.total_ms,
            "tokens_generated": len(generated),
        }

        return GenerateResult(
            text=text,
            generated_token_ids=generated,
            prompt_token_ids=runtime_result.prompt_token_ids,
            full_token_ids=full_ids,
            finish_reason=runtime_result.finish_reason,
            timing=timing,
        )


__all__ = ["SplitLLMModel", "GenerateResult"]
