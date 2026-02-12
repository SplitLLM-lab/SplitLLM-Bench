import torch
from model import ActivationCodec, CodecContext, EncodedActivation


class MyCodec(ActivationCodec):
    name = "my codec"

    def encode(
        self,
        hidden: torch.Tensor,
        *,
        context: CodecContext,
    ) -> EncodedActivation:
        del context
        return EncodedActivation(data=hidden, meta={})

    def decode(
        self,
        payload: EncodedActivation,
        *,
        context: CodecContext,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        del context
        hidden = payload.data
        return hidden