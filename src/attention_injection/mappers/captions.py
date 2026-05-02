import torch

from src.attention_injection.mappers.base import InjectionMapper
from src.attention_injection.tokenize import TextTokenizerLike, tokenize_caption_raw


def split_caption_tags(caption: str) -> list[str]:
    parts = [p.strip() for p in caption.split(",")]
    return [p for p in parts if p]


def _tag_pair_block(
    src_tag: str,
    tgt_tag: str,
    tokenizer: TextTokenizerLike,
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_length: int,
) -> torch.Tensor:
    src_t = tokenize_caption_raw(src_tag, tokenizer, max_length=max_length).active_ids()
    tgt_t = tokenize_caption_raw(tgt_tag, tokenizer, max_length=max_length).active_ids()
    ls, lt = len(src_t), len(tgt_t)
    if ls == 0 and lt == 0:
        return torch.zeros(0, 0, device=device, dtype=dtype)
    if ls == 0 or lt == 0:
        raise ValueError(f"empty token span for tag pair {src_tag!r} vs {tgt_tag!r}")
    B = torch.zeros(lt, ls, device=device, dtype=dtype)
    if src_tag.strip() == tgt_tag.strip() and ls == lt:
        for k in range(ls):
            B[k, k] = 1.0
        return B
    for j in range(lt):
        B[j, :] = 1.0 / float(ls)
    return B


class CaptionsReplacementMapper(InjectionMapper):

    def __init__(
        self,
        src_caption: str,
        tgt_caption: str,
        tokenizer: TextTokenizerLike,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        max_length: int = 256,
    ) -> None:
        self._src = src_caption
        self._tgt = tgt_caption
        self._tokenizer = tokenizer
        self._max_length = max_length
        super().__init__(device=device, dtype=dtype)

    def _build_matrix(self) -> torch.Tensor:
        src_tags = split_caption_tags(self._src)
        tgt_tags = split_caption_tags(self._tgt)
        if len(src_tags) != len(tgt_tags):
            raise AssertionError(
                f"caption tag count mismatch: src has {len(src_tags)} tags, tgt has {len(tgt_tags)} "
                f"(comma-separated segments)"
            )
        blocks: list[torch.Tensor] = []
        for st, tt in zip(src_tags, tgt_tags):
            blk = _tag_pair_block(
                st,
                tt,
                self._tokenizer,
                device=self._device,
                dtype=self._dtype,
                max_length=self._max_length,
            )
            if blk.numel() > 0:
                blocks.append(blk)
        if not blocks:
            return torch.ones(1, 1, device=self._device, dtype=self._dtype)
        if len(blocks) == 1:
            return blocks[0]
        return torch.block_diag(*blocks)
