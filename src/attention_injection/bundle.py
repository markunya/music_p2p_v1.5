from dataclasses import dataclass

from src.attention_injection.mappers.captions import CaptionsReplacementMapper, split_caption_tags
from src.attention_injection.mappers.lyrics import LyricReplacementMapper
from src.attention_injection.tokenize import TextTokenizerLike, tokenize_caption_raw, tokenize_lyrics_for_mapper
from src.schemas import PromptConfig


@dataclass(frozen=True)
class AttentionInjectionBundle:

    lyrics: LyricReplacementMapper
    captions: CaptionsReplacementMapper
    lyrics_mapping_text: str = ""
    captions_mapping_text: str = ""


def _ids_to_tokens(tokenizer: TextTokenizerLike, ids: list[int]) -> list[str]:
    conv = getattr(tokenizer, "convert_ids_to_tokens", None)
    if callable(conv):
        out = conv(ids)
        if isinstance(out, list):
            return [str(x) for x in out]
    return [str(i) for i in ids]


def _mapper_token_lists(mapper: LyricReplacementMapper | CaptionsReplacementMapper) -> tuple[list[str], list[str]]:
    if isinstance(mapper, LyricReplacementMapper):
        src_ids = tokenize_lyrics_for_mapper(
            mapper._src_lyrics,
            mapper._vocal_language,
            mapper._tokenizer,
            max_length=mapper._max_length,
        ).active_ids()
        tgt_ids = tokenize_lyrics_for_mapper(
            mapper._tgt_lyrics,
            mapper._vocal_language,
            mapper._tokenizer,
            max_length=mapper._max_length,
        ).active_ids()
        return _ids_to_tokens(mapper._tokenizer, src_ids), _ids_to_tokens(mapper._tokenizer, tgt_ids)

    src_ids: list[int] = []
    tgt_ids: list[int] = []
    for src_tag, tgt_tag in zip(split_caption_tags(mapper._src), split_caption_tags(mapper._tgt)):
        src_ids.extend(
            tokenize_caption_raw(
                src_tag,
                mapper._tokenizer,
                max_length=mapper._max_length,
            ).active_ids()
        )
        tgt_ids.extend(
            tokenize_caption_raw(
                tgt_tag,
                mapper._tokenizer,
                max_length=mapper._max_length,
            ).active_ids()
        )
    return _ids_to_tokens(mapper._tokenizer, src_ids), _ids_to_tokens(mapper._tokenizer, tgt_ids)


def _mapper_lines(matrix, src_tokens: list[str], tgt_tokens: list[str]) -> str:
    m = matrix.detach().cpu()
    n_tgt, n_src = int(m.shape[0]), int(m.shape[1])
    src_to_tgt = [set() for _ in range(n_src)]
    tgt_to_src = [set() for _ in range(n_tgt)]
    for j in range(n_tgt):
        nz = (m[j] > 1e-12).nonzero(as_tuple=False).flatten().tolist()
        for i in nz:
            tgt_to_src[j].add(int(i))
            src_to_tgt[int(i)].add(j)

    visited_src: set[int] = set()
    visited_tgt: set[int] = set()
    lines: list[str] = []
    for s0 in range(n_src):
        if s0 in visited_src:
            continue
        stack_src = [s0]
        comp_src: set[int] = set()
        comp_tgt: set[int] = set()
        while stack_src:
            s = stack_src.pop()
            if s in comp_src:
                continue
            comp_src.add(s)
            for t in src_to_tgt[s]:
                if t not in comp_tgt:
                    comp_tgt.add(t)
                    for s2 in tgt_to_src[t]:
                        if s2 not in comp_src:
                            stack_src.append(s2)
        if not comp_tgt:
            continue
        visited_src.update(comp_src)
        visited_tgt.update(comp_tgt)
        left = ", ".join([src_tokens[i] if i < len(src_tokens) else f"<src_{i}>" for i in sorted(comp_src)])
        right = ", ".join([tgt_tokens[j] if j < len(tgt_tokens) else f"<tgt_{j}>" for j in sorted(comp_tgt)])
        lines.append(f"({left}) -> ({right})")

    for t0 in range(n_tgt):
        if t0 in visited_tgt:
            continue
        srcs = sorted(tgt_to_src[t0])
        if not srcs:
            continue
        left = ", ".join([src_tokens[i] if i < len(src_tokens) else f"<src_{i}>" for i in srcs])
        right = tgt_tokens[t0] if t0 < len(tgt_tokens) else f"<tgt_{t0}>"
        lines.append(f"({left}) -> ({right})")

    return "\n".join(lines) if lines else "(no mapper edges)"


def build_mappers(
    src: PromptConfig,
    tgt: PromptConfig,
    tokenizer: TextTokenizerLike,
    *,
    device=None,
    dtype=None,
    lyric_max_length: int = 2048,
    caption_max_length: int = 256,
) -> AttentionInjectionBundle:
    import torch as _torch

    dev = device or _torch.device("cpu")
    dt = dtype or _torch.float32
    lyrics = LyricReplacementMapper(
        src.lyrics,
        tgt.lyrics,
        src.vocal_language,
        tokenizer,
        device=dev,
        dtype=dt,
        max_length=lyric_max_length,
    )
    captions = CaptionsReplacementMapper(
        src.captions,
        tgt.captions,
        tokenizer,
        device=dev,
        dtype=dt,
        max_length=caption_max_length,
    )
    ly_src_tok, ly_tgt_tok = _mapper_token_lists(lyrics)
    cap_src_tok, cap_tgt_tok = _mapper_token_lists(captions)
    return AttentionInjectionBundle(
        lyrics=lyrics,
        captions=captions,
        lyrics_mapping_text=_mapper_lines(lyrics.matrix, ly_src_tok, ly_tgt_tok),
        captions_mapping_text=_mapper_lines(captions.matrix, cap_src_tok, cap_tgt_tok),
    )
