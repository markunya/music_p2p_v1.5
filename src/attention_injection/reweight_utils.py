import re
from dataclasses import dataclass
from typing import Any, Literal

import torch

from src.schemas import PromptConfig
from src.utils.conditioning import ModelCondition

REWEIGHT_MARK = "\u203b"
_RE_LYRIC_WORD = re.compile(
    r"([^\s" + re.escape(REWEIGHT_MARK) + r"]+)(" + re.escape(REWEIGHT_MARK) + r")"
)


@dataclass(frozen=True)
class ReweightTarget:
    field: Literal["captions", "lyrics"]
    text: str


def strip_reweight_marks(s: str) -> str:
    return s.replace(REWEIGHT_MARK, "")


def _has_marks(s: str) -> bool:
    return REWEIGHT_MARK in s


def _parse_caption_tag_targets(captions_raw: str) -> list[str]:
    if not _has_marks(captions_raw):
        return []
    out: list[str] = []
    for part in captions_raw.split(","):
        s = part.strip()
        if not s:
            continue
        if s.endswith(REWEIGHT_MARK):
            t = s[: -1].strip()
            if not t:
                raise ValueError(
                    f"Invalid reweight: empty tag before {REWEIGHT_MARK!r} in captions segment: {part!r}"
                )
            out.append(t)
    return out


def _parse_lyric_word_targets(lyrics_raw: str) -> list[str]:
    if not _has_marks(lyrics_raw):
        return []
    return [m.group(1) for m in _RE_LYRIC_WORD.finditer(lyrics_raw) if m.group(1)]


def parse_reweight_from_tgt(
    raw_tgt: PromptConfig,
) -> tuple[PromptConfig, list[ReweightTarget]]:
    cap = raw_tgt.captions
    lyr = raw_tgt.lyrics
    cap_targets = _parse_caption_tag_targets(cap)
    lyr_targets = _parse_lyric_word_targets(lyr)
    clean = PromptConfig(
        captions=strip_reweight_marks(cap),
        lyrics=strip_reweight_marks(lyr),
        vocal_language=raw_tgt.vocal_language,
    )
    targets: list[ReweightTarget] = [ReweightTarget("captions", t) for t in cap_targets]
    targets += [ReweightTarget("lyrics", t) for t in lyr_targets]
    return clean, targets


def _sft_caption_text_prompt(
    handler: Any,
    *,
    metas: list[Any],
    item_index: int,
    captions: list[str],
    vocal_languages: list[str],
) -> tuple[str, str, str]:
    from acestep.constants import DEFAULT_DIT_INSTRUCTION, SFT_GEN_PROMPT

    if getattr(getattr(handler, "model", None), "config", object()) and getattr(
        handler.model.config, "is_lego_sft", False
    ):
        raise NotImplementedError(
            "Reweight from arrow markers: model has is_lego_sft=True. Use a non–lego-sft DiT for this feature."
        )
    parsed = handler._parse_metas(metas)
    actual_captions, actual_languages = handler._extract_caption_and_language(parsed, captions, vocal_languages)
    ins = handler._normalize_instructions(None, len(captions), DEFAULT_DIT_INSTRUCTION)
    i = item_index
    tp = SFT_GEN_PROMPT.format(handler._format_instruction(ins[i]), actual_captions[i], parsed[i])
    return tp, actual_captions[i], actual_languages[i]


def _sft_caption_only_body(t_prompt: str) -> str:
    mark = "# Caption\n"
    a = t_prompt.find(mark)
    if a < 0:
        return ""
    a += len(mark)
    b = t_prompt.find("\n\n# Metas\n", a)
    if b < 0:
        b = t_prompt.find("\n# Metas\n", a)
    if b < 0:
        return t_prompt[a:].strip()
    return t_prompt[a:b]


def _caption_char_range_in_sft(t_prompt: str, tag: str) -> tuple[int, int]:
    mark = "# Caption\n"
    i0 = t_prompt.find(mark)
    if i0 < 0:
        raise ValueError("Reweight: SFT text_prompt has no '# Caption' block")
    a0 = i0 + len(mark)
    body = _sft_caption_only_body(t_prompt)
    j = body.find(tag)
    if j < 0:
        j = body.lower().find(tag.lower())
    if j < 0:
        raise ValueError(
            f"Reweight: tag {tag!r} not found in SFT # Caption body {body!r} (commas / spaces / cleaned vs raw caption must match the model string)"
        )
    return a0 + j, a0 + j + len(tag)


def _lyrics_formatted(handler: Any, lyrics: str, language: str) -> str:
    return handler._format_lyrics(lyrics, language)


def _char_range_for_substring_in(text: str, query: str, field_name: str) -> tuple[int, int]:
    if not query:
        raise ValueError("Internal: empty reweight target substring")
    j = text.find(query)
    if j < 0:
        j = text.lower().find(query.lower())
    if j < 0:
        raise ValueError(
            f"Reweight: could not find target phrase {query!r} in {field_name} text. "
            "If you edited the prompt, ensure the phrase matches the cleaned caption/lyrics (after removing arrows) exactly, including spaces."
        )
    return j, j + len(query)


def _token_ids_covering(
    handler: Any, full_text: str, c0: int, c1: int, max_length: int
) -> list[int]:
    toker = handler.text_tokenizer
    out = toker(
        full_text,
        add_special_tokens=True,
        return_offsets_mapping=True,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_length,
    )
    om = getattr(out, "offset_mapping", None)
    if om is None:
        raise ValueError("text_tokenizer must return offset_mapping for reweight (HF tokenizer)")
    if om.dim() == 3:
        om0 = om[0]
    elif om.dim() == 2 and int(om.size(-1)) == 2:
        om0 = om
    else:
        raise ValueError(
            f"Unexpected offset_mapping shape {tuple(om.shape)}; expected (S,2) or (1,S,2) (B,S,2)"
        )
    idx: list[int] = []
    for t in range(om0.shape[0]):
        a, b = int(om0[t, 0].item()), int(om0[t, 1].item())
        if a < b and max(a, c0) < min(b, c1):
            idx.append(t)
    if c0 < c1 and not idx and out["input_ids"].shape[1] > 0:
        from loguru import logger

        logger.warning("Reweight: no subword offset overlap for char range [{}, {}); tag may be truncated or tokenizer mismatch", c0, c1)
    return idx


def _k_ranks_caption(
    b: int,
    text_mask: torch.Tensor,
    n1: int,
    token_positions: list[int],
) -> set[int]:
    ks: set[int] = set()
    for tpos in token_positions:
        if tpos < 0 or tpos >= text_mask.shape[1]:
            continue
        if not text_mask[b, tpos].item():
            continue
        c = int(text_mask[b, : tpos + 1].sum().item()) - 1
        ks.add(n1 + c)
    return ks


def _k_ranks_lyric(
    b: int,
    lyric_mask: torch.Tensor,
    token_positions: list[int],
) -> set[int]:
    ks: set[int] = set()
    for tpos in token_positions:
        if tpos < 0 or tpos >= lyric_mask.shape[1]:
            continue
        if not lyric_mask[b, tpos].item():
            continue
        k0 = int(lyric_mask[b, : tpos + 1].sum().item()) - 1
        ks.add(k0)
    return ks


def build_p2p_key_boost(
    handler: Any,
    *,
    metas: list[Any],
    captions: list[str],
    lyrics: list[str],
    vocal_languages: list[str],
    model_condition: ModelCondition,
    unpack: dict[str, Any],
    clean_tgt: PromptConfig,
    targets: list[ReweightTarget],
    batch_tgt_index: int = 1,
) -> list[int]:
    b = int(batch_tgt_index)
    bsz = int(model_condition.encoder_hidden_states.shape[0])
    if b < 0 or b >= bsz:
        raise ValueError(f"batch_tgt_index {b} out of range (batch {bsz})")
    k_dim = int(model_condition.encoder_hidden_states.shape[1])
    text_m = unpack["text_attention_mask"].bool()
    lyric_m = unpack["lyric_attention_mask"].bool()
    enc_m = model_condition.encoder_attention_mask.bool()
    n_ly = int(lyric_m[b].sum().item())
    n_cap = int(text_m[b].sum().item())
    n_val = int(enc_m[b].sum().item())
    n_t = n_val - n_ly - n_cap
    if n_t < 0 or n_ly < 0 or n_cap < 0:
        raise ValueError(
            f"Invalid mask stats for reweight: n_ly={n_ly} n_timbre={n_t} n_cap={n_cap} n_valid={n_val} (K={k_dim})"
        )
    n1 = n_ly + n_t
    if n1 + n_cap > k_dim or n_val > k_dim:
        raise ValueError(f"Encoder key layout mismatch: n1+n_cap={n1 + n_cap} n_valid={n_val} but K={k_dim}")

    bits = [0] * k_dim
    for tgt in targets:
        if tgt.field == "captions":
            t_prompt, _actual_cap, _al = _sft_caption_text_prompt(
                handler, metas=metas, item_index=b, captions=captions, vocal_languages=vocal_languages
            )
            c0, c1 = _caption_char_range_in_sft(t_prompt, tgt.text)
            t_idx = _token_ids_covering(handler, t_prompt, c0, c1, 256)
            ranks = _k_ranks_caption(b, text_m, n1, t_idx)
        else:
            lang = vocal_languages[b] if b < len(vocal_languages) else clean_tgt.vocal_language
            lyr_unfmt = lyrics[b] if b < len(lyrics) else clean_tgt.lyrics
            p0, p1 = _char_range_for_substring_in(lyr_unfmt, tgt.text, "raw lyrics")
            lyr_fmt = _lyrics_formatted(handler, lyr_unfmt, lang)
            head = f"# Languages\n{lang}\n\n# Lyric\n"
            c0, c1 = len(head) + p0, len(head) + p1
            t_idx = _token_ids_covering(handler, lyr_fmt, c0, c1, 2048)
            ranks = _k_ranks_lyric(b, lyric_m, t_idx)
        for r in ranks:
            if 0 <= r < k_dim:
                bits[r] = 1
    return bits


def build_2d_equalizer_for_p2p(
    reweight_strength: float,
    key_boost_mask: list[int],
    *,
    batch_size: int,
    k: int,
    tgt_batch_index: int = 1,
) -> list[list[float]]:
    if len(key_boost_mask) != k:
        raise ValueError(f"key_boost_mask length {len(key_boost_mask)} != K {k}")
    if batch_size < 1 or tgt_batch_index < 0 or tgt_batch_index >= batch_size:
        raise ValueError(f"batch_size={batch_size} tgt_batch_index={tgt_batch_index} invalid")
    w = float(reweight_strength)
    out: list[list[float]] = []
    for bb in range(batch_size):
        if bb == tgt_batch_index:
            out.append([w if int(m) else 1.0 for m in key_boost_mask])
        else:
            out.append([1.0] * k)
    return out
