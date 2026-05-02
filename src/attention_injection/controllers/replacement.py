import torch
from omegaconf import DictConfig

from src.attention_injection.bundle import build_mappers
from src.attention_injection.controllers.base import AttentionControllerBase
from src.attention_injection.injection_utils import infer_attention_head_query_key
from src.attention_injection.mappers import CaptionsReplacementMapper, LyricReplacementMapper
from src.utils.conditioning import p2p_src_tgt_prompt_configs, prepare_conditions


class ReplacementAttentionController(AttentionControllerBase):
    def __init__(self) -> None:
        super().__init__()
        self._enabled = False
        self._ly_src: slice = slice(0, 0)
        self._ly_tgt: slice = slice(0, 0)
        self._cap_src: slice = slice(0, 0)
        self._cap_tgt: slice = slice(0, 0)
        self._lyrics_mapper: LyricReplacementMapper | None = None
        self._captions_mapper: CaptionsReplacementMapper | None = None

    def build(self, *, handler, cfg: DictConfig, writer=None) -> None:
        src_raw, tgt_raw = p2p_src_tgt_prompt_configs(cfg.p2p_task)
        prompts = [src_raw, tgt_raw]
        duration = float(cfg.duration)
        cond_fwd, unpack = prepare_conditions(handler, prompts, duration, return_unpack=True)

        text_m = unpack["text_attention_mask"].bool()
        lyric_m = unpack["lyric_attention_mask"].bool()
        enc_m = cond_fwd.encoder_attention_mask.bool()

        n_ly_src = int(lyric_m[0].sum().item())
        n_cap_src = int(text_m[0].sum().item())
        n_val_src = int(enc_m[0].sum().item())
        n_t_src = n_val_src - n_ly_src - n_cap_src

        n_ly_tgt = int(lyric_m[1].sum().item())
        n_cap_tgt = int(text_m[1].sum().item())
        n_val_tgt = int(enc_m[1].sum().item())
        n_t_tgt = n_val_tgt - n_ly_tgt - n_cap_tgt

        bundle = build_mappers(
            src_raw,
            tgt_raw,
            handler.text_tokenizer,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        ly_mat = bundle.lyrics.matrix
        cap_mat = bundle.captions.matrix

        ly_src_len = min(int(ly_mat.shape[1]), n_ly_src)
        ly_tgt_len = min(int(ly_mat.shape[0]), n_ly_tgt)
        cap_src_len = min(int(cap_mat.shape[1]), n_cap_src)
        cap_tgt_len = min(int(cap_mat.shape[0]), n_cap_tgt)

        self._ly_src = slice(0, ly_src_len)
        self._ly_tgt = slice(0, ly_tgt_len)
        self._cap_src = slice(n_ly_src + n_t_src, n_ly_src + n_t_src + cap_src_len)
        self._cap_tgt = slice(n_ly_tgt + n_t_tgt, n_ly_tgt + n_t_tgt + cap_tgt_len)

        self._lyrics_mapper = bundle.lyrics if ly_src_len and ly_tgt_len else None
        self._captions_mapper = bundle.captions if cap_src_len and cap_tgt_len else None
        self._enabled = self._lyrics_mapper is not None or self._captions_mapper is not None

        if writer is not None:
            writer.add_text(
                "edit/replacement_mapper_lyrics",
                bundle.lyrics_mapping_text,
            )
            writer.add_text(
                "edit/replacement_mapper_captions",
                bundle.captions_mapping_text,
            )

    def forward(self, attn_weight: torch.Tensor) -> torch.Tensor:
        if not self._enabled:
            return attn_weight

        b, _, _, _ = infer_attention_head_query_key(attn_weight)
        if b != 2 or attn_weight.dim() != 4:
            return attn_weight

        out = attn_weight.clone()
        src = out[0]
        tgt = out[1]

        if self._lyrics_mapper is not None:
            ly_src = src[..., self._ly_src]
            ly_tgt = self._lyrics_mapper.apply(ly_src)
            tgt[..., self._ly_tgt] = ly_tgt

        if self._captions_mapper is not None:
            cap_src = src[..., self._cap_src]
            cap_tgt = self._captions_mapper.apply(cap_src)
            tgt[..., self._cap_tgt] = cap_tgt

        out[1] = tgt
        return out
