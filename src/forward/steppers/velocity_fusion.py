"""UniEdit velocity fusion: ``VelocityFusionLocalStepper`` (шаг) + ``VelocityFusionEditRunner`` (полный edit)."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, Type, cast

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

from src.artifact_bundle import GenerationArtifactPayload
from src.forward.conditioning import prepare_condition_from_payload
from src.forward.diffusion_driver import run_euler_ode_on_pairs
from src.forward.steppers.base import LocalDiffusionStepper
from src.forward.steppers.plain_cfg_stepper import PlainCfgEulerStepper
from src.logging import utils as logging
from src.nti.invert_conditioning import build_invert_payload
from src.nti.schedule import acestep_sigma_grid
from src.utils.utils import resolve_against_original_cwd


def _hann_window_stft(win_length: int, device: torch.device) -> torch.Tensor:
    return torch.hann_window(win_length, device=device, dtype=torch.float32)


def _stft_latent_along_l(
    x: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
    center: bool,
    window: torch.Tensor,
) -> torch.Tensor:
    """STFT вдоль оси ``L`` для ``(B, L, D)`` реального ``x``; по одному сигналу на пару ``(b, d)``.

    Возвращает комплекс ``(B, D, n_fft//2+1, n_frames)``.
    """
    if x.dim() != 3:
        raise ValueError(f"STFT latent: expected (B, L, D), got {tuple(x.shape)}")
    B, L, D = x.shape
    if L < n_fft:
        raise ValueError(f"STFT latent: L={L} must be >= n_fft={n_fft}")
    x_fd = x.permute(0, 2, 1).reshape(B * D, L).to(torch.float32)
    w = window.to(device=x_fd.device, dtype=torch.float32)
    return torch.stft(
        x_fd,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=w,
        center=center,
        return_complex=True,
    ).view(B, D, n_fft // 2 + 1, -1)


def _istft_latent_to_l(
    Z: torch.Tensor,
    L_target: int,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
    center: bool,
    window: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """iSTFT обратно в ``(B, L_target, D)`` из комплексного ``(B, D, F, T)``."""
    if Z.dim() != 4 or not Z.is_complex():
        raise ValueError(f"iSTFT latent: expected complex 4D (B,D,F,T), got {tuple(Z.shape)} {Z.dtype}")
    B, D, _F, _T = Z.shape
    Z2 = Z.reshape(B * D, Z.shape[2], Z.shape[3])
    w = window.to(device=Z2.device, dtype=torch.float32)
    y = torch.istft(
        Z2,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=w,
        center=center,
        length=L_target,
    )
    return y.view(B, D, L_target).permute(0, 2, 1).to(out_dtype)


def _minmax_mask_time(m: torch.Tensor) -> torch.Tensor:
    """UniEdit-style min–max по всем позициям ``L`` для батча: ``m`` shape ``(B, L, 1)``."""
    b = m.shape[0]
    lo = m.reshape(b, -1).min(dim=1).values.view(b, 1, 1)
    hi = m.reshape(b, -1).max(dim=1).values.view(b, 1, 1)
    return (m - lo) / (hi - lo + 1e-7)


def _decoder_velocity_cond_only(
    model: torch.nn.Module,
    xt: torch.Tensor,
    t_scalar: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    context_latents: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    bsz = xt.shape[0]
    device, dtype = xt.device, xt.dtype
    t_tensor = t_scalar * torch.ones((bsz,), device=device, dtype=dtype)
    past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
    out = model.decoder(
        hidden_states=xt,
        timestep=t_tensor,
        timestep_r=t_tensor,
        attention_mask=attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        context_latents=context_latents,
        use_cache=False,
        past_key_values=past_key_values,
    )
    return out[0]


def _time_domain_uniedit_delta(
    v_src: torch.Tensor,
    v_tgt: torch.Tensor,
    *,
    mask: torch.Tensor,
    h_sigma: torch.Tensor,
    omega: float,
) -> torch.Tensor:
    """Correction + fusion в time-domain (латент ``B,L,D``), маска ``(B,L,1)``."""
    guidance = v_tgt - v_src
    h = h_sigma
    while h.dim() < guidance.dim():
        h = h.unsqueeze(-1)
    stride_corr = float(omega) * h * (1.0 + mask) * guidance
    velocity_fusion = mask * v_tgt + (1.0 - mask) * v_src
    return stride_corr + h * velocity_fusion


class VelocityFusionLocalStepper(LocalDiffusionStepper):
    """Один шаг UniEdit-Euler (два decoder-вызова + fusion); тот же ODE-драйвер, что и для plain."""

    def __init__(
        self,
        *,
        omega: float,
        alpha: float,
        model: torch.nn.Module,
        enc_s: torch.Tensor,
        mask_s: torch.Tensor,
        ctx_s: torch.Tensor,
        attn_s: torch.Tensor,
        enc_t: torch.Tensor,
        mask_t: torch.Tensor,
        ctx_t: torch.Tensor,
        attn_t: torch.Tensor,
    ) -> None:
        self.omega = float(omega)
        self.alpha = float(alpha)
        self.model = model
        self.enc_s = enc_s
        self.mask_s = mask_s
        self.ctx_s = ctx_s
        self.attn_s = attn_s
        self.enc_t = enc_t
        self.mask_t = mask_t
        self.ctx_t = ctx_t
        self.attn_t = attn_t

    def _uniedit_velocity_delta(
        self, v_src: torch.Tensor, v_tgt: torch.Tensor, *, h_sigma: torch.Tensor
    ) -> torch.Tensor:
        guidance = v_tgt - v_src
        mask = guidance.mean(dim=2, keepdim=True)
        mask = _minmax_mask_time(mask)
        return _time_domain_uniedit_delta(v_src, v_tgt, mask=mask, h_sigma=h_sigma, omega=self.omega)

    def step(self, xt: torch.Tensor, t_curr: float, t_prev: float, step_idx: int) -> torch.Tensor:
        del step_idx
        device, dtype = xt.device, xt.dtype
        t_c = torch.as_tensor(t_curr, device=device, dtype=dtype)
        h_sigma = torch.as_tensor(t_prev - t_curr, device=device, dtype=dtype)
        v_src = _decoder_velocity_cond_only(
            self.model, xt, t_c, self.enc_s, self.mask_s, self.ctx_s, self.attn_s
        )
        v_tgt = _decoder_velocity_cond_only(
            self.model, xt, t_c, self.enc_t, self.mask_t, self.ctx_t, self.attn_t
        )
        return xt + self._uniedit_velocity_delta(v_src, v_tgt, h_sigma=h_sigma)


class SpectralTimeVelocityFusionLocalStepper(VelocityFusionLocalStepper):
    """Вариант (2): маска из ``|STFT(v_tgt - v_src)|`` по оси ``L``, проекция в ``(B,L,1)``; fusion/correction в time."""

    def __init__(
        self,
        *,
        n_fft: int,
        hop_length: int,
        win_length: int,
        center: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._stft_n_fft = int(n_fft)
        self._stft_hop = int(hop_length)
        self._stft_win = int(win_length)
        self._stft_center = bool(center)

    def _uniedit_velocity_delta(
        self, v_src: torch.Tensor, v_tgt: torch.Tensor, *, h_sigma: torch.Tensor
    ) -> torch.Tensor:
        _B, L, _D = v_src.shape
        window = _hann_window_stft(self._stft_win, v_src.device)
        # ``STFT(v_tgt - v_src)`` (для фиксированного STFT совпадает с ``STFT(v_tgt)-STFT(v_src)``).
        Z_diff = _stft_latent_along_l(
            v_tgt - v_src,
            n_fft=self._stft_n_fft,
            hop_length=self._stft_hop,
            win_length=self._stft_win,
            center=self._stft_center,
            window=window,
        )
        mag = Z_diff.abs()
        mag_fd = mag.mean(dim=2)  # (B, D, T_stft)
        mag_bt = mag_fd.mean(dim=1, keepdim=True)  # (B, 1, T_stft)
        m_lp = F.interpolate(mag_bt, size=L, mode="linear", align_corners=False)
        m = m_lp.transpose(1, 2).contiguous()
        mask = _minmax_mask_time(m)
        return _time_domain_uniedit_delta(v_src, v_tgt, mask=mask, h_sigma=h_sigma, omega=self.omega)


class SpectralSpectralVelocityFusionLocalStepper(VelocityFusionLocalStepper):
    """Вариант (3): correction и blend скоростей в STFT-домене, один iSTFT в ``(B,L,D)``."""

    def __init__(
        self,
        *,
        n_fft: int,
        hop_length: int,
        win_length: int,
        center: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._stft_n_fft = int(n_fft)
        self._stft_hop = int(hop_length)
        self._stft_win = int(win_length)
        self._stft_center = bool(center)

    def _uniedit_velocity_delta(
        self, v_src: torch.Tensor, v_tgt: torch.Tensor, *, h_sigma: torch.Tensor
    ) -> torch.Tensor:
        _B, L, _D = v_src.shape
        out_dtype = v_src.dtype
        window = _hann_window_stft(self._stft_win, v_src.device)
        Zs = _stft_latent_along_l(
            v_src.to(torch.float32),
            n_fft=self._stft_n_fft,
            hop_length=self._stft_hop,
            win_length=self._stft_win,
            center=self._stft_center,
            window=window,
        )
        Zt = _stft_latent_along_l(
            v_tgt.to(torch.float32),
            n_fft=self._stft_n_fft,
            hop_length=self._stft_hop,
            win_length=self._stft_win,
            center=self._stft_center,
            window=window,
        )
        V_minus = Zt - Zs
        mag = V_minus.abs()
        b = mag.shape[0]
        lo = mag.reshape(b, -1).min(dim=1).values.view(b, 1, 1, 1)
        hi = mag.reshape(b, -1).max(dim=1).values.view(b, 1, 1, 1)
        M = (mag - lo) / (hi - lo + 1e-7)
        h = h_sigma
        while h.dim() < M.dim():
            h = h.unsqueeze(-1)
        h32 = h.to(torch.float32)
        fused = M * Zt + (1.0 - M) * Zs
        Delta_spec = float(self.omega) * h32 * (1.0 + M) * V_minus + h32 * fused
        return _istft_latent_to_l(
            Delta_spec,
            L,
            n_fft=self._stft_n_fft,
            hop_length=self._stft_hop,
            win_length=self._stft_win,
            center=self._stft_center,
            window=window,
            out_dtype=out_dtype,
        )


_STEPPER_FOR_FUSION_MODE: dict[str, Type[VelocityFusionLocalStepper]] = {
    "time": VelocityFusionLocalStepper,
    "spectral_time": SpectralTimeVelocityFusionLocalStepper,
    "spectral_spectral": SpectralSpectralVelocityFusionLocalStepper,
}


class VelocityFusionEditRunner:
    """Hydra: ``omega``/``alpha``. При ``alpha<1``: сначала plain ODE по **src** на префиксе сетки \(\sigma\)
    (как отдельная диффузия длиной ``skip``), затем velocity fusion на хвосте (как UniEdit по шагам).
    При ``alpha==1`` — только fusion на полной сетке."""

    def __init__(
        self,
        *,
        omega: float = 5.0,
        alpha: float = 0.6,
        fusion_mode: str = "time",
        stft_n_fft: int = 256,
        stft_hop_length: int = 64,
        stft_win_length: int = 256,
        stft_center: bool = True,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.omega = float(omega)
        self.alpha = float(alpha)
        mode = str(fusion_mode).strip().lower()
        if mode not in _STEPPER_FOR_FUSION_MODE:
            raise ValueError(
                f"fusion_mode must be one of {sorted(_STEPPER_FOR_FUSION_MODE)}, got {fusion_mode!r}"
            )
        self.fusion_mode = mode
        self._stft_n_fft = int(stft_n_fft)
        self._stft_hop_length = int(stft_hop_length)
        self._stft_win_length = int(stft_win_length)
        self._stft_center = bool(stft_center)
        if mode != "time":
            if not (0 < self._stft_hop_length <= self._stft_win_length <= self._stft_n_fft):
                raise ValueError(
                    "spectral fusion requires 0 < stft_hop_length <= stft_win_length <= stft_n_fft, "
                    f"got hop={self._stft_hop_length}, win={self._stft_win_length}, n_fft={self._stft_n_fft}"
                )

    def run(self, handler: Any, work_cfg: DictConfig, artifact: GenerationArtifactPayload) -> dict[str, Any]:
        mlx = bool(getattr(handler, "use_mlx_dit", False)) and getattr(handler, "mlx_decoder", None) is not None
        if mlx:
            raise RuntimeError("velocity_fusion edit requires PyTorch DiT (acestep.use_mlx_dit=false).")

        p2p = work_cfg.p2p_task
        src_c, src_l = str(p2p.src.captions), str(p2p.src.lyrics)
        tgt_c, tgt_l = str(p2p.tgt.captions), str(p2p.tgt.lyrics)
        vocal = str(OmegaConf.select(p2p, "vocal_language", default="en"))

        raw_art = OmegaConf.select(work_cfg, "artifact_path", default=None)
        art_path = (
            Path(resolve_against_original_cwd(str(raw_art)))
            if raw_art is not None and str(raw_art).strip() not in ("", "null", "None")
            else None
        )
        music_path = Path(resolve_against_original_cwd(str(work_cfg.source_audio_path)))
        init_noise = artifact.noise
        if init_noise is None:
            if art_path is None or not art_path.is_file():
                raise FileNotFoundError(
                    "velocity_fusion needs ``artifact_path`` to an existing artifact file, or ``artifact.noise`` "
                    "from in-memory inversion."
                )
        if not music_path.is_file():
            raise FileNotFoundError(f"source_audio_path not found: {music_path}")

        infer_steps = int(work_cfg.inference_steps)
        seed = int(work_cfg.seed) if not bool(work_cfg.use_random_seed) else 0

        wav = handler.process_target_audio(str(music_path))
        if wav is None:
            raise RuntimeError(f"Failed to load audio: {music_path}")

        payload_src, _ = build_invert_payload(
            handler,
            captions=src_c,
            lyrics=src_l,
            vocal_language=vocal,
            music_stereo_48k=wav,
            infer_steps=infer_steps,
            seed=seed,
        )
        payload_tgt, _ = build_invert_payload(
            handler,
            captions=tgt_c,
            lyrics=tgt_l,
            vocal_language=vocal,
            music_stereo_48k=wav,
            infer_steps=infer_steps,
            seed=seed,
        )

        if init_noise is not None:
            noise = init_noise
        else:
            noise = _load_artifact_noise(art_path)
        if noise.dim() == 2:
            noise = noise.unsqueeze(0)
        exp_t = int(payload_src["src_latents"].shape[1])
        if int(noise.shape[1]) != exp_t:
            raise ValueError(
                f"Artifact noise T={noise.shape[1]} != expected latent length T={exp_t} from source audio / payload."
            )

        if getattr(handler, "config", None) is not None and bool(getattr(handler.config, "is_turbo", False)):
            raise RuntimeError("velocity_fusion requires a non-turbo ACE-Step checkpoint.")

        if str(work_cfg.infer_method) != "ode":
            raise ValueError(f"velocity_fusion: only infer_method='ode' is supported, got {work_cfg.infer_method!r}")
        if str(work_cfg.sampler_mode) == "heun":
            raise NotImplementedError(
                "velocity_fusion: Heun not implemented; use sampler_mode=euler (UniEdit-Flow diffusers uses Euler)."
            )
        if payload_src["src_latents"].shape != payload_tgt["src_latents"].shape:
            raise ValueError(
                f"src/tgt payload src_latents mismatch {tuple(payload_src['src_latents'].shape)} "
                f"vs {tuple(payload_tgt['src_latents'].shape)}"
            )

        model = handler.model
        model.eval()

        enc_s, mask_s, ctx_s, attn_s = prepare_condition_from_payload(model, handler, payload_src)
        enc_t, mask_t, ctx_t, attn_t = prepare_condition_from_payload(model, handler, payload_tgt)

        device, dtype = enc_s.device, enc_s.dtype
        guidance_scale = float(work_cfg.guidance_scale)
        sample_steps = max(1, int(math.floor(self.alpha * float(infer_steps))))
        skip = infer_steps - sample_steps
        t_vec = acestep_sigma_grid(infer_steps, float(work_cfg.shift), device=device, dtype=dtype)
        t_trim = t_vec[skip:]
        fusion_pairs = list(zip(t_trim[:-1], t_trim[1:]))
        if len(fusion_pairs) < 1:
            raise ValueError(
                f"velocity_fusion: effective fusion steps is {len(fusion_pairs)}; "
                "increase inference_steps or alpha."
            )

        prefix_pairs = [(t_vec[i], t_vec[i + 1]) for i in range(skip)]

        if guidance_scale > 1.0:
            context_lat_plain = torch.cat([ctx_s, ctx_s], dim=0)
            attention_mask_plain = torch.cat([attn_s, attn_s], dim=0)
        else:
            context_lat_plain = ctx_s
            attention_mask_plain = attn_s

        null_list = artifact.null_encoder_hidden_states_per_step
        if null_list is not None and guidance_scale > 1.0 and len(null_list) != infer_steps:
            raise ValueError(
                f"artifact null_encoder_hidden_states_per_step len {len(null_list)} != inference_steps={infer_steps}"
            )
        if null_list is not None and guidance_scale <= 1.0:
            raise ValueError("null_encoder_hidden_states_per_step requires guidance_scale > 1.0")

        plain_stepper = PlainCfgEulerStepper(
            model,
            enc_s,
            mask_s,
            context_lat_plain,
            attention_mask_plain,
            int(noise.shape[0]),
            use_adg=bool(work_cfg.use_adg),
            guidance_scale=guidance_scale,
            cfg_interval_start=float(work_cfg.cfg_interval_start),
            cfg_interval_end=float(work_cfg.cfg_interval_end),
            infer_method=str(work_cfg.infer_method),
            sampler_mode=str(work_cfg.sampler_mode),
            null_encoder_hidden_states_per_step=null_list,
        )

        stepper_cls = _STEPPER_FOR_FUSION_MODE[self.fusion_mode]
        common_kw: dict[str, Any] = dict(
            omega=self.omega,
            alpha=self.alpha,
            model=model,
            enc_s=enc_s,
            mask_s=mask_s,
            ctx_s=ctx_s,
            attn_s=attn_s,
            enc_t=enc_t,
            mask_t=mask_t,
            ctx_t=ctx_t,
            attn_t=attn_t,
        )
        if self.fusion_mode == "time":
            local = stepper_cls(**common_kw)
        else:
            if int(noise.shape[1]) < self._stft_n_fft:
                raise ValueError(
                    f"fusion_mode={self.fusion_mode!r} requires latent length L >= stft_n_fft "
                    f"({int(noise.shape[1])} < {self._stft_n_fft})"
                )
            local = stepper_cls(
                n_fft=self._stft_n_fft,
                hop_length=self._stft_hop_length,
                win_length=self._stft_win_length,
                center=self._stft_center,
                **common_kw,
            )
        xt = noise.to(device=device, dtype=dtype)
        use_pb = not bool(getattr(handler, "disable_tqdm", False))
        t0 = time.perf_counter()
        with torch.inference_mode():
            if skip > 0:
                xt = run_euler_ode_on_pairs(
                    xt,
                    prefix_pairs,
                    plain_stepper,
                    use_progress_bar=use_pb,
                    desc="Edit prefix (plain src)",
                )
            pred_latents = run_euler_ode_on_pairs(
                xt,
                fusion_pairs,
                local,
                use_progress_bar=use_pb,
                desc=f"Velocity fusion ({self.fusion_mode})",
            )
        elapsed = time.perf_counter() - t0
        logging.info(
            f"velocity_fusion done: mode={self.fusion_mode} omega={self.omega} alpha={self.alpha} "
            f"infer_steps={infer_steps} prefix_plain={skip} fusion={len(fusion_pairs)} diffusion_s={elapsed:.2f}"
        )

        pred_latents = cast(torch.Tensor, pred_latents)
        pred_latents_for_decode = pred_latents.transpose(1, 2).contiguous().to(handler.vae.dtype)
        with torch.inference_mode():
            with handler._load_model_context("vae"):
                pred_wavs = handler.tiled_decode(pred_latents_for_decode)
        if pred_wavs.dtype != torch.float32:
            pred_wavs = pred_wavs.float()
        peak = pred_wavs.abs().amax(dim=[1, 2], keepdim=True)
        if torch.any(peak > 1.0):
            pred_wavs = pred_wavs / peak.clamp(min=1.0)

        sr = int(getattr(handler, "sample_rate", 48_000))
        audios = [{"tensor": pred_wavs[i].cpu(), "sample_rate": sr} for i in range(pred_wavs.shape[0])]
        return {
            "success": True,
            "audios": audios,
            "status_message": "velocity_fusion edit finished",
            "extra_outputs": {},
            "error": None,
        }


def _load_artifact_noise(artifact_path: Path) -> torch.Tensor:
    bundle = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict) or "noise" not in bundle:
        raise KeyError(f"Artifact {artifact_path} must be a dict with key 'noise'")
    noise = bundle["noise"]
    if not isinstance(noise, torch.Tensor):
        raise TypeError(f"artifact['noise'] must be torch.Tensor, got {type(noise)}")
    return noise
