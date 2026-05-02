_applied = False


def apply_adg_mps_patch() -> None:
    global _applied
    if _applied:
        return
    _applied = True

    import torch
    import acestep.models.common.apg_guidance as capg

    call_cos_tensor = capg.call_cos_tensor
    compute_perpendicular_component = capg.compute_perpendicular_component

    def adg_forward(
        latents,
        noise_pred_cond,
        noise_pred_uncond,
        sigma,
        guidance_scale: float,
        angle_clip: float = 3.14 / 6,
        apply_norm: bool = False,
        apply_clip: bool = True,
    ):
        if latents.shape[1] != noise_pred_cond.shape[1]:
            if noise_pred_cond.shape[1] % latents.shape[1] != 0:
                raise ValueError(
                    "noise_pred_cond time dimension must be a whole-number multiple of latents time dimension."
                )
            repeats = noise_pred_cond.shape[1] // latents.shape[1]
            latents = latents.repeat_interleave(repeats, dim=1)

        n = noise_pred_cond.shape[0]
        noise_pred_text = noise_pred_cond
        n, t, c = noise_pred_text.shape

        if isinstance(sigma, (int, float)):
            sigma = torch.tensor(sigma, device=latents.device, dtype=latents.dtype)
            sigma = sigma.view(1, 1, 1).expand(n, 1, 1)
        elif torch.is_tensor(sigma):
            if sigma.numel() == 1:
                sigma = sigma.view(1, 1, 1).expand(n, 1, 1)
            elif sigma.numel() == n:
                sigma = sigma.view(n, 1, 1)
            else:
                raise ValueError(f"sigma has incompatible shape. Expected scalar or size {n}, got {sigma.shape}")
        else:
            raise TypeError(f"sigma must be a number or tensor, got {type(sigma)}")

        weight = guidance_scale - 1
        weight = weight * (weight > 0) + 1e-3

        latent_hat_text = latents - sigma * noise_pred_text
        latent_hat_uncond = latents - sigma * noise_pred_uncond
        latent_diff = latent_hat_text - latent_hat_uncond

        cos_theta = call_cos_tensor(
            latent_hat_text.view(-1, c).float(),
            latent_hat_uncond.reshape(-1, c).contiguous().float(),
        ).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        latent_theta = torch.acos(cos_theta).view(n, t, 1)
        latent_theta_new = (
            torch.clip(weight * latent_theta, -angle_clip, angle_clip)
            if apply_clip
            else weight * latent_theta
        )
        proj, perp = compute_perpendicular_component(latent_diff, latent_hat_uncond)
        latent_v_new = torch.cos(latent_theta_new) * latent_hat_text

        latent_p_new = perp * torch.sin(latent_theta_new) / torch.sin(latent_theta) * (
            torch.sin(latent_theta) > 1e-3
        ) + perp * weight * (torch.sin(latent_theta) <= 1e-3)
        latent_new = latent_v_new + latent_p_new
        if apply_norm:
            latent_new = latent_new * torch.linalg.norm(latent_hat_text, dim=1, keepdim=True) / torch.linalg.norm(
                latent_new, dim=1, keepdim=True
            )

        noise_pred = (latents - latent_new) / sigma
        noise_pred = noise_pred.reshape(n, t, c).to(latents.dtype)
        return noise_pred

    capg.adg_forward = adg_forward
