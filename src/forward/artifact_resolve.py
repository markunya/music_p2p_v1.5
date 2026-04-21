"""Build ``GenerationArtifactPayload`` for ``ForwardPipeline`` (file or implicit random init)."""

from __future__ import annotations

from omegaconf import DictConfig

from src.artifact_bundle import GenerationArtifactPayload, load_generation_bundle


def forward_artifact_and_work_cfg(cli_cfg: DictConfig) -> tuple[DictConfig, GenerationArtifactPayload]:
    """Resolve ``work_cfg`` + payload: no ``artifact.path`` → empty payload (ACE draws random noise)."""
    work_cfg, loaded = load_generation_bundle(cli_cfg)
    return work_cfg, loaded if loaded is not None else GenerationArtifactPayload()
