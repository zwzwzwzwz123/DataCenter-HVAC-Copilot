from __future__ import annotations

from pathlib import Path

from src.policies.base import PolicyResult


class DiffusionPolicyAdapter:
    """Adapter boundary for DiffFNO / Guided-DiffFNO policy backends."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path else None

    def run(self, state: dict) -> PolicyResult:
        if self.model_path is None:
            raise NotImplementedError(
                "Diffusion policy backend is not configured. Use offline_replay until a real model is available."
            )
        raise NotImplementedError(
            "Diffusion policy inference is not implemented in stage 1. Wire a real backend here."
        )

