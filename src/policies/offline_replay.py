from __future__ import annotations

import json
from pathlib import Path

from src.policies.base import PolicyResult, state_id_from


class OfflineReplayPolicy:
    """Read saved policy outputs instead of fabricating model behavior."""

    def __init__(self, replay_path: str | Path) -> None:
        self.replay_path = Path(replay_path)
        self._results = self._load_results()

    def _load_results(self) -> dict[str, PolicyResult]:
        raw = json.loads(self.replay_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("Offline replay file must contain a JSON list of policy results.")
        results = [PolicyResult.model_validate(item) for item in raw]
        return {result.input_state_id: result for result in results}

    def run(self, state: dict) -> PolicyResult:
        input_state_id = state_id_from(state)
        if input_state_id not in self._results:
            raise KeyError(f"No offline replay result found for state '{input_state_id}'.")
        return self._results[input_state_id]

