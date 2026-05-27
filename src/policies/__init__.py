"""Policy adapters and fallback control policies."""

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "DROPTCheckpointPolicy":
        from src.policies.dropt_adapter import DROPTCheckpointPolicy

        return DROPTCheckpointPolicy
    raise AttributeError(name)

