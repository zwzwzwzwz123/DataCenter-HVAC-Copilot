from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class FieldProvenance(str, Enum):
    """Where a standardized trajectory field comes from."""

    NATIVE = "native"
    DERIVED = "derived"
    OPTIONAL_DERIVED = "optional_derived"
    OPTIONAL_SYNTHETIC = "optional_synthetic"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    dtype: str
    provenance: FieldProvenance
    required: bool
    description: str


def make_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "max": None,
            "sum": 0.0,
        }

    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "sum": sum(values),
    }

