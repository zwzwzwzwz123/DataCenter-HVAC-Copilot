from __future__ import annotations


STABLE_CONTEXT_VERSION = "2026-05-25"

STABLE_BOUNDARY_SUMMARY = (
    "BEAR is an HVAC simulation and controllable proxy scenario, not production telemetry. "
    "LLMs do not directly generate or write back control actions. Policy actions must come "
    "from policy tools. Answers should cite retrieved contexts or tool results when possible. "
    "Safety audit checks remain active."
)


def get_stable_context() -> dict[str, str]:
    return {
        "version": STABLE_CONTEXT_VERSION,
        "boundary_summary": STABLE_BOUNDARY_SUMMARY,
    }
