from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.ingestion.bear_adapter import build_bear_env, export_bear_rollout


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a BEAR BuildingEnvReal rollout.")
    parser.add_argument("--bear-root", required=True, help="Path to the cloned chz056/BEAR repository.")
    parser.add_argument("--building", default="OfficeSmall")
    parser.add_argument("--weather", default="Hot_Dry")
    parser.add_argument("--location", default="Tucson")
    parser.add_argument("--scenario-id", default="bear_officesmall_tucson_random")
    parser.add_argument("--num-steps", type=int, default=24)
    parser.add_argument("--start-time", default="2026-01-01T00:00:00Z")
    parser.add_argument("--time-resolution-seconds", type=int, default=3600)
    parser.add_argument("--output", default="data/bear_processed/bear_rollout.csv")
    args = parser.parse_args()

    env = build_bear_env(
        bear_root=args.bear_root,
        building=args.building,
        weather=args.weather,
        location=args.location,
    )
    frame = export_bear_rollout(
        env=env,
        scenario_id=args.scenario_id,
        num_steps=args.num_steps,
        start_time=args.start_time,
        time_resolution_seconds=args.time_resolution_seconds,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(f"Saved BEAR rollout to {output_path}")


if __name__ == "__main__":
    main()

