from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ingestion.bear_schema import normalize_bear_trajectory


def load_processed_bear_trajectory(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    frame = pd.read_csv(csv_path)
    frame = normalize_bear_trajectory(frame)
    frame.attrs["source"] = "processed_bear_csv"
    frame.attrs["source_path"] = str(csv_path)
    return frame

