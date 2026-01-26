"""Dataset loading and normalization."""

from pathlib import Path

import pandas as pd

from .constants import TEXT_COLUMNS


def _normalize_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if pd.isna(value):
        return ""
    return str(value).strip()


def load_data(path: Path | str) -> pd.DataFrame:
    dataset = Path(path)
    if not dataset.exists():
        raise FileNotFoundError(f"Не найден файл с данными: {dataset}")

    df = pd.read_csv(dataset)

    for col in TEXT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].apply(_normalize_cell)

    if "brand" not in df.columns:
        df["brand"] = ""
    else:
        df["brand"] = df["brand"].apply(_normalize_cell)

    return df
