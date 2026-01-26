"""CLI helpers."""

from typing import List


def parse_example_indices(raw: str) -> List[int]:
    # парсим список индексов из CLI без привязки к длине
    if not raw:
        raise ValueError("Список индексов пустой")

    indices: List[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        idx = int(chunk)
        indices.append(idx)

    if not indices:
        raise ValueError("После парсинга не осталось индексов")
    return indices
