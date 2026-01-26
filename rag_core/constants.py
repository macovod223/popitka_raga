"""Project-wide constants."""

import os
from pathlib import Path

# Корень проекта для надежных путей в ноутбуках и скриптах
PROJECT_ROOT = Path(__file__).parent.parent

DEFAULT_DATASET = PROJECT_ROOT / "Best-Buy-dataset-clean.csv"
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DEFAULT_INDICES = (0, 10, 20)
DEFAULT_TOP_K = 5

TEXT_COLUMNS = [
    "title",
    "root_category",
    "product_description",
    "features_summary",
    "product_specifications",
]
