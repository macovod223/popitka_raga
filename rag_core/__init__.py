"""Shared utilities for retrieval pipelines."""

from .cli import parse_example_indices
from .components import add_component_text, build_component
from .constants import (
    DEFAULT_DATASET,
    DEFAULT_EMBED_MODEL,
    DEFAULT_INDICES,
    DEFAULT_TOP_K,
    TEXT_COLUMNS,
)
from .data import load_data
from .embeddings import (
    SimpleVectorIndex,
    describe_embedding_model,
    embed_texts,
    load_cached_embeddings,
    load_embedding_model,
    save_cached_embeddings,
)

__all__ = [
    "DEFAULT_DATASET",
    "DEFAULT_EMBED_MODEL",
    "DEFAULT_INDICES",
    "DEFAULT_TOP_K",
    "TEXT_COLUMNS",
    "add_component_text",
    "build_component",
    "describe_embedding_model",
    "embed_texts",
    "load_cached_embeddings",
    "load_data",
    "load_embedding_model",
    "parse_example_indices",
    "save_cached_embeddings",
    "SimpleVectorIndex",
]
