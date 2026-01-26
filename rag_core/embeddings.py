"""Embedding model utilities and vector index."""

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .constants import PROJECT_ROOT

logger = logging.getLogger("rag_embeddings")


def _resolve_local_snapshot(model_name: str) -> Optional[str]:
    if not model_name:
        return None
    candidate = Path(model_name)
    if candidate.exists():
        return str(candidate)

    cache_root = os.getenv("HF_HOME")
    if cache_root:
        hub_root = Path(cache_root) / "hub"
    else:
        hub_root = Path.home() / ".cache" / "huggingface" / "hub"

    model_dir = hub_root / f"models--{model_name.replace('/', '--')}"
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    snapshots = []
    try:
        snapshots = sorted(
            (p for p in snapshots_dir.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except Exception:
        snapshots = [p for p in snapshots_dir.iterdir() if p.is_dir()]

    return str(snapshots[0]) if snapshots else None


def load_embedding_model(model_name: str) -> SentenceTransformer:
    # загружаем sentence-transformers модель по названию
    import torch

    logger.info(f"Загружаем модель эмбеддингов: {model_name}")
    logger.info("      (это может занять время при первом запуске - модель скачивается)")

    # отключаем mps перед загрузкой модели
    if hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available():
            logger.info("      MPS доступен, но используем CPU для стабильности")
            torch.backends.mps.is_available = lambda: False

    load_start = time.time()
    logger.info("      начало загрузки модели...")
    logger.info("      (если модель не скачана, это может занять 1-2 минуты)")

    model_path = model_name
    if os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1":
        local_snapshot = _resolve_local_snapshot(model_name)
        if local_snapshot:
            model_path = local_snapshot
            logger.info(f"      offline: используем локальный snapshot {local_snapshot}")

    try:
        model = SentenceTransformer(model_path, device="cpu")
        load_time = time.time() - load_start
        logger.info(f"      модель загружена за {load_time:.1f} сек")
    except Exception as e:
        load_time = time.time() - load_start
        logger.error(f"      ошибка при загрузке модели за {load_time:.1f} сек: {e}")
        raise

    if hasattr(model, "_modules"):
        for module in model._modules.values():
            if hasattr(module, "to"):
                try:
                    module.to("cpu")
                    if hasattr(module, "eval"):
                        module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
                except Exception:
                    pass

    if hasattr(model, "to"):
        try:
            model.to("cpu")
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
        except Exception:
            pass

    return model


def _get_cache_path(model_name: str, dataset_path: Path, num_items: int) -> Path:
    safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    dataset_name = dataset_path.stem
    try:
        stat = dataset_path.stat()
        dataset_sig = f"{stat.st_size}_{stat.st_mtime_ns}"
    except FileNotFoundError:
        dataset_sig = "missing"
    cache_key = f"{safe_model_name}_{dataset_name}_{num_items}_{dataset_sig}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
    cache_dir = PROJECT_ROOT / ".cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"embeddings_{safe_model_name}_{dataset_name}_{cache_hash}.npy"


def load_cached_embeddings(cache_path: Path) -> Optional[np.ndarray]:
    if cache_path.exists():
        try:
            logger.info(f"      загрузка эмбеддингов из кэша: {cache_path.name}")
            embeddings = np.load(cache_path)
            logger.info(f"      загружено {len(embeddings)} эмбеддингов из кэша")
            return embeddings
        except Exception as e:
            logger.warning(f"      ошибка при загрузке кэша: {e}, пересчитываем...")
    return None


def save_cached_embeddings(embeddings: np.ndarray, cache_path: Path) -> None:
    try:
        np.save(cache_path, embeddings)
        logger.info(f"      эмбеддинги сохранены в кэш: {cache_path.name}")
    except Exception as e:
        logger.warning(f"      не удалось сохранить кэш: {e}")


def embed_texts(
    model: SentenceTransformer,
    texts: Sequence[str],
    *,
    show_progress: bool = True,
) -> np.ndarray:
    # генерируем и нормализуем эмбеддинги для списка текстов
    import torch

    if hasattr(torch.backends, "mps"):
        try:
            torch.backends.mps.is_available = lambda: False
            if hasattr(torch.backends.mps, "_is_available"):
                torch.backends.mps._is_available = False
        except Exception:
            pass

    if hasattr(model, "_modules"):
        for module in model._modules.values():
            if hasattr(module, "to"):
                try:
                    module.to("cpu").eval()
                    for param in module.parameters():
                        param.requires_grad = False
                        if hasattr(param, "data"):
                            param.data = param.data.cpu()
                except Exception:
                    pass

    batch_size = 1 if len(texts) <= 1 else 32
    encode_params = {
        "show_progress_bar": show_progress if len(texts) > 1 else False,
        "convert_to_numpy": True,
        "batch_size": batch_size,
        "normalize_embeddings": False,
    }

    try:
        emb = model.encode(texts, device="cpu", **encode_params)
    except (TypeError, ValueError, AttributeError):
        try:
            emb = model.encode(texts, **encode_params)
        except Exception:
            emb = model.encode(texts, show_progress_bar=show_progress, convert_to_numpy=True)

    if hasattr(emb, "cpu"):
        emb = emb.cpu().numpy()
    elif hasattr(emb, "numpy"):
        emb = emb.numpy()
    elif not isinstance(emb, np.ndarray):
        emb = np.array(emb)

    if len(emb.shape) == 1:
        emb = emb.reshape(1, -1)
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    return emb / norms


def describe_embedding_model(model: SentenceTransformer, model_name: str) -> None:
    dim = model.get_sentence_embedding_dimension()
    max_seq = model.get_max_seq_length()
    print(f"\nЭМБЕДДИНГ-МОДЕЛЬ: {model_name} | dim={dim} | max_seq={max_seq}")


class SimpleVectorIndex:
    # храним матрицу эмбеддингов и соответствующие идентификаторы

    def __init__(self, embeddings: np.ndarray, ids: Sequence[int]):
        if len(embeddings) != len(ids):
            raise ValueError("Размерность эмбеддингов и списка id не совпадает")
        self.embeddings = embeddings
        self.ids = list(ids)
        self.id_to_pos = {item_id: pos for pos, item_id in enumerate(self.ids)}

    def search(self, query_vec: np.ndarray, top_k: int = 5):
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        scores = cosine_similarity(query_vec, self.embeddings)[0]
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.ids[i], float(scores[i])) for i in top_idx]

    def embedding_for(self, item_id: int) -> np.ndarray:
        pos = self.id_to_pos.get(item_id)
        if pos is None:
            raise KeyError(f"ID {item_id} не найден в индексе")
        return self.embeddings[pos]
