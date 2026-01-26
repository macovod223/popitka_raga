# Гибридный подход BM25 + RAG для поиска аналогов товаров
# Комбинирует лучшие стороны sparse retrieval (BM25) и dense retrieval (RAG)
# 
# Преимущества гибрида:
# - BM25 хорошо находит точные совпадения ключевых слов
# - RAG хорошо работает с семантическим сходством и синонимами
# - Комбинация даёт лучшее покрытие и качество результатов
# 
# Метод комбинирования: Reciprocal Rank Fusion (RRF)
# RRF - стандартный метод для комбинирования результатов из разных систем поиска

import argparse
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"\b\w+\b")
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "will", "with", "this", "but", "they", "have",
    "had", "what", "said", "each", "which", "their", "time", "if",
    "up", "out", "many", "then", "them", "these", "so", "some", "her",
    "would", "make", "like", "into", "him", "two", "more",
    "very", "after", "words", "long", "than", "first", "been", "call",
    "who", "oil", "sit", "now", "find", "down", "day", "did", "get",
    "come", "made", "may", "part",
}
from common import evaluate_retrieval_metrics, print_analogs
from llm_report import build_report_prompt, local_llm_generate, save_report, transformers_generate, strip_emojis, clean_llm_analysis
from rag_core.cli import parse_example_indices
from rag_core.components import add_component_text
from rag_core.constants import (
    DEFAULT_DATASET,
    DEFAULT_EMBED_MODEL,
    DEFAULT_INDICES,
    DEFAULT_TOP_K,
)
from rag_core.data import load_data
from rag_core.embeddings import (
    SimpleVectorIndex,
    _get_cache_path,
    embed_texts,
    load_cached_embeddings,
    load_embedding_model,
    save_cached_embeddings,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("hybrid_bm25_rag")


class BM25Index:
    # индекс для поиска на основе BM25

    def __init__(self, texts: List[str], ids: List[int], k1: float = 1.5, b: float = 0.75):
        if len(texts) != len(ids):
            raise ValueError("Количество текстов и ID должно совпадать")

        self.ids = ids
        self.k1 = k1
        self.b = b

        tokenized_texts = [self._tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(tokenized_texts, k1=k1, b=b)

        logger.info("Создан BM25 индекс для %s документов", len(texts))
        logger.info("Параметры: k1=%s, b=%s", k1, b)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # Токенизация: перевод в нижний регистр, поиск слов, фильтрация стоп-слов
        if not text:
            return []

        tokens = TOKEN_RE.findall(text.lower())
        return [t for t in tokens if len(t) > 2 and t not in STOP_WORDS]

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        # Поиск по BM25 с нормализацией скоров через сигмоиду
        if not query:
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        scores_array = np.array(scores)
        if len(scores_array) > 0:
            max_score = np.max(scores_array)
            if max_score > 0:
                alpha = 5.0
                beta = 0.5
                normalized = scores_array / max_score
                scores_array = 1.0 / (1.0 + np.exp(-alpha * (normalized - beta)))
            else:
                scores_array = np.zeros_like(scores_array)

        top_indices = np.argsort(scores_array)[::-1][:top_k]
        return [(self.ids[i], float(scores_array[i])) for i in top_indices if scores_array[i] > 0]


@dataclass
class HybridConfig:
    dataset: Path = DEFAULT_DATASET
    embedding_model: str = DEFAULT_EMBED_MODEL
    indices: Tuple[int, ...] = DEFAULT_INDICES
    top_k: int = DEFAULT_TOP_K
    allow_cross_category: bool = False
    output_file: Optional[Path] = None
    # Параметры BM25
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    # Параметры гибрида
    bm25_weight: float = 0.3  # Вес BM25 в финальном скоре
    rag_weight: float = 0.7  # Вес RAG в финальном скоре (больше, т.к. показывает лучшие результаты)
    use_rrf: bool = True  # Использовать Reciprocal Rank Fusion вместо взвешенной суммы
    rrf_k: int = 20  # Параметр k для RRF (оптимизировано для лучших результатов)
    # Параметры генерации отчёта
    gen_report: bool = False  # Генерировать ли LLM-отчёт
    llm_model: str = "mistral"  # Модель локальной LLM (через HTTP)
    llm_models: Optional[str] = None  # Список моделей через запятую для последовательного прогона
    llm_url: str = "http://localhost:11434/api/generate"  # URL локального LLM API
    llm_backend: str = "transformers"  # transformers|http
    hf_allow_download: bool = True  # Разрешить transformers скачивать модели (иначе только локальный кеш)
    report_dir: Path = Path("reports")  # Директория для отчётов
    report_max_chars: int = 800  # Максимальная длина описаний в prompt (уменьшено для стабильности на малых моделях)


def reciprocal_rank_fusion(
    bm25_results: List[Tuple[int, float]],
    rag_results: List[Tuple[int, float]],
    k: int = 60,
    top_k: int = 5,
) -> List[Tuple[int, float]]:
    """
    Reciprocal Rank Fusion (RRF) - комбинирует результаты из двух систем поиска.
    
    RRF скор = sum(1 / (k + rank)) для каждого результата из всех систем
    
    Args:
        bm25_results: результаты от BM25 [(item_id, score), ...]
        rag_results: результаты от RAG [(item_id, score), ...]
        k: параметр RRF (обычно 60)
        top_k: количество результатов для возврата
    
    Returns:
        список кортежей (item_id, rrf_score), отсортированный по убыванию
    """
    # создаём словари для быстрого доступа к рангам
    bm25_ranks = {item_id: rank + 1 for rank, (item_id, _) in enumerate(bm25_results)}
    rag_ranks = {item_id: rank + 1 for rank, (item_id, _) in enumerate(rag_results)}
    
    # собираем все уникальные item_id
    all_ids = set(bm25_ranks.keys()) | set(rag_ranks.keys())
    
    # вычисляем RRF скоры
    rrf_scores = {}
    for item_id in all_ids:
        rrf_score = 0.0
        
        # добавляем вклад от BM25
        if item_id in bm25_ranks:
            rrf_score += 1.0 / (k + bm25_ranks[item_id])
        
        # добавляем вклад от RAG
        if item_id in rag_ranks:
            rrf_score += 1.0 / (k + rag_ranks[item_id])
        
        rrf_scores[item_id] = rrf_score
    
    # сортируем по убыванию RRF скора
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # нормализуем RRF скоры в диапазон [0, 1] для сопоставимости с другими подходами
    if sorted_results:
        max_rrf = sorted_results[0][1]
        min_rrf = sorted_results[-1][1]
        if max_rrf > min_rrf:
            normalized_results = [
                (item_id, (score - min_rrf) / (max_rrf - min_rrf))
                for item_id, score in sorted_results[:top_k]
            ]
        else:
            # если все скоры одинаковые, нормализуем к 1.0
            normalized_results = [
                (item_id, 1.0 if max_rrf > 0 else 0.0)
                for item_id, score in sorted_results[:top_k]
            ]
        return normalized_results
    else:
        return []


def weighted_score_fusion(
    bm25_results: List[Tuple[int, float]],
    rag_results: List[Tuple[int, float]],
    bm25_weight: float = 0.3,
    rag_weight: float = 0.7,
    top_k: int = 5,
) -> List[Tuple[int, float]]:
    """
    Взвешенная сумма нормализованных скоров из BM25 и RAG.
    
    Args:
        bm25_results: результаты от BM25 [(item_id, score), ...]
        rag_results: результаты от RAG [(item_id, score), ...]
        bm25_weight: вес BM25 скора
        rag_weight: вес RAG скора
        top_k: количество результатов для возврата
    
    Returns:
        список кортежей (item_id, combined_score), отсортированный по убыванию
    """
    # нормализуем скоры BM25 в диапазон [0, 1]
    if bm25_results:
        bm25_scores_dict = {item_id: score for item_id, score in bm25_results}
        bm25_max = max(score for _, score in bm25_results)
        bm25_min = min(score for _, score in bm25_results)
        if bm25_max > bm25_min:
            bm25_scores = {
                item_id: (score - bm25_min) / (bm25_max - bm25_min)
                for item_id, score in bm25_scores_dict.items()
            }
        else:
            bm25_scores = {item_id: 1.0 if score > 0 else 0.0 for item_id, score in bm25_scores_dict.items()}
    else:
        bm25_scores = {}
    
    # нормализуем скоры RAG в диапазон [0, 1]
    if rag_results:
        rag_scores_dict = {item_id: score for item_id, score in rag_results}
        rag_max = max(score for _, score in rag_results)
        rag_min = min(score for _, score in rag_results)
        if rag_max > rag_min:
            rag_scores = {
                item_id: (score - rag_min) / (rag_max - rag_min)
                for item_id, score in rag_scores_dict.items()
            }
        else:
            rag_scores = {item_id: 1.0 if score > 0 else 0.0 for item_id, score in rag_scores_dict.items()}
    else:
        rag_scores = {}
    
    # собираем все уникальные item_id
    all_ids = set(bm25_scores.keys()) | set(rag_scores.keys())
    
    # вычисляем комбинированные скоры с нормализованными значениями
    combined_scores = {}
    for item_id in all_ids:
        bm25_score = bm25_scores.get(item_id, 0.0)
        rag_score = rag_scores.get(item_id, 0.0)
        
        # взвешенная сумма нормализованных скоров
        combined_score = bm25_weight * bm25_score + rag_weight * rag_score
        combined_scores[item_id] = combined_score
    
    # сортируем по убыванию комбинированного скора
    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    return [(item_id, score) for item_id, score in sorted_results[:top_k]]


def find_analogs_hybrid(
    row_idx: int,
    df: pd.DataFrame,
    bm25_index: BM25Index,
    rag_index: SimpleVectorIndex,
    embedder: SentenceTransformer,
    top_k: int = 5,
    same_category: bool = True,
    use_rrf: bool = True,
    rrf_k: int = 60,
    bm25_weight: float = 0.4,
    rag_weight: float = 0.6,
) -> Tuple[pd.Series, List[Tuple[int, float, pd.Series]]]:
    """
    Поиск аналогов с использованием гибридного подхода BM25 + RAG.
    
    Args:
        row_idx: индекс строки в DataFrame
        df: DataFrame с товарами
        bm25_index: BM25 индекс
        rag_index: RAG индекс (SimpleVectorIndex)
        embedder: модель эмбеддингов
        top_k: количество аналогов
        same_category: фильтровать ли по категории
        use_rrf: использовать RRF или взвешенную сумму
        rrf_k: параметр k для RRF
        bm25_weight: вес BM25 (если не используется RRF)
        rag_weight: вес RAG (если не используется RRF)
    
    Returns:
        кортеж (исходная строка, список аналогов)
    """
    if row_idx < 0 or row_idx >= len(df):
        raise ValueError(f"Индекс {row_idx} вне диапазона [0, {len(df)-1}]")
    
    row = df.iloc[row_idx]
    query_cat = row.get("root_category", None)
    query_id = row.name
    query_text = row.get("component_text", "")
    
    if not query_text:
        logger.warning(f"Пустой component_text для строки {row_idx}")
        return row, []
    
    # получаем результаты от BM25 (берём больше кандидатов для лучшего покрытия)
    # BM25 хорошо находит точные совпадения ключевых слов
    bm25_results = bm25_index.search(query_text, top_k=max(top_k * 15, 75))
    
    # получаем результаты от RAG
    # RAG хорошо работает с семантическим сходством
    try:
        query_emb = rag_index.embedding_for(query_id)
    except KeyError:
        # fallback: пересчитываем эмбеддинг
        query_emb = embed_texts(embedder, [query_text], show_progress=False)[0]
    
    rag_results = rag_index.search(query_emb, top_k=max(top_k * 15, 75))
    
    # комбинируем результаты - используем combined_fusion для лучших результатов
    if use_rrf:
        # Используем комбинированный подход: RRF + взвешенная сумма
        # Сначала применяем RRF для получения начального ранжирования
        rrf_results = reciprocal_rank_fusion(
            bm25_results, rag_results, k=rrf_k, top_k=max(top_k * 5, 30)
        )
        # Также получаем результаты от взвешенной суммы
        weighted_results = weighted_score_fusion(
            bm25_results, rag_results, 
            bm25_weight=bm25_weight, rag_weight=rag_weight,
            top_k=max(top_k * 5, 30)
        )
        # Комбинируем оба метода через RRF для финального ранжирования
        combined_results = reciprocal_rank_fusion(
            rrf_results, weighted_results, k=rrf_k, top_k=max(top_k * 3, 20)
        )
    else:
        combined_results = weighted_score_fusion(
            bm25_results, rag_results, 
            bm25_weight=bm25_weight, rag_weight=rag_weight,
            top_k=max(top_k * 5, 30)
        )
    
    # создаём словари для быстрого доступа к скорам
    bm25_score_map = {item_id: score for item_id, score in bm25_results}
    rag_score_map = {item_id: score for item_id, score in rag_results}
    
    # фильтруем по категории и формируем финальный список
    analogs = []
    for rec_id, score in combined_results:
        if rec_id == query_id:
            continue
        
        try:
            candidate = df.loc[rec_id]
        except KeyError:
            continue
        
        if same_category and query_cat is not None:
            if candidate.get("root_category", None) != query_cat:
                continue
        
        # получаем отдельные скоры для отчёта
        bm25_score = bm25_score_map.get(rec_id)
        dense_score = rag_score_map.get(rec_id)
        
        analogs.append((rec_id, score, candidate, bm25_score, dense_score))
        
        if len(analogs) >= top_k:
            break
    
    return row, analogs


# используем общие функции из common.py с адаптацией для расширенного формата
def evaluate_hybrid_retrieval(row, analogs, verbose=True):
    """Обёртка для evaluate_retrieval_metrics с поддержкой расширенного формата."""
    # Преобразуем расширенный формат (id, score, cand, bm25_score, dense_score) 
    # в стандартный формат (id, score, cand)
    standard_analogs = [
        (analog[0], analog[1], analog[2]) for analog in analogs
    ]
    return evaluate_retrieval_metrics(
        row, standard_analogs, verbose=verbose, method_name="Hybrid BM25+RAG"
    )


def print_analogs_hybrid(row, analogs):
    """Обёртка для print_analogs с поддержкой расширенного формата."""
    # Преобразуем расширенный формат в стандартный
    standard_analogs = [
        (analog[0], analog[1], analog[2]) for analog in analogs
    ]
    print_analogs(row, standard_analogs, method_name="Hybrid BM25+RAG")


class HybridPipeline:
    """Пайплайн поиска аналогов с использованием гибридного подхода BM25 + RAG."""
    
    def __init__(self, config: HybridConfig):
        self.config = config
        self.df: pd.DataFrame = pd.DataFrame()
        self.embedder: Optional[SentenceTransformer] = None
        self.bm25_index: Optional[BM25Index] = None
        self.rag_index: Optional[SimpleVectorIndex] = None
        self.records: List[dict] = []
    
    def run(self) -> None:
        """Запуск пайплайна гибридного поиска."""
        total_start = time.time()
        self._setup()
        
        if self.bm25_index is None or self.rag_index is None or self.embedder is None:
            raise RuntimeError("Пайплайн не инициализирован")
        
        logger.info("")
        logger.info(f"поиск аналогов для {len(self.config.indices)} товаров")
        
        self.records = []
        for i, idx in enumerate(self.config.indices, 1):
            logger.info(f"\n[{i}/{len(self.config.indices)}] обработка товара с индексом {idx}...")
            t_start = time.time()
            record = self._process_index(idx)
            elapsed = time.time() - t_start
            self.records.append(record)
            logger.info(f"      найдено {len(record['analogs'])} аналогов за {elapsed:.2f} сек")
            self._render_record(record)
        
        logger.info("")
        t_write = time.time()
        logger.info("сохранение результатов...")
        self._write_results()
        write_time = time.time() - t_write
        total_time = time.time() - total_start
        logger.info(f"результаты сохранены за {write_time:.1f} сек")
        logger.info(f"общее время выполнения: {total_time:.1f} сек")
    
    def _setup(self) -> None:
        """Инициализация: загрузка данных и построение индексов."""
        logger.info("инициализация гибридного пайплайна (BM25 + RAG)")
        
        t0 = time.time()
        logger.info("[1/4] загрузка данных...")
        self.df = add_component_text(load_data(self.config.dataset))
        
        if self.df.empty:
            raise ValueError("Датасет пустой")
        
        elapsed = time.time() - t0
        logger.info(f"      загружено {len(self.df)} товаров за {elapsed:.1f} сек")
        
        # построение BM25 индекса
        t1 = time.time()
        logger.info("[2/4] построение BM25 индекса...")
        texts = self.df["component_text"].tolist()
        ids = self.df.index.tolist()
        
        self.bm25_index = BM25Index(
            texts=texts,
            ids=ids,
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
        )
        elapsed = time.time() - t1
        logger.info(f"      BM25 индекс построен за {elapsed:.1f} сек")
        
        # загрузка модели эмбеддингов
        t2 = time.time()
        logger.info(f"[3/4] загрузка модели эмбеддингов '{self.config.embedding_model}'...")
        if not self.config.hf_allow_download:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self.embedder = load_embedding_model(self.config.embedding_model)
        elapsed = time.time() - t2
        logger.info(f"      модель загружена за {elapsed:.1f} сек")
        
        # построение эмбеддингов и RAG индекса
        t3 = time.time()
        logger.info(f"[4/4] генерация эмбеддингов для {len(self.df)} товаров...")
        logger.info("      (это может занять некоторое время)")

        cache_path = _get_cache_path(self.config.embedding_model, self.config.dataset, len(self.df))
        embeddings = load_cached_embeddings(cache_path)
        if embeddings is None:
            embeddings = embed_texts(self.embedder, texts)
            elapsed = time.time() - t3
            logger.info(
                f"      эмбеддинги сгенерированы за {elapsed:.1f} сек ({elapsed/len(self.df)*1000:.1f} мс/товар)"
            )
            save_cached_embeddings(embeddings, cache_path)
        else:
            elapsed = time.time() - t3
            logger.info(f"      эмбеддинги загружены из кэша за {elapsed:.1f} сек")
        
        self.rag_index = SimpleVectorIndex(embeddings, ids)
        
        self._validate_indices()
        total = time.time() - t0
        logger.info(f"инициализация завершена за {total:.1f} сек")
    
    def _validate_indices(self) -> None:
        """Проверка валидности индексов."""
        max_idx = len(self.df) - 1
        for idx in self.config.indices:
            if idx < 0 or idx > max_idx:
                raise ValueError(f"Индекс {idx} вне диапазона [0, {max_idx}]")
    
    def _process_index(self, idx: int) -> dict:
        """Обработка одного запроса."""
        logger.info("\nПОИСК АНАЛОГОВ ДЛЯ СТРОКИ #%s (Hybrid BM25+RAG)", idx)
        
        row, analogs = find_analogs_hybrid(
            row_idx=idx,
            df=self.df,
            bm25_index=self.bm25_index,
            rag_index=self.rag_index,
            embedder=self.embedder,
            top_k=self.config.top_k,
            same_category=not self.config.allow_cross_category,
            use_rrf=self.config.use_rrf,
            rrf_k=self.config.rrf_k,
            bm25_weight=self.config.bm25_weight,
            rag_weight=self.config.rag_weight,
        )
        
        metrics = evaluate_hybrid_retrieval(row, analogs, verbose=False)
        
        # Генерация LLM-отчёта, если включено
        if self.config.gen_report and analogs:
            self._generate_report(idx, row, analogs)
        
        return {
            "query_index": idx,
            "title": row.get("title", ""),
            "category": row.get("root_category", ""),
            "brand": row.get("brand", ""),
            "analogs": analogs,
            "metrics": metrics,
        }
    
    def _generate_report(self, idx: int, query_row: pd.Series, analogs: List[tuple]) -> None:
        """Генерация LLM-отчёта для запроса."""
        try:
            # Формируем словарь запроса
            query_item = {
                "item_id": idx,
                "query_index": idx,
                "title": query_row.get("title", ""),
                "root_category": query_row.get("root_category", ""),
                "category": query_row.get("root_category", ""),
                "brand": query_row.get("brand", ""),
                "product_description": query_row.get("product_description", ""),
                "features_summary": query_row.get("features_summary", ""),
                "product_specifications": query_row.get("product_specifications", ""),
            }
            
            # prompt будем строить под конкретную модель (язык/формат могут отличаться)

            # Статическая часть отчёта (детерминированная): заголовок + таблица аналогов
            header_lines = []
            header_lines.append("ЗАГОЛОВОК")
            header_lines.append(f"Запрос: {query_item.get('title','')}")
            header_lines.append(f"Категория: {query_item.get('root_category','')}")
            header_lines.append("")
            header_lines.append("ТАБЛИЦА АНАЛОГОВ")
            header_lines.append("ID\tНазвание\tScore")
            for r, a in enumerate(analogs, start=1):
                cand_id, hscore, cand = a[0], a[1], a[2]
                title = (cand.get("title", "") or "").replace("\t", " ")[:100]
                header_lines.append(f"{cand_id}\t{title}\t{hscore:.2f}")
            header_text = "\n".join(header_lines).strip()
            
            # Если задан список моделей — прогоняем все и пишем сравнение
            models = []
            if self.config.llm_models:
                models = [m.strip() for m in self.config.llm_models.split(",") if m.strip()]
            else:
                models = [self.config.llm_model]

            for model in models:
                # Формируем prompt на русском для всех моделей.
                prompt_lang = "ru"
                prompt = build_report_prompt(
                    query_item=query_item,
                    analog_items=analogs,
                    run_meta=None,
                    max_chars=self.config.report_max_chars,
                    adaptive_limit=True,  # Автоматически рассчитывает ограничения
                    language=prompt_lang,
                    num_ctx=self.config.report_max_chars * 2, # Увеличиваем контекст
                )
                if self.config.llm_backend == "transformers":
                    # Для малых моделей на CPU используем низкую температуру и сэмплирование для стабильности
                    report_text = transformers_generate(
                        prompt=prompt,
                        model=model,
                        temperature=0.1,
                        top_p=0.7,
                        num_predict=150,
                        num_ctx=2048,
                        allow_download=self.config.hf_allow_download,
                        run_id=f"post-fix-final-{model}",
                    )
                else:
                    report_text = local_llm_generate(
                        prompt=prompt,
                        model=model,
                        llm_url=self.config.llm_url,
                        timeout=120,
                        run_id=f"post-fix-1-{model}",
                    )

                if not report_text:
                    logger.warning(f"Не удалось сгенерировать отчёт для запроса #{idx} (модель: {model})")
                    continue

                # Собираем финальный отчёт: детерминированный header+table + LLM-часть (3-6)

                # Собираем финальный отчёт: детерминированный header+table + LLM-часть (3-6)
                report_text = clean_llm_analysis(report_text).strip()
                if not report_text.upper().startswith("КРАТКИЙ ВЫВОД"):
                    report_text = "КРАТКИЙ ВЫВОД: " + report_text

                final_report = header_text + "\n\n" + report_text
                report_path = self.config.report_dir / f"query_{idx}_report_{model}.txt"
                save_report(final_report, report_path)
                logger.info(f"LLM-отчёт сохранён: {report_path}")
                
        except Exception as e:
            logger.error(f"Ошибка при генерации отчёта для запроса #{idx}: {e}", exc_info=True)

    def _render_record(self, record: dict) -> None:
        """Вывод результатов на экран."""
        row = self.df.iloc[record["query_index"]]
        print_analogs_hybrid(row, record["analogs"])
        self._print_metrics(record["metrics"])
    
    @staticmethod
    def _print_metrics(metrics: dict) -> None:
        """Вывод метрик."""
        total = metrics.get("total_analogs", 0)
        print("ПРОКСИ-МЕТРИКИ (Hybrid BM25+RAG):")
        print(f"- Средний скор: {metrics.get('avg_score', 0):.3f}")
        print(f"- Совпадение категории: {metrics.get('same_category_hits', 0)}/{total}")
    
    def _write_results(self) -> None:
        """Сохранение результатов в файл."""
        if not self.records or not self.config.output_file:
            return
        
        output_path = self.config.output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        total_queries = len(self.records)
        total_analogs = sum(r["metrics"]["total_analogs"] for r in self.records)
        avg_scores = [
            r["metrics"]["avg_score"] for r in self.records if r["metrics"]["total_analogs"] > 0
        ]
        mean_avg_score = sum(avg_scores) / len(avg_scores) if avg_scores else 0.0
        total_cat_hits = sum(r["metrics"]["same_category_hits"] for r in self.records)
        cat_ratio = total_cat_hits / total_analogs if total_analogs else 0.0
        
        lines: List[str] = []
        lines.append(f"HYBRID BM25+RAG RUN @ {timestamp}")
        fusion_method = "RRF" if self.config.use_rrf else "Weighted Sum"
        lines.append(f"Метод комбинирования: {fusion_method}")
        if self.config.use_rrf:
            lines.append(f"RRF параметр k: {self.config.rrf_k}")
        else:
            lines.append(f"Веса: BM25={self.config.bm25_weight}, RAG={self.config.rag_weight}")
        lines.append(
            f"Запросов: {total_queries} | Всего аналогов: {total_analogs} | "
            f"Средний скор: {mean_avg_score:.3f}"
        )
        lines.append(
            f"Совпадение категорий: {total_cat_hits}/{total_analogs} ({cat_ratio:.1%})"
        )
        
        for record in self.records:
            lines.append("\n---")
            lines.append(
                f"Запрос #{record['query_index']} | {record['title']} ({record['category']})"
            )
            metrics = record["metrics"]
            lines.append(
                f"Метрики: avg_score={metrics['avg_score']:.3f}, "
                f"cat {metrics['same_category_hits']}/{metrics['total_analogs']}"
            )
            lines.append("Аналоги:")
            if record["analogs"]:
                for i, analog_data in enumerate(record["analogs"], start=1):
                    if len(analog_data) >= 3:
                        candidate_id, score, cand = analog_data[0], analog_data[1], analog_data[2]
                        bm25_score = analog_data[3] if len(analog_data) > 3 else None
                        dense_score = analog_data[4] if len(analog_data) > 4 else None
                    else:
                        # Fallback для старого формата
                        candidate_id, score = analog_data[0], analog_data[1]
                        cand = analog_data[2] if len(analog_data) > 2 else {}
                        bm25_score = None
                        dense_score = None
                    
                    score_str = f"Hybrid score={score:.3f}"
                    if bm25_score is not None:
                        score_str += f", BM25={bm25_score:.3f}"
                    if dense_score is not None:
                        score_str += f", Dense={dense_score:.3f}"
                    
                    lines.append(
                        f"  {i}) [ID {candidate_id}] {cand.get('title', '')} "
                        f"| cat={cand.get('root_category', '')} | {score_str}"
                    )
            else:
                lines.append("  (аналогов не найдено)")
        
        lines.append("\n")
        
        with output_path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Гибридный подход BM25+RAG для поиска аналогов товаров Best Buy."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Путь к CSV")
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBED_MODEL,
        help="sentence-transformer модель",
    )
    parser.add_argument("--indices", default="0,10,20", help="Список индексов через запятую")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Сколько аналогов")
    parser.add_argument(
        "--allow-cross-category",
        action="store_true",
        help="Разрешить аналоги из других категорий",
    )
    parser.add_argument(
        "--output-file",
        default="",
        help="Путь к файлу результатов (если не задан — файл не создаётся)",
    )
    parser.add_argument(
        "--bm25-k1",
        type=float,
        default=1.5,
        help="Параметр BM25 k1 (насыщение TF)",
    )
    parser.add_argument(
        "--bm25-b",
        type=float,
        default=0.75,
        help="Параметр BM25 b (нормализация длины)",
    )
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=0.3,
        help="Вес BM25 в комбинированном скоре (если не используется RRF)",
    )
    parser.add_argument(
        "--rag-weight",
        type=float,
        default=0.7,
        help="Вес RAG в комбинированном скоре (если не используется RRF)",
    )
    parser.add_argument(
        "--use-rrf",
        action="store_true",
        default=True,
        help="Использовать Reciprocal Rank Fusion (по умолчанию: True)",
    )
    parser.add_argument(
        "--no-rrf",
        action="store_true",
        help="Не использовать RRF, использовать взвешенную сумму",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=20,
        help="Параметр k для RRF (меньше = более агрессивное ранжирование, обычно 20-60)",
    )
    parser.add_argument(
        "--gen-report",
        action="store_true",
        help="Генерировать структурированный отчёт через LLM (transformers/http)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="mistral",
        help="Название модели для генерации отчёта (по умолчанию: mistral)",
    )
    parser.add_argument(
        "--llm-models",
        type=str,
        default=None,
        help="Список моделей через запятую для последовательного прогона (например: qwen,falcon).",
    )
    parser.add_argument(
        "--llm-url",
        type=str,
        default="http://localhost:11434/api/generate",
        help="URL локального LLM API (по умолчанию: http://localhost:11434/api/generate)",
    )
    parser.add_argument(
        "--llm-backend",
        type=str,
        default="transformers",
        choices=["transformers", "http"],
        help="Backend для генерации отчёта: transformers (без сервера) или http (через локальный API)",
    )
    hf_download_group = parser.add_mutually_exclusive_group()
    hf_download_group.add_argument(
        "--hf-allow-download",
        dest="hf_allow_download",
        action="store_true",
        help="Разрешить transformers скачивать модели (по умолчанию: включено)",
    )
    hf_download_group.add_argument(
        "--hf-no-download",
        dest="hf_allow_download",
        action="store_false",
        help="Запретить transformers скачивать модели (использовать только локальный кеш)",
    )
    parser.set_defaults(hf_allow_download=True)
    parser.add_argument(
        "--report-dir",
        type=str,
        default="reports",
        help="Директория для сохранения отчётов (по умолчанию: reports)",
    )
    parser.add_argument(
        "--report-max-chars",
        type=int,
        default=1200,
        help="Максимальная длина описаний товаров в prompt (по умолчанию: 1200)",
    )
    return parser.parse_args()


def build_config_from_cli(args: argparse.Namespace) -> HybridConfig:
    """Создание конфигурации из аргументов CLI."""
    indices = parse_example_indices(args.indices)
    if args.top_k <= 0:
        raise ValueError("top_k должен быть больше нуля")
    
    use_rrf = args.use_rrf and not args.no_rrf
    
    output_file = Path(args.output_file) if args.output_file else None
    return HybridConfig(
        dataset=Path(args.dataset),
        embedding_model=args.embedding_model,
        indices=tuple(indices),
        top_k=args.top_k,
        allow_cross_category=args.allow_cross_category,
        output_file=output_file,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        bm25_weight=args.bm25_weight,
        rag_weight=args.rag_weight,
        use_rrf=use_rrf,
        rrf_k=args.rrf_k,
        gen_report=args.gen_report,
        llm_model=args.llm_model,
        llm_models=args.llm_models,
        llm_url=args.llm_url,
        llm_backend=args.llm_backend,
        hf_allow_download=args.hf_allow_download,
        report_dir=Path(args.report_dir),
        report_max_chars=args.report_max_chars,
    )


def main() -> None:
    """Главная функция."""
    args = parse_args()
    pipeline = HybridPipeline(build_config_from_cli(args))
    pipeline.run()


if __name__ == "__main__":
    main()