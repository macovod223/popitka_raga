# BM25 Baseline для поиска аналогов товаров
# BM25 (Best Matching 25) - классический алгоритм ранжирования на основе
# TF-IDF с улучшенной нормализацией. Используется как baseline для сравнения
# с dense retrieval (RAG) подходом
# 
# Преимущества BM25:
# - Прозрачность: легко понять, почему товар попал в результаты
# - Быстрота: не требует GPU, работает на CPU
# - Хорошо работает для точных совпадений терминов
# 
# Недостатки BM25:
# - Не учитывает семантическое сходство (синонимы, контекст)
# - Плохо работает с вариативностью формулировок
# - Требует ручной настройки параметров k1 и b

import argparse
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from common import evaluate_retrieval_metrics, print_analogs
from rag_prob import (
    DEFAULT_DATASET,
    DEFAULT_INDICES,
    DEFAULT_TOP_K,
    add_component_text,
    load_data,
    parse_example_indices,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("bm25_baseline")


@dataclass
class BM25Config:
    dataset: Path = DEFAULT_DATASET
    indices: Tuple[int, ...] = DEFAULT_INDICES
    top_k: int = DEFAULT_TOP_K
    allow_cross_category: bool = False
    output_file: Path = Path("comparison_results/bm25_results.txt")
    k1: float = 2.0  # параметр насыщения tf (увеличено для лучшего ранжирования)
    b: float = 0.75  # параметр нормализации длины документа


class BM25Index:
    # индекс для поиска на основе BM25
    # хранит токенизированные документы и предоставляет метод search()
    # для поиска наиболее релевантных документов по запросу

    def __init__(self, texts: List[str], ids: List[int], k1: float = 1.5, b: float = 0.75):
        # инициализация BM25 индекса
        # texts: список текстовых документов (component_text)
        # ids: список ID товаров, соответствующих текстам
        # k1: параметр насыщения TF (обычно 1.2-2.0)
        # b: параметр нормализации длины (обычно 0.75)
        if len(texts) != len(ids):
            raise ValueError("Количество текстов и ID должно совпадать")
        
        self.ids = ids
        self.k1 = k1
        self.b = b
        
        # токенизация: разбиваем тексты на слова (lowercase, без пунктуации)
        tokenized_texts = [self._tokenize(text) for text in texts]
        
        # создаём bm25 индекс
        self.bm25 = BM25Okapi(tokenized_texts, k1=k1, b=b)
        
        logger.info(f"Создан BM25 индекс для {len(texts)} документов")
        logger.info(f"Параметры: k1={k1}, b={b}")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # улучшенная токенизация с удалением стоп-слов
        if not text:
            return []
        
        import re
        # базовые английские стоп-слова
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'their', 'time', 'if',
            'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her',
            'would', 'make', 'like', 'into', 'him', 'has', 'two', 'more',
            'very', 'after', 'words', 'long', 'than', 'first', 'been', 'call',
            'who', 'oil', 'sit', 'now', 'find', 'down', 'day', 'did', 'get',
            'come', 'made', 'may', 'part'
        }
        
        # разбиваем по словам, приводим к lowercase
        tokens = re.findall(r'\b\w+\b', text.lower())
        # фильтруем: убираем стоп-слова, однобуквенные токены и слишком короткие
        filtered = [t for t in tokens if len(t) > 2 and t not in stop_words]
        return filtered

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        # поиск наиболее релевантных документов по запросу
        # query: текстовый запрос
        # top_k: количество результатов для возврата
        # возвращает: список кортежей (item_id, score), отсортированный по убыванию score
        if not query:
            return []
        
        # токенизируем запрос
        tokenized_query = self._tokenize(query)
        
        if not tokenized_query:
            return []
        
        # вычисляем bm25 скоры для всех документов
        scores = self.bm25.get_scores(tokenized_query)
        
        # улучшенная нормализация: используем сигмоиду для более плавного распределения
        scores_array = np.array(scores)
        if len(scores_array) > 0:
            # используем сигмоидальную нормализацию для лучшего распределения скоров
            max_score = np.max(scores_array)
            if max_score > 0:
                # нормализуем через сигмоиду: 1 / (1 + exp(-alpha * (score/max - beta)))
                # это даёт более плавное распределение в диапазоне [0, 1]
                alpha = 5.0  # крутизна сигмоиды
                beta = 0.5   # сдвиг
                normalized = scores_array / max_score
                scores_array = 1.0 / (1.0 + np.exp(-alpha * (normalized - beta)))
            else:
                scores_array = np.zeros_like(scores_array)
        
        # сортируем по убыванию и берём top_k
        top_indices = np.argsort(scores_array)[::-1][:top_k]
        
        # формируем результат: (item_id, normalized_score)
        results = [(self.ids[i], float(scores_array[i])) for i in top_indices if scores_array[i] > 0]
        
        return results


def find_analogs_bm25(
    row_idx: int,
    df: pd.DataFrame,
    index: BM25Index,
    top_k: int = 5,
    same_category: bool = True,
) -> Tuple[pd.Series, List[Tuple[int, float, pd.Series]]]:
    # поиск аналогов для товара по индексу строки с использованием BM25
    # row_idx: индекс строки в DataFrame
    # df: DataFrame с товарами
    # index: BM25 индекс
    # top_k: количество аналогов для возврата
    # same_category: фильтровать ли по категории
    # возвращает: кортеж (исходная строка, список аналогов (id, score, candidate_row))
    if row_idx < 0 or row_idx >= len(df):
        raise ValueError(f"Индекс {row_idx} вне диапазона [0, {len(df)-1}]")
    
    row = df.iloc[row_idx]
    query_cat = row.get("root_category", None)
    query_id = row.name
    query_text = row.get("component_text", "")
    
    if not query_text:
        logger.warning(f"Пустой component_text для строки {row_idx}")
        return row, []
    
    # поиск кандидатов (берём больше, чтобы после фильтрации осталось достаточно)
    results = index.search(query_text, top_k=top_k + 20)
    
    analogs = []
    for rec_id, score in results:
        # пропускаем сам товар-запрос
        if rec_id == query_id:
            continue
        
        try:
            candidate = df.loc[rec_id]
        except KeyError:
            # пропускаем, если запись не найдена
            continue
        
        # фильтр по категории
        if same_category and query_cat is not None:
            if candidate.get("root_category", None) != query_cat:
                continue
        
        analogs.append((rec_id, score, candidate))
        
        if len(analogs) >= top_k:
            break
    
    return row, analogs


# используем общие функции из common.py
evaluate_bm25_retrieval = lambda row, analogs, verbose=True: evaluate_retrieval_metrics(row, analogs, verbose=verbose, method_name="BM25")
print_analogs_bm25 = lambda row, analogs: print_analogs(row, analogs, method_name="BM25")


class BM25Pipeline:
    # пайплайн поиска аналогов с использованием BM25
    
    def __init__(self, config: BM25Config):
        self.config = config
        self.df: pd.DataFrame = pd.DataFrame()
        self.index: Optional[BM25Index] = None
        self.records: List[dict] = []
    
    def run(self) -> None:
        # запуск пайплайна BM25
        total_start = time.time()
        self._setup()
        
        if self.index is None:
            raise RuntimeError("Индекс не инициализирован")
        
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
        # инициализация: загрузка данных и построение индекса
        logger.info("инициализация bm25 пайплайна")
        
        t0 = time.time()
        logger.info("[1/2] загрузка данных...")
        self.df = add_component_text(load_data(self.config.dataset))
        
        if self.df.empty:
            raise ValueError("Датасет пустой")
        
        elapsed = time.time() - t0
        logger.info(f"      загружено {len(self.df)} товаров за {elapsed:.1f} сек")
        
        # построение bm25 индекса
        t1 = time.time()
        logger.info("[2/2] построение BM25 индекса...")
        texts = self.df["component_text"].tolist()
        ids = self.df.index.tolist()
        
        self.index = BM25Index(
            texts=texts,
            ids=ids,
            k1=self.config.k1,
            b=self.config.b,
        )
        elapsed = time.time() - t1
        total = time.time() - t0
        logger.info(f"      индекс построен за {elapsed:.1f} сек")
        logger.info(f"инициализация завершена за {total:.1f} сек")
        
        self._validate_indices()
    
    def _validate_indices(self) -> None:
        # проверка валидности индексов
        max_idx = len(self.df) - 1
        for idx in self.config.indices:
            if idx < 0 or idx > max_idx:
                raise ValueError(f"Индекс {idx} вне диапазона [0, {max_idx}]")
    
    def _process_index(self, idx: int) -> dict:
        # обработка одного запроса
        logger.info("\nПОИСК АНАЛОГОВ ДЛЯ СТРОКИ #%s (BM25)", idx)
        
        row, analogs = find_analogs_bm25(
            row_idx=idx,
            df=self.df,
            index=self.index,
            top_k=self.config.top_k,
            same_category=not self.config.allow_cross_category,
        )
        
        metrics = evaluate_bm25_retrieval(row, analogs, verbose=False)
        
        return {
            "query_index": idx,
            "title": row.get("title", ""),
            "category": row.get("root_category", ""),
            "brand": row.get("brand", ""),
            "analogs": analogs,
            "metrics": metrics,
        }
    
    def _render_record(self, record: dict) -> None:
        # вывод результатов на экран
        row = self.df.iloc[record["query_index"]]
        print_analogs_bm25(row, record["analogs"])
        self._print_metrics(record["metrics"])
    
    @staticmethod
    def _print_metrics(metrics: dict) -> None:
        # вывод метрик
        total = metrics.get("total_analogs", 0)
        print("ПРОКСИ-МЕТРИКИ (BM25):")
        print(f"- Средний BM25 скор: {metrics.get('avg_score', 0):.3f}")
        print(f"- Совпадение категории: {metrics.get('same_category_hits', 0)}/{total}")
    
    def _write_results(self) -> None:
        # сохранение результатов в файл
        if not self.records:
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
        lines.append(f"BM25 BASELINE RUN @ {timestamp}")
        lines.append(f"Параметры: k1={self.config.k1}, b={self.config.b}")
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
                for i, (candidate_id, score, cand) in enumerate(record["analogs"], start=1):
                    lines.append(
                        f"  {i}) [ID {candidate_id}] {cand.get('title', '')} "
                        f"| cat={cand.get('root_category', '')} | BM25 score={score:.3f}"
                    )
            else:
                lines.append("  (аналогов не найдено)")
        
        lines.append("\n")
        
        with output_path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    # парсинг аргументов командной строки
    parser = argparse.ArgumentParser(
        description="BM25 baseline для поиска аналогов товаров Best Buy."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Путь к CSV")
    parser.add_argument("--indices", default="0,10,20", help="Список индексов через запятую")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Сколько аналогов")
    parser.add_argument(
        "--allow-cross-category",
        action="store_true",
        help="Разрешить аналоги из других категорий",
    )
    parser.add_argument(
        "--output-file",
        default="comparison_results/bm25_results.txt",
        help="Путь к файлу результатов",
    )
    parser.add_argument(
        "--k1",
        type=float,
        default=2.0,
        help="Параметр BM25 k1 (насыщение TF)",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=0.75,
        help="Параметр BM25 b (нормализация длины)",
    )
    return parser.parse_args()


def build_config_from_cli(args: argparse.Namespace) -> BM25Config:
    # создание конфигурации из аргументов CLI
    indices = parse_example_indices(args.indices)
    if args.top_k <= 0:
        raise ValueError("top_k должен быть больше нуля")
    
    return BM25Config(
        dataset=Path(args.dataset),
        indices=tuple(indices),
        top_k=args.top_k,
        allow_cross_category=args.allow_cross_category,
        output_file=Path(args.output_file),
        k1=args.k1,
        b=args.b,
    )


def main() -> None:
    # главная функция
    args = parse_args()
    pipeline = BM25Pipeline(build_config_from_cli(args))
    pipeline.run()


if __name__ == "__main__":
    main()
