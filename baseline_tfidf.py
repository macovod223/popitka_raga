# TF-IDF Baseline для поиска аналогов товаров
# TF-IDF (Term Frequency-Inverse Document Frequency) - классический метод
# векторизации текстов, который учитывает важность терминов в документе
# относительно всей коллекции
# 
# Преимущества TF-IDF:
# - Простота и прозрачность
# - Быстрота вычислений
# - Хорошо работает для точных совпадений ключевых слов
# - Не требует GPU
# 
# Недостатки TF-IDF:
# - Не учитывает семантическое сходство
# - Плохо работает с синонимами
# - Игнорирует порядок слов
# - Требует ручной настройки параметров (max_features, ngram_range)

import argparse
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

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
logger = logging.getLogger("tfidf_baseline")


@dataclass
class TFIDFConfig:
    dataset: Path = DEFAULT_DATASET
    indices: Tuple[int, ...] = DEFAULT_INDICES
    top_k: int = DEFAULT_TOP_K
    allow_cross_category: bool = False
    output_file: Path = Path("comparison_results/tfidf_results.txt")
    max_features: int = 15000  # Увеличено для лучшего покрытия признаков
    ngram_range: Tuple[int, int] = (1, 2)  # Униграммы и биграммы
    min_df: int = 2  # Минимальная частота документа для термина
    max_df: float = 0.95  # Максимальная частота документа
    sublinear_tf: bool = False  # Отключено для сохранения исходного поведения


class TFIDFIndex:
    # индекс для поиска на основе TF-IDF
    # использует sklearn TfidfVectorizer для векторизации текстов и
    # косинусное сходство для поиска похожих документов

    def __init__(
        self,
        texts: List[str],
        ids: List[int],
        max_features: int = 15000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = False,
    ):
        # инициализация TF-IDF индекса
        # texts: список текстовых документов
        # ids: список ID товаров
        # max_features: максимальное количество признаков (размерность векторов)
        # ngram_range: диапазон n-грамм (1, 2) = униграммы + биграммы
        # min_df: минимальная частота документа для термина
        # max_df: максимальная частота документа (0.95 = игнорируем слова в >95% документов)
        if len(texts) != len(ids):
            raise ValueError("Количество текстов и ID должно совпадать")
        
        self.ids = ids
        
        logger.info(f"Векторизация {len(texts)} документов с помощью TF-IDF...")
        logger.info(
            f"Параметры: max_features={max_features}, ngram_range={ngram_range}, "
            f"min_df={min_df}, max_df={max_df}"
        )
        
        # создаём улучшенный tf-idf векторизатор
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            lowercase=True,
            token_pattern=r'\b\w+\b',
            sublinear_tf=sublinear_tf,  # Логарифмическое масштабирование TF
            smooth_idf=True,  # Сглаживание IDF
            norm='l2',  # L2 нормализация
        )
        
        # векторизуем все документы
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # нормализуем векторы для косинусного сходства
        # (tfidfvectorizer уже возвращает нормализованные векторы, но для надёжности)
        self.tfidf_matrix = normalize(self.tfidf_matrix, norm='l2')
        
        logger.info(
            f"Создан TF-IDF индекс: {self.tfidf_matrix.shape[0]} документов, "
            f"{self.tfidf_matrix.shape[1]} признаков"
        )

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        # поиск наиболее релевантных документов по запросу
        # query: текстовый запрос
        # top_k: количество результатов
        # возвращает: список кортежей (item_id, cosine_score), отсортированный по убыванию
        if not query:
            return []
        
        # векторизуем запрос
        query_vector = self.vectorizer.transform([query])
        query_vector = normalize(query_vector, norm='l2')
        
        # вычисляем косинусное сходство со всеми документами
        # (для sparse матриц используем специальный метод)
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # сортируем по убыванию и берём top_k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # формируем результат
        results = [
            (self.ids[i], float(similarities[i]))
            for i in top_indices
            if similarities[i] > 0
        ]
        
        return results


def find_analogs_tfidf(
    row_idx: int,
    df: pd.DataFrame,
    index: TFIDFIndex,
    top_k: int = 5,
    same_category: bool = True,
) -> Tuple[pd.Series, List[Tuple[int, float, pd.Series]]]:
    # поиск аналогов для товара с использованием TF-IDF
    # row_idx: индекс строки в DataFrame
    # df: DataFrame с товарами
    # index: TFIDF индекс
    # top_k: количество аналогов
    # same_category: фильтровать ли по категории
    # возвращает: кортеж (исходная строка, список аналогов)
    if row_idx < 0 or row_idx >= len(df):
        raise ValueError(f"Индекс {row_idx} вне диапазона [0, {len(df)-1}]")
    
    row = df.iloc[row_idx]
    query_cat = row.get("root_category", None)
    query_id = row.name
    query_text = row.get("component_text", "")
    
    if not query_text:
        logger.warning(f"Пустой component_text для строки {row_idx}")
        return row, []
    
    # поиск кандидатов
    results = index.search(query_text, top_k=top_k + 20)
    
    analogs = []
    for rec_id, score in results:
        if rec_id == query_id:
            continue
        
        try:
            candidate = df.loc[rec_id]
        except KeyError:
            # пропускаем, если запись не найдена
            continue
        
        if same_category and query_cat is not None:
            if candidate.get("root_category", None) != query_cat:
                continue
        
        analogs.append((rec_id, score, candidate))
        
        if len(analogs) >= top_k:
            break
    
    return row, analogs


# используем общие функции из common.py
evaluate_tfidf_retrieval = lambda row, analogs, verbose=True: evaluate_retrieval_metrics(row, analogs, verbose=verbose, method_name="TF-IDF")
print_analogs_tfidf = lambda row, analogs: print_analogs(row, analogs, method_name="TF-IDF")


class TFIDFPipeline:
    # пайплайн поиска аналогов с использованием TF-IDF
    
    def __init__(self, config: TFIDFConfig):
        self.config = config
        self.df: pd.DataFrame = pd.DataFrame()
        self.index: Optional[TFIDFIndex] = None
        self.records: List[dict] = []
    
    def run(self) -> None:
        # запуск пайплайна TF-IDF
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
        logger.info("инициализация tf-idf пайплайна")
        
        t0 = time.time()
        logger.info("[1/2] загрузка данных...")
        self.df = add_component_text(load_data(self.config.dataset))
        
        if self.df.empty:
            raise ValueError("Датасет пустой")
        
        elapsed = time.time() - t0
        logger.info(f"      загружено {len(self.df)} товаров за {elapsed:.1f} сек")
        
        # построение tf-idf индекса
        t1 = time.time()
        logger.info("[2/2] построение TF-IDF индекса...")
        texts = self.df["component_text"].tolist()
        ids = self.df.index.tolist()
        
        self.index = TFIDFIndex(
            texts=texts,
            ids=ids,
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            sublinear_tf=self.config.sublinear_tf,
        )
        
        self._validate_indices()
        elapsed = time.time() - t1
        total = time.time() - t0
        logger.info(f"      индекс построен за {elapsed:.1f} сек")
        logger.info(f"инициализация завершена за {total:.1f} сек")
    
    def _validate_indices(self) -> None:
        # проверка валидности индексов
        max_idx = len(self.df) - 1
        for idx in self.config.indices:
            if idx < 0 or idx > max_idx:
                raise ValueError(f"Индекс {idx} вне диапазона [0, {max_idx}]")
    
    def _process_index(self, idx: int) -> dict:
        # обработка одного запроса
        logger.info("\nПОИСК АНАЛОГОВ ДЛЯ СТРОКИ #%s (TF-IDF)", idx)
        
        row, analogs = find_analogs_tfidf(
            row_idx=idx,
            df=self.df,
            index=self.index,
            top_k=self.config.top_k,
            same_category=not self.config.allow_cross_category,
        )
        
        metrics = evaluate_tfidf_retrieval(row, analogs, verbose=False)
        
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
        print_analogs_tfidf(row, record["analogs"])
        self._print_metrics(record["metrics"])
    
    @staticmethod
    def _print_metrics(metrics: dict) -> None:
        # вывод метрик
        total = metrics.get("total_analogs", 0)
        print("ПРОКСИ-МЕТРИКИ (TF-IDF):")
        print(f"- Средний косинусный скор: {metrics.get('avg_score', 0):.3f}")
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
        lines.append(f"TF-IDF BASELINE RUN @ {timestamp}")
        lines.append(
            f"Параметры: max_features={self.config.max_features}, "
            f"ngram_range={self.config.ngram_range}, "
            f"min_df={self.config.min_df}, max_df={self.config.max_df}"
        )
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
                        f"| cat={cand.get('root_category', '')} | TF-IDF score={score:.3f}"
                    )
            else:
                lines.append("  (аналогов не найдено)")
        
        lines.append("\n")
        
        with output_path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    # парсинг аргументов командной строки
    parser = argparse.ArgumentParser(
        description="TF-IDF baseline для поиска аналогов товаров Best Buy."
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
        default="comparison_results/tfidf_results.txt",
        help="Путь к файлу результатов",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=15000,
        help="Максимальное количество признаков TF-IDF",
    )
    parser.add_argument(
        "--ngram-range",
        type=str,
        default="1,2",
        help="Диапазон n-грамм (например, '1,2' для униграмм и биграмм)",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=2,
        help="Минимальная частота документа для термина",
    )
    parser.add_argument(
        "--max-df",
        type=float,
        default=0.95,
        help="Максимальная частота документа (0.0-1.0)",
    )
    parser.add_argument(
        "--sublinear-tf",
        action="store_true",
        default=False,
        help="Использовать логарифмическое масштабирование TF",
    )
    return parser.parse_args()


def build_config_from_cli(args: argparse.Namespace) -> TFIDFConfig:
    # создание конфигурации из аргументов CLI
    indices = parse_example_indices(args.indices)
    if args.top_k <= 0:
        raise ValueError("top_k должен быть больше нуля")
    
    # парсим ngram_range
    ngram_parts = args.ngram_range.split(",")
    if len(ngram_parts) != 2:
        raise ValueError("ngram-range должен быть в формате 'min,max' (например, '1,2')")
    ngram_range = (int(ngram_parts[0]), int(ngram_parts[1]))
    
    return TFIDFConfig(
        dataset=Path(args.dataset),
        indices=tuple(indices),
        top_k=args.top_k,
        allow_cross_category=args.allow_cross_category,
        output_file=Path(args.output_file),
        max_features=args.max_features,
        ngram_range=ngram_range,
        min_df=args.min_df,
        max_df=args.max_df,
        sublinear_tf=args.sublinear_tf,
    )


def main() -> None:
    # главная функция
    args = parse_args()
    pipeline = TFIDFPipeline(build_config_from_cli(args))
    pipeline.run()


if __name__ == "__main__":
    main()
