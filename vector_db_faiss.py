# FAISS Vector Database для поиска аналогов товаров
# FAISS (Facebook AI Similarity Search) - высокопроизводительная библиотека
# для векторного поиска, разработанная Facebook Research
# 
# Преимущества FAISS:
# - Очень быстрый поиск даже на больших датасетах (миллионы векторов)
# - Поддержка различных типов индексов (Flat, HNSW, IVF, IVFPQ)
# - GPU ускорение (опционально)
# - Эффективное использование памяти
# 
# Недостатки FAISS:
# - Требует установки дополнительной библиотеки (faiss-cpu или faiss-gpu)
# - Сложнее в использовании, чем простой in-memory индекс
# - Для маленьких датасетов (<10K) overhead может быть избыточным
# 
# Типы индексов:
# - Flat: Точный поиск, O(n) сложность, используется для небольших датасетов
# - HNSW: Приближённый поиск, O(log(n)) сложность, хорош для больших датасетов
# - IVF: Кластерный индекс, быстрый поиск с небольшой потерей точности
# - IVFPQ: Сжатие векторов, экономия памяти, подходит для очень больших датасетов

import argparse
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# отключаем mps для faiss (избегаем sigsegv на macos)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

from common import evaluate_retrieval_metrics, print_analogs
from rag_prob import (
    DEFAULT_DATASET,
    DEFAULT_EMBED_MODEL,
    DEFAULT_INDICES,
    DEFAULT_TOP_K,
    add_component_text,
    embed_texts,
    load_data,
    load_embedding_model,
    parse_example_indices,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("faiss_db")


@dataclass
class FAISSConfig:
    dataset: Path = DEFAULT_DATASET
    embedding_model: str = DEFAULT_EMBED_MODEL
    indices: Tuple[int, ...] = DEFAULT_INDICES
    top_k: int = DEFAULT_TOP_K
    allow_cross_category: bool = False
    output_file: Path = Path("comparison_results/faiss_results.txt")
    index_type: str = "Flat"  # "Flat", "HNSW", "IVF" (HNSW может вызывать segmentation fault на macOS)
    index_file: Optional[Path] = None  # Путь для сохранения/загрузки индекса
    hnsw_m: int = 32  # Параметр HNSW: количество связей
    hnsw_ef_construction: int = 200  # Параметр HNSW: размер кандидатов при построении
    hnsw_ef_search: int = 50  # Параметр HNSW: размер кандидатов при поиске
    ivf_nlist: int = 100  # Параметр IVF: количество кластеров


class FAISSIndex:
    # FAISS индекс для векторного поиска
    # поддерживает различные типы индексов в зависимости от размера датасета
    # и требований к скорости/точности

    def __init__(
        self,
        embeddings: np.ndarray,
        ids: Sequence[int],
        index_type: str = "HNSW",
        index_file: Optional[Path] = None,
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 50,
        ivf_nlist: int = 100,
    ):
        # инициализация FAISS индекса
        # embeddings: матрица эмбеддингов [n_items, embedding_dim]
        # ids: список ID товаров
        # index_type: тип индекса ("Flat", "HNSW", "IVF")
        # index_file: путь для сохранения/загрузки индекса
        # hnsw_m: параметр HNSW (количество связей)
        # hnsw_ef_construction: параметр HNSW (размер кандидатов при построении)
        # hnsw_ef_search: параметр HNSW (размер кандидатов при поиске)
        # ivf_nlist: параметр IVF (количество кластеров)
        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS не установлен. Установите: pip install faiss-cpu "
                "(или faiss-gpu для GPU поддержки)"
            )
        
        if len(embeddings) != len(ids):
            raise ValueError("Размерность эмбеддингов и списка id не совпадает")
        
        self.ids = list(ids)
        self.id_to_pos = {item_id: pos for pos, item_id in enumerate(self.ids)}
        self.embedding_dim = embeddings.shape[1]
        self.index_type = index_type
        
        # проверяем, можно ли загрузить существующий индекс
        if index_file and index_file.exists():
            logger.info(f"Загрузка FAISS индекса из {index_file}...")
            self.index = faiss.read_index(str(index_file))
            logger.info(f"Загружен индекс типа: {self.index_type}")
        else:
            # создаём новый индекс
            logger.info(f"Создание FAISS индекса типа: {index_type}")
            self.index = self._create_index(
                embeddings,
                index_type,
                hnsw_m,
                hnsw_ef_construction,
                ivf_nlist,
            )
            
            # добавляем векторы в индекс
            # faiss требует float32
            embeddings_f32 = embeddings.astype('float32')
            self.index.add(embeddings_f32)
            
            logger.info(f"Добавлено {len(embeddings)} векторов в индекс")
            
            # сохраняем индекс, если указан путь
            if index_file:
                index_file.parent.mkdir(parents=True, exist_ok=True)
                faiss.write_index(self.index, str(index_file))
                logger.info(f"Индекс сохранён в {index_file}")
        
        # устанавливаем параметры поиска для hnsw
        if index_type == "HNSW" and hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = hnsw_ef_search
            logger.info(f"Установлен ef_search={hnsw_ef_search} для HNSW")

    def _create_index(
        self,
        embeddings: np.ndarray,
        index_type: str,
        hnsw_m: int,
        hnsw_ef_construction: int,
        ivf_nlist: int,
    ) -> "faiss.Index":
        # создание FAISS индекса заданного типа
        # embeddings: матрица эмбеддингов
        # index_type: тип индекса
        # hnsw_m: параметр HNSW
        # hnsw_ef_construction: параметр HNSW
        # ivf_nlist: параметр IVF
        # возвращает: FAISS индекс
        dim = embeddings.shape[1]
        
        if index_type == "Flat":
            # точный поиск, o(n) сложность
            # используется для небольших датасетов (<100k)
            index = faiss.IndexFlatL2(dim)
            logger.info("Создан Flat индекс (точный поиск)")
        
        elif index_type == "HNSW":
            # приближённый поиск, o(log(n)) сложность
            # хорош для больших датасетов (100k - миллионы)
            # на macOS HNSW может вызывать segmentation fault, используем Flat как fallback
            try:
                index = faiss.IndexHNSWFlat(dim, hnsw_m)
                index.hnsw.efConstruction = hnsw_ef_construction
                logger.info(
                    f"Создан HNSW индекс (m={hnsw_m}, ef_construction={hnsw_ef_construction})"
                )
            except (RuntimeError, SystemError, OSError) as e:
                # fallback на Flat индекс при ошибках (особенно на macOS)
                logger.warning(
                    f"Не удалось создать HNSW индекс: {e}. "
                    f"Используется Flat индекс как fallback."
                )
                index = faiss.IndexFlatL2(dim)
                logger.info("Создан Flat индекс (fallback из-за ошибки HNSW)")
        
        elif index_type == "IVF":
            # кластерный индекс, быстрый поиск с небольшой потерей точности
            # требует обучения на данных
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, ivf_nlist)
            logger.info(f"Создан IVF индекс (nlist={ivf_nlist})")
        
        else:
            raise ValueError(
                f"Неизвестный тип индекса: {index_type}. "
                f"Доступные: Flat, HNSW, IVF"
            )
        
        return index

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        # поиск наиболее похожих векторов
        # query_vec: вектор запроса [embedding_dim] или [1, embedding_dim]
        # top_k: количество результатов
        # возвращает: список кортежей (item_id, distance), отсортированный по возрастанию distance
        # (меньше distance = больше сходство)
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        
        # FAISS требует float32
        query_vec = query_vec.astype('float32')
        
        # для ivf индекса нужно указать количество кластеров для поиска
        if self.index_type == "IVF":
            self.index.nprobe = min(10, self.index.nlist)  # Ищем в 10 ближайших кластерах
        
        # поиск через FAISS (используется L2 расстояние)
        # примечание: для L2-нормализованных векторов L2 монотонно эквивалентно cosine,
        # но метрика score выводится в удобный диапазон через нормализацию от max_dist
        # для строгого cosine можно использовать IndexFlatIP + нормализация эмбеддингов
        distances, indices = self.index.search(query_vec, top_k)
        
        # улучшенное преобразование L2 расстояний в скоры сходства
        # используем более мягкое преобразование для лучшего распределения
        if len(distances[0]) > 0:
            # нормализуем расстояния
            max_dist = distances[0].max()
            min_dist = distances[0].min()
            if max_dist > min_dist:
                normalized = (distances[0] - min_dist) / (max_dist - min_dist + 1e-8)
            else:
                normalized = np.zeros_like(distances[0])
            # более мягкое преобразование: 1 - normalized с квадратичным затуханием
            scores = 1.0 - normalized * normalized  # квадратичное затухание для плавности
        else:
            scores = np.array([])
        
        # формируем результат: (item_id, score)
        results = [
            (self.ids[int(idx)], float(score))
            for idx, score in zip(indices[0], scores)
            if int(idx) < len(self.ids) and int(idx) >= 0
        ]
        
        return results

    def embedding_for(self, item_id: int) -> np.ndarray:
        # получение эмбеддинга по ID
        # для FAISS это требует хранения исходных эмбеддингов отдельно,
        # так как индексы могут сжимать/квантовать векторы
        # в этой реализации мы не храним исходные эмбеддинги в индексе,
        # поэтому этот метод должен быть реализован на уровне пайплайна
        raise NotImplementedError(
            "FAISS индекс не хранит исходные эмбеддинги. "
            "Используйте эмбеддинги из пайплайна."
        )


def find_analogs_faiss(
    row_idx: int,
    df: pd.DataFrame,
    model: SentenceTransformer,
    index: FAISSIndex,
    embeddings_dict: dict,  # словарь {item_id: embedding}
    top_k: int = 5,
    same_category: bool = True,
) -> Tuple[pd.Series, List[Tuple[int, float, pd.Series]]]:
    # поиск аналогов для товара с использованием FAISS
    # row_idx: индекс строки в DataFrame
    # df: DataFrame с товарами
    # model: модель эмбеддингов (для пересчёта, если нужно)
    # index: FAISS индекс
    # embeddings_dict: словарь с эмбеддингами {item_id: embedding}
    # top_k: количество аналогов
    # same_category: фильтровать ли по категории
    # возвращает: кортеж (исходная строка, список аналогов)
    if row_idx < 0 or row_idx >= len(df):
        raise ValueError(f"Индекс {row_idx} вне диапазона [0, {len(df)-1}]")
    
    row = df.iloc[row_idx]
    query_cat = row.get("root_category", None)
    query_id = row.name
    
    # получаем эмбеддинг запроса
    if query_id in embeddings_dict:
        query_emb = embeddings_dict[query_id]
    else:
        # fallback: пересчитываем
        query_text = row.get("component_text", "")
        query_emb = embed_texts(model, [query_text], show_progress=False)[0]
    
    # поиск кандидатов
    results = index.search(query_emb, top_k=top_k + 20)
    
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
evaluate_faiss_retrieval = lambda row, analogs, verbose=True: evaluate_retrieval_metrics(row, analogs, verbose=verbose, method_name="FAISS")
print_analogs_faiss = lambda row, analogs: print_analogs(row, analogs, method_name="FAISS")


class FAISSPipeline:
    # пайплайн поиска аналогов с использованием FAISS
    
    def __init__(self, config: FAISSConfig):
        self.config = config
        self.df: pd.DataFrame = pd.DataFrame()
        self.embedder: Optional[SentenceTransformer] = None
        self.index: Optional[FAISSIndex] = None
        self.embeddings_dict: dict = {}
        self.records: List[dict] = []
    
    def run(self) -> None:
        # запуск пайплайна FAISS
        total_start = time.time()
        self._setup()
        
        if self.index is None or self.embedder is None:
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
        # инициализация: загрузка данных, построение эмбеддингов и индекса
        logger.info("инициализация faiss пайплайна")
        
        t0 = time.time()
        logger.info("[1/4] загрузка данных...")
        self.df = add_component_text(load_data(self.config.dataset))
        
        if self.df.empty:
            raise ValueError("Датасет пустой")
        
        elapsed = time.time() - t0
        logger.info(f"      загружено {len(self.df)} товаров за {elapsed:.1f} сек")
        
        # загрузка модели эмбеддингов
        t1 = time.time()
        logger.info(f"[2/4] загрузка модели эмбеддингов '{self.config.embedding_model}'...")
        self.embedder = load_embedding_model(self.config.embedding_model)
        elapsed = time.time() - t1
        logger.info(f"      модель загружена за {elapsed:.1f} сек")
        
        # построение эмбеддингов
        t2 = time.time()
        logger.info(f"[3/4] генерация эмбеддингов для {len(self.df)} товаров...")
        logger.info("      (это может занять некоторое время)")
        texts = self.df["component_text"].tolist()
        ids = self.df.index.tolist()
        embeddings = embed_texts(self.embedder, texts)
        elapsed = time.time() - t2
        logger.info(f"      эмбеддинги сгенерированы за {elapsed:.1f} сек ({elapsed/len(self.df)*1000:.1f} мс/товар)")
        
        # сохраняем эмбеддинги в словарь для быстрого доступа
        self.embeddings_dict = {item_id: emb for item_id, emb in zip(ids, embeddings)}
        
        # построение faiss индекса
        t3 = time.time()
        logger.info(f"[4/4] построение FAISS индекса типа: {self.config.index_type}...")
        try:
            self.index = FAISSIndex(
                embeddings=embeddings,
                ids=ids,
                index_type=self.config.index_type,
                index_file=self.config.index_file,
                hnsw_m=self.config.hnsw_m,
                hnsw_ef_construction=self.config.hnsw_ef_construction,
                hnsw_ef_search=self.config.hnsw_ef_search,
                ivf_nlist=self.config.ivf_nlist,
            )
        except (RuntimeError, SystemError, OSError) as e:
            # если HNSW не работает, пробуем Flat
            if self.config.index_type == "HNSW":
                logger.warning(
                    f"HNSW индекс вызвал ошибку: {e}. "
                    f"Повторная попытка с Flat индексом..."
                )
                self.index = FAISSIndex(
                    embeddings=embeddings,
                    ids=ids,
                    index_type="Flat",
                    index_file=self.config.index_file,
                    hnsw_m=self.config.hnsw_m,
                    hnsw_ef_construction=self.config.hnsw_ef_construction,
                    hnsw_ef_search=self.config.hnsw_ef_search,
                    ivf_nlist=self.config.ivf_nlist,
                )
            else:
                raise
        
        self._validate_indices()
        elapsed = time.time() - t3
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
        logger.info("\nПОИСК АНАЛОГОВ ДЛЯ СТРОКИ #%s (FAISS)", idx)
        
        row, analogs = find_analogs_faiss(
            row_idx=idx,
            df=self.df,
            model=self.embedder,
            index=self.index,
            embeddings_dict=self.embeddings_dict,
            top_k=self.config.top_k,
            same_category=not self.config.allow_cross_category,
        )
        
        metrics = evaluate_retrieval_metrics(row, analogs, verbose=False, method_name="FAISS")
        
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
        print_analogs_faiss(row, record["analogs"])
        self._print_metrics(record["metrics"])
    
    @staticmethod
    def _print_metrics(metrics: dict) -> None:
        # вывод метрик
        total = metrics.get("total_analogs", 0)
        print("ПРОКСИ-МЕТРИКИ (FAISS):")
        print(f"- Средний скор: {metrics.get('avg_score', 0):.3f}")
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
        lines.append(f"FAISS VECTOR DB RUN @ {timestamp}")
        lines.append(f"Тип индекса: {self.config.index_type}")
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
                        f"| cat={cand.get('root_category', '')} | FAISS score={score:.3f}"
                    )
            else:
                lines.append("  (аналогов не найдено)")
        
        lines.append("\n")
        
        with output_path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    # парсинг аргументов командной строки
    parser = argparse.ArgumentParser(
        description="FAISS векторная БД для поиска аналогов товаров Best Buy."
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
        default="comparison_results/faiss_results.txt",
        help="Путь к файлу результатов",
    )
    parser.add_argument(
        "--index-type",
        choices=["Flat", "HNSW", "IVF"],
        default="Flat",  # Flat по умолчанию, т.к. HNSW вызывает segmentation fault на macOS
        help="Тип FAISS индекса (HNSW может вызывать segmentation fault на macOS)",
    )
    parser.add_argument(
        "--index-file",
        type=str,
        default=None,
        help="Путь для сохранения/загрузки индекса",
    )
    parser.add_argument(
        "--hnsw-m",
        type=int,
        default=32,
        help="Параметр HNSW: количество связей",
    )
    parser.add_argument(
        "--hnsw-ef-construction",
        type=int,
        default=200,
        help="Параметр HNSW: размер кандидатов при построении",
    )
    parser.add_argument(
        "--hnsw-ef-search",
        type=int,
        default=50,
        help="Параметр HNSW: размер кандидатов при поиске",
    )
    parser.add_argument(
        "--ivf-nlist",
        type=int,
        default=100,
        help="Параметр IVF: количество кластеров",
    )
    return parser.parse_args()


def build_config_from_cli(args: argparse.Namespace) -> FAISSConfig:
    # создание конфигурации из аргументов CLI
    indices = parse_example_indices(args.indices)
    if args.top_k <= 0:
        raise ValueError("top_k должен быть больше нуля")
    
    index_file = Path(args.index_file) if args.index_file else None
    
    return FAISSConfig(
        dataset=Path(args.dataset),
        embedding_model=args.embedding_model,
        indices=tuple(indices),
        top_k=args.top_k,
        allow_cross_category=args.allow_cross_category,
        output_file=Path(args.output_file),
        index_type=args.index_type,
        index_file=index_file,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
        hnsw_ef_search=args.hnsw_ef_search,
        ivf_nlist=args.ivf_nlist,
    )


def main() -> None:
    # главная функция
    if not FAISS_AVAILABLE:
        print("ОШИБКА: FAISS не установлен.")
        print("Установите: pip install faiss-cpu")
        print("Или для GPU: pip install faiss-gpu")
        return
    
    args = parse_args()
    pipeline = FAISSPipeline(build_config_from_cli(args))
    pipeline.run()


if __name__ == "__main__":
    main()
