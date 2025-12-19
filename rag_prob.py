import argparse
import hashlib
import importlib.util
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# защита от крашей
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['PYTORCH_MPS_DISABLE'] = '1'
os.environ['TORCH_DEVICE'] = 'cpu'

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from common import evaluate_retrieval_metrics, print_analogs

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover - transformers опционален
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None
    TRANSFORMERS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("rag")

# обработчик сигналов для перехвата bus error на macOS
def bus_error_handler(signum, frame):
    # обработчик bus error - логируем и завершаем работу корректно
    logger.error("Обнаружен bus error (SIGBUS). Это может быть связано с MPS на macOS.")
    logger.error("Попробуйте использовать модель по умолчанию или уменьшить количество индексов.")
    sys.exit(1)

# регистрируем обработчик для SIGBUS (bus error)
if hasattr(signal, 'SIGBUS'):
    signal.signal(signal.SIGBUS, bus_error_handler)

DEFAULT_DATASET = Path("Best-Buy-dataset-clean.csv")
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DEFAULT_GEN_MODEL = os.getenv("GEN_MODEL_NAME", "google/flan-t5-base")
DEFAULT_INDICES = (0, 10, 20)
DEFAULT_TOP_K = 5
MAX_NEW_TOKENS = 200
TEXT_COLUMNS = [
    "title",
    "root_category",
    "product_description",
    "features_summary",
    "product_specifications",
]


@dataclass
class RAGConfig:
    dataset: Path = DEFAULT_DATASET
    embedding_model: str = DEFAULT_EMBED_MODEL
    generation_model: Optional[str] = DEFAULT_GEN_MODEL
    indices: Tuple[int, ...] = DEFAULT_INDICES
    top_k: int = DEFAULT_TOP_K
    allow_cross_category: bool = False
    disable_generation: bool = False
    output_file: Path = Path("comparison_results/rag_results.txt")


@dataclass
class AnalogRecord:
    item_id: int
    score: float
    title: str
    category: str
    brand: str


@dataclass
class ResultRecord:
    query_index: int
    title: str
    category: str
    brand: str
    analogs: List[AnalogRecord]
    metrics: dict
    summary: str


# загрузка данных и базовая подготовка


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


# формирование "паспортов" товаров


def build_component(row: pd.Series) -> str:
    # собираем ключевые поля товара в единый текст
    brand = row.get("brand", "")
    title = row.get("title", "")
    category = row.get("root_category", "")
    desc = row.get("product_description", "")
    features = row.get("features_summary", "")
    specs = row.get("product_specifications", "")

    # шаблон
    component = (
        f"Бренд: {brand}. "
        f"Категория: {category}. "
        f"Название: {title}. "
        f"Описание: {desc}. "
        f"Особенности: {features}. "
        f"Характеристики: {specs}."
    )

    return component


def add_component_text(df: pd.DataFrame) -> pd.DataFrame:
    df["component_text"] = df.apply(build_component, axis=1)
    return df


# работа с эмбеддингами


def load_embedding_model(model_name: str) -> SentenceTransformer:
    # загружаем sentence-transformers модель по названию
    import torch
    
    logger.info(f"Загружаем модель эмбеддингов: {model_name}")
    logger.info("      (это может занять время при первом запуске - модель скачивается)")
    
    # отключаем mps перед загрузкой модели
    if hasattr(torch.backends, 'mps'):
        if torch.backends.mps.is_available():
            logger.info("      MPS доступен, но используем CPU для стабильности")
            torch.backends.mps.is_available = lambda: False
    
    # загружаем модель на CPU с логированием времени
    load_start = time.time()
    logger.info("      начало загрузки модели...")
    logger.info("      (если модель не скачана, это может занять 1-2 минуты)")
    
    try:
        model = SentenceTransformer(model_name, device='cpu')
        load_time = time.time() - load_start
        logger.info(f"      модель загружена за {load_time:.1f} сек")
    except Exception as e:
        load_time = time.time() - load_start
        logger.error(f"      ошибка при загрузке модели за {load_time:.1f} сек: {e}")
        raise
    
    # явно перемещаем все модули на CPU
    if hasattr(model, '_modules'):
        for name, module in model._modules.items():
            if hasattr(module, 'to'):
                try:
                    module.to('cpu')
                    # также отключаем gradient для экономии памяти
                    if hasattr(module, 'eval'):
                        module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
                except Exception:
                    pass
    
    # также перемещаем все параметры модели на CPU
    if hasattr(model, 'to'):
        try:
            model.to('cpu')
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
        except Exception:
            pass
    
    return model


def _get_cache_path(model_name: str, dataset_path: Path, num_items: int) -> Path:
    # генерируем путь к файлу кэша эмбеддингов
    # создаем безопасное имя модели (заменяем / на _)
    safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    dataset_name = dataset_path.stem
    # добавляем хеш от количества элементов для валидации
    cache_key = f"{safe_model_name}_{dataset_name}_{num_items}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"embeddings_{safe_model_name}_{dataset_name}_{cache_hash}.npy"


def load_cached_embeddings(cache_path: Path) -> Optional[np.ndarray]:
    # загружаем эмбеддинги из кэша, если файл существует
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
    # сохраняем эмбеддинги в кэш
    try:
        np.save(cache_path, embeddings)
        logger.info(f"      эмбеддинги сохранены в кэш: {cache_path.name}")
    except Exception as e:
        logger.warning(f"      не удалось сохранить кэш: {e}")


def embed_texts(model: SentenceTransformer, texts: Sequence[str], *, show_progress: bool = True) -> np.ndarray:
    # генерируем и нормализуем эмбеддинги для списка текстов
    # эмбеддинги нормализуются для использования с косинусовской метрикой
    import torch
    
    # отключаем mps для избежания bus error на macOS
    if hasattr(torch.backends, 'mps'):
        try:
            torch.backends.mps.is_available = lambda: False
            if hasattr(torch.backends.mps, '_is_available'):
                torch.backends.mps._is_available = False
        except Exception:
            pass
    
    # перемещаем модель на CPU
    if hasattr(model, '_modules'):
        for module in model._modules.values():
            if hasattr(module, 'to'):
                try:
                    module.to('cpu').eval()
                    for param in module.parameters():
                        param.requires_grad = False
                        if hasattr(param, 'data'):
                            param.data = param.data.cpu()
                except Exception:
                    pass
    
    # генерируем эмбеддинги с защитой от ошибок
    batch_size = 1 if len(texts) <= 1 else 32
    encode_params = {
        'show_progress_bar': show_progress if len(texts) > 1 else False,
        'convert_to_numpy': True,
        'batch_size': batch_size,
        'normalize_embeddings': False
    }
    
    try:
        emb = model.encode(texts, device='cpu', **encode_params)
    except (TypeError, ValueError, AttributeError):
        try:
            emb = model.encode(texts, **encode_params)
        except Exception:
            emb = model.encode(texts, show_progress_bar=show_progress, convert_to_numpy=True)
    
    # конвертируем в numpy и нормализуем
    if hasattr(emb, 'cpu'):
        emb = emb.cpu().numpy()
    elif hasattr(emb, 'numpy'):
        emb = emb.numpy()
    elif not isinstance(emb, np.ndarray):
        emb = np.array(emb)
    
    if len(emb.shape) == 1:
        emb = emb.reshape(1, -1)
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    return emb / norms


def describe_embedding_model(model: SentenceTransformer, model_name: str) -> None:
    # выводим ключевые параметры эмбеддинг-модели
    dim = model.get_sentence_embedding_dimension()
    max_seq = model.get_max_seq_length()
    print(f"\nЭМБЕДДИНГ-МОДЕЛЬ: {model_name} | dim={dim} | max_seq={max_seq}")


# простейший in-memory индекс


class SimpleVectorIndex:
    # храним матрицу эмбеддингов и соответствующие идентификаторы

    def __init__(self, embeddings: np.ndarray, ids: Sequence[int]):
        if len(embeddings) != len(ids):
            raise ValueError("Размерность эмбеддингов и списка id не совпадает")
        self.embeddings = embeddings
        self.ids = list(ids)
        self.id_to_pos = {item_id: pos for pos, item_id in enumerate(self.ids)}

    def search(self, query_vec: np.ndarray, top_k: int = 5):
        # поиск через косинусовское сходство между нормализованными эмбеддингами
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        scores = cosine_similarity(query_vec, self.embeddings)[0]
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = [(self.ids[i], float(scores[i])) for i in top_idx]
        return results

    def embedding_for(self, item_id: int) -> np.ndarray:
        pos = self.id_to_pos.get(item_id)
        if pos is None:
            raise KeyError(f"ID {item_id} не найден в индексе")
        return self.embeddings[pos]


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


# поиск аналогов

def find_analogs_for_row(
    row_idx: int,
    df: pd.DataFrame,
    model: SentenceTransformer,
    index: SimpleVectorIndex,
    top_k: int = 5,
    same_category: bool = True
) -> Tuple[pd.Series, List[Tuple[int, float, pd.Series]]]:
    # ищем аналоги для товара по позиции row_idx
    # same_category=True фильтрует результаты, оставляя только записи из той же root_category

    if row_idx < 0 or row_idx >= len(df):
        raise ValueError(f"Индекс {row_idx} вне диапазона [0, {len(df)-1}]")

    row = df.iloc[row_idx]
    query_cat = row.get("root_category", None)
    query_id = row.name

    # всегда используем эмбеддинги из индекса, чтобы избежать bus error при пересчете
    # если ID не найден, это ошибка конфигурации, а не повод для пересчета
    try:
        query_emb = index.embedding_for(query_id)
    except KeyError:
        # если ID не найден в индексе, это критическая ошибка
        # не пытаемся пересчитать, чтобы избежать bus error на macOS
        error_msg = (
            f"ID {query_id} не найден в индексе. "
            f"Это может быть связано с несоответствием индексов DataFrame и индекса эмбеддингов. "
            f"Убедитесь, что индекс создается из того же DataFrame, что используется для поиска."
        )
        logger.error(error_msg)
        raise KeyError(error_msg)

    results = index.search(query_emb, top_k=top_k + 10)

    analogs = []
    for rec_id, score in results:
        if rec_id == query_id:
            continue

        try:
            candidate = df.loc[rec_id]
        except KeyError:
            # пропускаем, если запись не найдена
            continue

        # фильтрация по категории для сохранения сути
        # примечание: категория - первый и основной критерий "сути" товара
        if same_category and query_cat is not None:
            if candidate.get("root_category", None) != query_cat:
                continue

        analogs.append((rec_id, score, candidate))

        if len(analogs) >= top_k:
            break

    return row, analogs




# используем evaluate_retrieval_metrics из common.py
evaluate_retrieval_proxy = lambda row, analogs, verbose=True: evaluate_retrieval_metrics(row, analogs, verbose=verbose, method_name="")


def _ensure_generation_dependencies(model_name: str) -> None:
    # проверяем, что установлены библиотеки, необходимые для загрузки LLM
    required = []
    if importlib.util.find_spec("sentencepiece") is None:
        required.append("sentencepiece")
    try:  # protobuf живёт внутри пространства имён google.*
        import google.protobuf  # type: ignore
    except ImportError:
        required.append("protobuf")
    if required:
        raise RuntimeError(
            "Не хватает зависимостей для модели "
            f"{model_name}: {', '.join(required)}. "
            "Установи их командой `pip install "
            + " ".join(required)
            + "` и перезапусти скрипт."
        )


def load_generation_backend(model_name: Optional[str]):
    # загружаем генеративную модель (LLM) для генерации аналитического отчёта
    # примечание: generation используется для генерации отчёта о результатах поиска,
    # а не для генерации ответа пользователю или нормализации результата
    # возвращаем None, если transformers не установлен или выключен флагом
    if not model_name:
        print("Генерация отключена (не задано имя модели).")
        return None
    if not TRANSFORMERS_AVAILABLE:
        print("transformers недоступен, используем текстовый шаблон без LLM.")
        return None

    try:
        _ensure_generation_dependencies(model_name)
    except RuntimeError as err:
        logger.error("%s", err)
        return None

    print(f"\nЗагружаем генеративную модель: {model_name}")
    try:
        import torch
        # принудительно используем cpu для избежания bus error
        device = "cpu"
        if torch.backends.mps.is_available():
            print("MPS доступен, но используем CPU для стабильности")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        # явно перемещаем на cpu перед возвратом
        model = model.to(device)
        # отключаем gradient для экономии памяти
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    except ImportError as err:
        logger.error(
            "transformers не смог загрузить модель %s: %s. "
            "Убедись, что установлены sentencepiece и protobuf.",
            model_name,
            err,
        )
        return None
    except Exception as err:
        logger.error(f"Ошибка при загрузке модели {model_name}: {err}")
        return None

    return tokenizer, model


def build_generation_prompt(row: pd.Series, analogs: List[Tuple[int, float, pd.Series]]) -> str:
    # формируем промпт для LLM на основе исходного товара и найденных аналогов
    analog_lines = []
    for idx, (candidate_id, score, cand) in enumerate(analogs, start=1):
        analog_lines.append(
            f"{idx}) [{candidate_id}] {cand.get('title', '')} | "
            f"Категория: {cand.get('root_category', '')} | Скор={score:.3f}"
        )

    prompt = f"""Ты аналитик витрины товаров. Проанализируй исходное описание и список найденных аналогов
        и сделай краткое заключение: сохраняется ли суть, чем похожи/отличаются позиции, какие
        числовые характеристики стоит уточнить. Пиши по-русски, 2-3 предложения.

        Исходный товар: {row.get('title', '')}
        Категория: {row.get('root_category', '')}
        Описание: {row.get('product_description', '')}

        Аналоги:
        {chr(10).join(analog_lines)}

        Ответ:
""".strip()
    return prompt


def generate_augmented_summary(
    generator_backend,
    row: pd.Series,
    analogs: List[Tuple[int, float, pd.Series]],
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    # формируем текстовое объяснение результатов поиска
    # примечание: generation используется для генерации аналитического отчёта,
    # а не для генерации ответа пользователю или нормализации результата
    if not analogs:
        return "Не найдено кандидатов, генерация отчёта пропущена."

    if generator_backend is None:
        # fallback без llm - простой анализ на основе метрик
        template = []
        
        # анализируем категории
        base_cat = row.get('root_category', '')
        cat_matches = sum(1 for _, _, cand in analogs if cand.get('root_category') == base_cat)
        cat_match_pct = (cat_matches / len(analogs) * 100) if analogs else 0
        
        # средний скор
        avg_score = sum(score for _, score, _ in analogs) / len(analogs) if analogs else 0
        
        template.append(f"анализ результатов поиска:")
        template.append(f"- найдено аналогов: {len(analogs)}")
        template.append(f"- средний скор сходства: {avg_score:.3f}")
        template.append(f"- совпадение категории: {cat_matches}/{len(analogs)} ({cat_match_pct:.0f}%)")
        
        # выводы
        template.append("")
        if cat_match_pct == 100:
            template.append("вывод: суть категории сохраняется, все аналоги из той же категории.")
        elif cat_match_pct >= 50:
            template.append("вывод: большинство аналогов из той же категории, но есть расхождения.")
        else:
            template.append("вывод: много аналогов из других категорий - проверь фильтры.")
        
        if avg_score > 0.7:
            template.append("качество поиска: высокое (скор > 0.7)")
        elif avg_score > 0.5:
            template.append("качество поиска: среднее (скор 0.5-0.7)")
        else:
            template.append("качество поиска: низкое (скор < 0.5)")
        
        return "\n".join(template)

    # генерация через LLM
    prompt = build_generation_prompt(row, analogs)
    tokenizer, model = generator_backend
    
    try:
        import torch
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        
        # агрессивная оптимизация для предотвращения зависаний
        # используем минимальные параметры для максимальной скорости
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=min(max_new_tokens, 100),  # еще больше ограничиваем
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                # отключаем все что может замедлить
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except Exception as e:
        logger.warning(f"      ошибка при генерации через LLM: {e}, используем fallback")
        # возвращаем fallback если генерация не удалась
        avg_score = sum(s for _, s, _ in analogs) / len(analogs) if analogs else 0
        cat_matches = sum(1 for _, _, cand in analogs if cand.get('root_category') == row.get('root_category', ''))
        return "\n".join([
            f"анализ результатов поиска:",
            f"- найдено аналогов: {len(analogs)}",
            f"- средний скор: {avg_score:.3f}",
            f"- совпадение категории: {cat_matches}/{len(analogs)}",
            "вывод: анализ завершен (LLM генерация недоступна)"
        ])


# основной пайплайн


class RAGPipeline:
    # подготовка данных, поиск аналогов и генерация отчёта
    # примечание: используется RAG-подобная архитектура
    # - Retrieval: dense retrieval через эмбеддинги (реализовано)
    # - Generation: генерация аналитического отчёта о результатах поиска,
    #   а не генерация ответа пользователю или нормализация результата

    def __init__(self, config: RAGConfig):
        self.config = config
        self.df: pd.DataFrame = pd.DataFrame()
        self.embedder: Optional[SentenceTransformer] = None
        self.index: Optional[SimpleVectorIndex] = None
        self.generator_backend = None
        self.records: List[ResultRecord] = []

    def run(self) -> None:
        total_start = time.time()
        self._setup()
        if self.embedder is None or self.index is None:
            raise RuntimeError("Пайплайн не инициализирован. Вызовите run() после _setup().")

        logger.info("")
        logger.info(f"поиск аналогов для {len(self.config.indices)} товаров")

        self.records = []
        for i, idx in enumerate(self.config.indices, 1):
            logger.info(f"\n[{i}/{len(self.config.indices)}] обработка товара с индексом {idx}...")
            t_start = time.time()
            record = self._process_index(idx)
            elapsed = time.time() - t_start
            self.records.append(record)
            logger.info(f"      найдено {len(record.analogs)} аналогов за {elapsed:.2f} сек")
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
        logger.info("инициализация пайплайна")
        
        t0 = time.time()
        logger.info(f"[1/3] загрузка данных из {self.config.dataset}...")
        self.df = add_component_text(load_data(self.config.dataset))
        if self.df.empty:
            raise ValueError("Датасет пустой, нечего индексировать")
        elapsed = time.time() - t0
        logger.info(f"      загружено {len(self.df)} товаров за {elapsed:.1f} сек")

        t1 = time.time()
        logger.info(f"[2/3] загрузка модели эмбеддингов '{self.config.embedding_model}'...")
        self.embedder = load_embedding_model(self.config.embedding_model)
        describe_embedding_model(self.embedder, self.config.embedding_model)
        elapsed = time.time() - t1

        t2 = time.time()
        logger.info(f"[3/3] генерация эмбеддингов для {len(self.df)} товаров...")
        
        # проверяем кэш
        cache_path = _get_cache_path(self.config.embedding_model, self.config.dataset, len(self.df))
        embeddings = load_cached_embeddings(cache_path)
        
        if embeddings is None:
            logger.info("      (это может занять некоторое время)")
            embeddings = embed_texts(self.embedder, self.df["component_text"].tolist())
            elapsed = time.time() - t2
            logger.info(f"      эмбеддинги сгенерированы за {elapsed:.1f} сек ({elapsed/len(self.df)*1000:.1f} мс/товар)")
            save_cached_embeddings(embeddings, cache_path)
        else:
            elapsed = time.time() - t2
            logger.info(f"      эмбеддинги загружены из кэша за {elapsed:.1f} сек")
        
        self.index = SimpleVectorIndex(embeddings, self.df.index.tolist())
        
        total_setup = time.time() - t0
        logger.info(f"инициализация завершена за {total_setup:.1f} сек")

        self._validate_indices()

        if not self.config.disable_generation:
            self.generator_backend = load_generation_backend(self.config.generation_model)
        else:
            logger.info("Генерация отключена (--disable-generation).")

    def _validate_indices(self) -> None:
        max_idx = len(self.df) - 1
        validated: List[int] = []
        for idx in self.config.indices:
            if idx < 0 or idx > max_idx:
                raise ValueError(f"Индекс {idx} вне диапазона [0, {max_idx}]")
            validated.append(idx)
        self.config = replace(self.config, indices=tuple(validated))

    def _process_index(self, idx: int) -> ResultRecord:
        logger.info("\nПОИСК АНАЛОГОВ ДЛЯ СТРОКИ #%s", idx)
        
        t_search = time.time()
        row, analogs = find_analogs_for_row(
            row_idx=idx,
            df=self.df,
            model=self.embedder,
            index=self.index,
            top_k=self.config.top_k,
            same_category=not self.config.allow_cross_category,
        )
        search_time = time.time() - t_search
        logger.info(f"      поиск завершен за {search_time:.2f} сек")
        
        metrics = evaluate_retrieval_proxy(row, analogs, verbose=False)
        
        t_gen = time.time()
        logger.info("      генерация отчета...")
        summary = generate_augmented_summary(self.generator_backend, row, analogs)
        gen_time = time.time() - t_gen
        logger.info(f"      отчет сгенерирован за {gen_time:.2f} сек")

        analog_records = [
            AnalogRecord(
                item_id=candidate_id,
                score=score,
                title=cand.get("title", ""),
                category=cand.get("root_category", ""),
                brand=cand.get("brand", ""),
            )
            for candidate_id, score, cand in analogs
        ]

        return ResultRecord(
            query_index=idx,
            title=row.get("title", ""),
            category=row.get("root_category", ""),
            brand=row.get("brand", ""),
            analogs=analog_records,
            metrics=metrics,
            summary=summary,
        )

    def _render_record(self, record: ResultRecord) -> None:
        row = self.df.iloc[record.query_index]
        analog_rows = [
            (
                analog.item_id,
                analog.score,
                self.df.loc[analog.item_id]
                if 0 <= analog.item_id < len(self.df)
                else pd.Series(
                    {
                        "title": analog.title,
                        "brand": analog.brand,
                        "root_category": analog.category,
                        "product_description": "",
                    }
                ),
            )
            for analog in record.analogs
        ]
        print_analogs(row, analog_rows)
        self._print_metrics(record.metrics)
        print("\nГЕНЕРАЦИЯ ОТЧЁТА:")
        print(record.summary)

    @staticmethod
    def _print_metrics(metrics: dict) -> None:
        total = metrics.get("total_analogs", 0)
        print("ПРОКСИ-МЕТРИКИ:")
        print(f"- Средний косинусный скор: {metrics.get('avg_score', 0):.3f}")
        print(f"- Совпадение категории: {metrics.get('same_category_hits', 0)}/{total}")
        if total and metrics.get("same_category_hits") == total:
            print("  (Суть категории сохраняется, RAG-подход выглядит уместным)")
        else:
            print("  (Есть расхождения по категориям — проверь фильтры или шаблон компоненты)")

    def _write_results(self) -> None:
        if not self.records:
            return
        output_path = self.config.output_file
        # создаём директорию для результатов
        output_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        total_queries = len(self.records)
        total_analogs = sum(r.metrics["total_analogs"] for r in self.records)
        avg_scores = [
            r.metrics["avg_score"] for r in self.records if r.metrics["total_analogs"] > 0
        ]
        mean_avg_score = sum(avg_scores) / len(avg_scores) if avg_scores else 0.0
        total_cat_hits = sum(r.metrics["same_category_hits"] for r in self.records)
        cat_ratio = total_cat_hits / total_analogs if total_analogs else 0.0

        lines: List[str] = []
        lines.append(f"RAG RUN @ {timestamp}")
        lines.append(
            f"Запросов: {total_queries} | Всего аналогов: {total_analogs} | Средний скор: {mean_avg_score:.3f}"
        )
        lines.append(
            f"Совпадение категорий: {total_cat_hits}/{total_analogs} ({cat_ratio:.1%})"
        )

        for record in self.records:
            lines.append("\n---")
            lines.append(
                f"Запрос #{record.query_index} | {record.title} ({record.category})"
            )
            metrics = record.metrics
            lines.append(
                f"Метрики: avg_score={metrics['avg_score']:.3f}, "
                f"cat {metrics['same_category_hits']}/{metrics['total_analogs']}"
            )
            lines.append("Аналоги:")
            if record.analogs:
                for i, analog in enumerate(record.analogs, start=1):
                    lines.append(
                        f"  {i}) [ID {analog.item_id}] {analog.title} "
                        f"| cat={analog.category} | score={analog.score:.3f}"
                    )
            else:
                lines.append("  (аналогов не найдено)")
            lines.append("Генерация:")
            lines.append(record.summary)

        lines.append("\n")

        with output_path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines))





def parse_args() -> argparse.Namespace:
    # парсинг аргументов командной строки
    parser = argparse.ArgumentParser(
        description="RAG-пайплайн для поиска аналогов товаров Best Buy."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Путь к CSV")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL, help="sentence-transformer")
    parser.add_argument("--gen-model", default=DEFAULT_GEN_MODEL, help="LLM для отчёта")
    parser.add_argument("--indices", default="0,10,20", help="Список индексов через запятую")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Сколько аналогов возвращать")
    parser.add_argument(
        "--allow-cross-category",
        action="store_true",
        help="Разрешить аналоги из других категорий",
    )
    parser.add_argument(
        "--disable-generation",
        action="store_true",
        help="Отключить генерацию текста",
    )
    parser.add_argument(
        "--output-file",
        default="comparison_results/rag_results.txt",
        help="Путь к файлу, куда писать результаты и метрики (append)",
    )
    return parser.parse_args()


def build_config_from_cli(args: argparse.Namespace) -> RAGConfig:
    indices = parse_example_indices(args.indices)
    if args.top_k <= 0:
        raise ValueError("top_k должен быть больше нуля")
    generation_model = None if args.disable_generation else args.gen_model
    return RAGConfig(
        dataset=Path(args.dataset),
        embedding_model=args.embedding_model,
        generation_model=generation_model,
        indices=tuple(indices),
        top_k=args.top_k,
        allow_cross_category=args.allow_cross_category,
        disable_generation=args.disable_generation,
        output_file=Path(args.output_file),
    )


def main() -> None:
    args = parse_args()
    pipeline = RAGPipeline(build_config_from_cli(args))
    pipeline.run()

if __name__ == "__main__":
    main()
