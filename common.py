# общие функции для всех подходов поиска аналогов

import pandas as pd
from typing import List, Tuple, Optional


def normalize_meta(value: Optional[str]) -> str:
    # нормализация метаданных (бренд, категория)
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip().lower()
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def evaluate_retrieval_metrics(
    row: pd.Series,
    analogs: List[Tuple[int, float, pd.Series]],
    *,
    verbose: bool = True,
    method_name: str = "",
) -> dict:
    # универсальная функция для вычисления прокси-метрик
    if not analogs:
        if verbose:
            print("Не найдено аналогов — проверьте фильтры/индексы.")
        return {
            "avg_score": 0.0,
            "same_category_hits": 0,
            "total_analogs": 0,
        }
    
    base_category = normalize_meta(row.get("root_category", ""))
    same_category_hits = 0
    avg_score = 0.0
    
    for _, score, cand in analogs:
        avg_score += score
        cand_category = normalize_meta(cand.get("root_category", ""))
        if base_category and cand_category == base_category:
            same_category_hits += 1
    
    avg_score /= len(analogs)
    
    if verbose:
        method_label = f" ({method_name})" if method_name else ""
        print(f"ПРОКСИ-МЕТРИКИ{method_label}:")
        print(f"- Средний скор: {avg_score:.3f}")
        print(f"- Совпадение категории: {same_category_hits}/{len(analogs)}")
        if same_category_hits == len(analogs):
            print("  (Суть категории сохраняется)")
        else:
            print("  (Есть расхождения по категориям)")
    
    return {
        "avg_score": avg_score,
        "same_category_hits": same_category_hits,
        "total_analogs": len(analogs),
    }


def print_analogs(
    row: pd.Series,
    analogs: List[Tuple[int, float, pd.Series]],
    method_name: str = "",
) -> None:
    # универсальная функция для вывода результатов поиска
    print("\nИСХОДНЫЙ ТОВАР:")
    print("Название:", row.get("title", ""))
    print("Бренд:", row.get("brand", ""))
    print("Категория:", row.get("root_category", ""))
    desc = (row.get("product_description", "") or "")[:300]
    print("Описание:", desc, "...\n")
    
    method_label = f" ({method_name})" if method_name else ""
    print(f"АНАЛОГИ{method_label}:")
    for idx, (candidate_id, score, cand) in enumerate(analogs, start=1):
        print(f"\n#{idx} | score={score:.3f}")
        print("ID строки:", candidate_id)
        print("Название:", cand.get("title", ""))
        print("Бренд:", cand.get("brand", ""))
        print("Категория:", cand.get("root_category", ""))
        cand_desc = (cand.get("product_description", "") or "")[:300]
        print("Описание:", cand_desc, "...")
