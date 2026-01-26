"""Component text builder."""

import pandas as pd


def build_component(row: pd.Series) -> str:
    # собираем ключевые поля товара в единый текст
    brand = row.get("brand", "")
    title = row.get("title", "")
    category = row.get("root_category", "")
    desc = row.get("product_description", "")
    features = row.get("features_summary", "")
    specs = row.get("product_specifications", "")

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
