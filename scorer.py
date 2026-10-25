"""Оценка приоритетности кандидатов: экспертный бейзлайн либо CatBoost."""
from pathlib import Path
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).with_name("catboost_model.cbm")

FEATURE_NAMES = [
    "Сотрудники: доля, %",
    "Сотрудники: последовательные",
    "Поставщики: покрытие, %",
    "Покупатели: покрытие, %",
    "ККТ: доля, %",
    "Критериев совпало",
]


def calculate_expert_score(row):
    """
    Прозрачный экспертный индекс приоритетности (от 0 до 100).
    Не является вероятностью, показывает совокупную выраженность связей.
    """
    staff_score = min(100.0, row["Сотрудники: доля, %"] * 0.8 + row["Сотрудники: последовательные"] * 5.0)
    supp_score = row["Поставщики: покрытие, %"]
    buy_score = row["Покупатели: покрытие, %"]
    kkt_score = row["ККТ: доля, %"]

    base = (
        0.30 * staff_score +
        0.25 * supp_score +
        0.25 * buy_score +
        0.20 * kkt_score
    )

    # Синергетический бонус за совпадение сразу нескольких независимых каналов
    matched = row["Критериев совпало"]
    multiplier = 1.0
    if matched >= 4:
        multiplier = 1.25
    elif matched == 3:
        multiplier = 1.15
    elif matched == 2:
        multiplier = 1.05

    final_score = min(100.0, base * multiplier)
    return round(final_score, 1)


def score_candidates(features_df):
    """
    Применяет модель CatBoost (если файл существует) или экспертный расчёт.
    Возвращает (обновленный DataFrame, название применённого режима).
    """
    if features_df.empty:
        features_df["Индекс совпадения"] = []
        return features_df, "Нет кандидатов"

    df = features_df.copy()

    if MODEL_PATH.exists():
        try:
            from catboost import CatBoostClassifier
            model = CatBoostClassifier()
            model.load_model(str(MODEL_PATH))

            X = df[FEATURE_NAMES]
            probabilities = model.predict_proba(X)[:, 1]
            df["Индекс совпадения"] = np.round(probabilities * 100, 1)
            mode = "CatBoost ML (файл catboost_model.cbm)"
        except Exception as exc:
            df["Индекс совпадения"] = df.apply(calculate_expert_score, axis=1)
            mode = f"Экспертный бейзлайн (ошибка загрузки CatBoost: {exc})"
    else:
        df["Индекс совпадения"] = df.apply(calculate_expert_score, axis=1)
        mode = "Экспертный бейзлайн (модель CatBoost не обучена)"

    # Сортировка: сначала высокий индекс и количество совпавших каналов
    df = df.sort_values(
        by=["Индекс совпадения", "Критериев совпало", "ИНН кандидата"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    # Перемещаем главные колонки в начало таблицы
    first_cols = ["ИНН кандидата", "Индекс совпадения", "Критериев совпало"]
    other_cols = [c for c in df.columns if c not in first_cols]
    df = df[first_cols + other_cols]

    return df, mode