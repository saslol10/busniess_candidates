"""Объединение рассчитанных признаков."""
from __future__ import annotations
import math
import re
from numbers import Integral
from typing import Optional
import pandas as pd

# Имя выходного признака, столбец существующего модуля, тип.
SPECS = {
    "staff": [
        ("staff_common_count", "Общие сотрудники", "count"),
        ("staff_source_share_pct", "Доля, %", "percent"),
        ("staff_regular_common_count", "Регулярные у обеих организаций", "count"),
        ("staff_sequential_count", "Последовательные периоды в выгрузке", "count"),
    ],
    "suppliers": [
        ("suppliers_common_count", "Общих контрагентов", "count"),
        ("suppliers_source_count_share_pct", "Доля контрагентов исходной, %", "percent"),
        ("suppliers_source_turnover_share_pct", "Покрытие оборота исходной, %", "percent"),
        ("suppliers_common_source_amount", "Оборот исходной с общими, с НДС", "amount"),
        ("suppliers_common_candidate_amount", "Оборот кандидата с общими, с НДС", "amount"),
    ],
    "buyers": [
        ("buyers_common_count", "Общих контрагентов", "count"),
        ("buyers_source_count_share_pct", "Доля контрагентов исходной, %", "percent"),
        ("buyers_source_turnover_share_pct", "Покрытие оборота исходной, %", "percent"),
        ("buyers_common_source_amount", "Оборот исходной с общими, с НДС", "amount"),
        ("buyers_common_candidate_amount", "Оборот кандидата с общими, с НДС", "amount"),
    ],
    "kkt": [
        ("kkt_common_count", "Общие ККТ", "count"),
        ("kkt_source_share_pct", "Доля совпадения, %", "percent"),
    ],
}
# Ролевые направления (учредители/руководители). Отдельно от SPECS, чтобы схема v1
# и ранее обученные модели CatBoost оставались неизменными.
ROLE_SPECS = {
    "founders": [
        ("founders_common_persons", "Совпавших лиц", "count"),
        ("founders_basis_count", "Оснований", "count"),
    ],
    "directors": [
        ("directors_common_persons", "Совпавших лиц", "count"),
        ("directors_basis_count", "Оснований", "count"),
    ],
}
ALL_SPECS = {**SPECS, **ROLE_SPECS}
ID_COL = "ИНН кандидата"
FEATURE_COLUMNS = [
    column
    for group, specs in SPECS.items()
    for column in [group + "_search_completed", group + "_candidate_found"]
        + [item[0] for item in specs]
]
ROLE_FEATURE_COLUMNS = [
    column
    for group, specs in ROLE_SPECS.items()
    for column in [group + "_search_completed", group + "_candidate_found"]
        + [item[0] for item in specs]
]
# FEATURE_COLUMNS — схема v1 (4 направления), ALL_FEATURE_COLUMNS — схема v2 (+ роли).
ALL_FEATURE_COLUMNS = FEATURE_COLUMNS + ROLE_FEATURE_COLUMNS
OUTPUT_COLUMNS = ["source_inn", "candidate_inn"] + ALL_FEATURE_COLUMNS


def _inn(value, context):
    if not isinstance(value, str):
        raise ValueError(f"{context}: ИНН должен быть текстом, а не числом")
    value = value.strip()
    if not re.fullmatch(r"[0-9]{10}", value) or value == "0" * 10:
        raise ValueError(f"{context}: нужен 10-значный ИНН организации")
    return value


def _number(value, kind, context):
    if isinstance(value, bool) or pd.isna(value):
        raise ValueError(f"{context}: пустое или недопустимое значение")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{context}: требуется число") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{context}: требуется конечное неотрицательное число")
    if kind == "percent" and result > 100:
        raise ValueError(f"{context}: процент должен быть от 0 до 100")
    if kind == "count" and not result.is_integer():
        raise ValueError(f"{context}: количество должно быть целым")
    return int(result) if kind == "count" else result


def _prepare(frame, group, source_inn):
    if frame is None:
        return None
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{group}: ожидается pandas.DataFrame либо None")
    if not frame.columns.is_unique:
        raise ValueError(f"{group}: повторяющиеся заголовки столбцов")
    if frame.empty:
        return {}
    required = [ID_COL] + [item[1] for item in ALL_SPECS[group]]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"{group}: отсутствуют столбцы: {', '.join(missing)}")
    result = {}
    for row in frame.to_dict("records"):
        candidate = _inn(row[ID_COL], f"{group}, {ID_COL}")
        if candidate == source_inn:
            continue
        if candidate in result:
            raise ValueError(f"{group}: кандидат {candidate} встречается несколько раз")
        result[candidate] = {
            feature: _number(row[column], kind, f"{group}, {candidate}, {column}")
            for feature, column, kind in ALL_SPECS[group]
        }
    return result


def build_candidate_features(
    source_inn: str,
    *,
    staff: Optional[pd.DataFrame] = None,
    suppliers: Optional[pd.DataFrame] = None,
    buyers: Optional[pd.DataFrame] = None,
    kkt: Optional[pd.DataFrame] = None,
    founders: Optional[pd.DataFrame] = None,
    directors: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Объединить полные результаты четырёх поисков без вычисления рейтинга.

    None: поиск по направлению не выполнен, завершился ошибкой или нет данных.
    Пустой DataFrame: поиск успешно выполнен, совпадений не найдено.
    Непустой DataFrame: полный ranking соответствующего поискового модуля.
    Не передавайте только верхние строки рейтинга или отфильтрованный экран.

    Все поиски должны относиться к одному ИНН и согласованному анализируемому
    случаю. Периоды, версии данных и ссылки на исходные записи сохраняются
    вызывающим кодом вместе с этой таблицей. Сам DataFrame их не подтверждает.
    """
    source_inn = _inn(source_inn, "Исходная организация")
    prepared = {
        group: _prepare(frame, group, source_inn)
        for group, frame in dict(
            staff=staff, suppliers=suppliers, buyers=buyers, kkt=kkt,
            founders=founders, directors=directors
        ).items()
    }
    candidates = sorted(set().union(*(set(rows or {}) for rows in prepared.values())))
    output = []
    for candidate in candidates:
        row = {"source_inn": source_inn, "candidate_inn": candidate}
        for group, rows in prepared.items():
            completed = rows is not None
            found = completed and candidate in rows
            row[group + "_search_completed"] = int(completed)
            row[group + "_candidate_found"] = int(found) if completed else float("nan")
            for feature, _, _ in ALL_SPECS[group]:
                row[feature] = (
                    rows[candidate][feature] if found
                    else 0 if completed
                    else float("nan")
                )
        output.append(row)
    result = pd.DataFrame(output, columns=OUTPUT_COLUMNS)
    for column in ALL_FEATURE_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="raise").astype("float64")
    return result
