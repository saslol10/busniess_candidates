"""Сбор и нормализация признаков по кандидатам из всех доступных источников."""
from decimal import Decimal
import pandas as pd

from staff import search_staff
from trade import find_trade_candidates
from kkt_search import search_kkt

FEATURE_COLUMNS = [
    "ИНН кандидата",
    "Индекс совпадения",
    "Критериев совпало",
    "Сотрудники: общих",
    "Сотрудники: доля, %",
    "Сотрудники: последовательные",
    "Поставщики: общих",
    "Поставщики: покрытие, %",
    "Покупатели: общих",
    "Покупатели: покрытие, %",
    "ККТ: общих",
    "ККТ: доля, %",
]


def extract_candidate_features(frames, source_inn):
    """
    Выполняет поиск по 4 направлениям и собирает единую сводку признаков.
    Возвращает:
      - features_df: таблица признаков по каждому кандидату
      - raw_results: словарь с сырыми объектами результатов для детального просмотра
      - warnings: предупреждения по источникам
    """
    warnings = []
    candidates_data = {}

    def candidate_entry(inn):
        if inn not in candidates_data:
            candidates_data[inn] = {
                "ИНН кандидата": inn,
                "staff_count": 0,
                "staff_share": 0.0,
                "staff_seq": 0,
                "suppliers_count": 0,
                "suppliers_coverage": 0.0,
                "buyers_count": 0,
                "buyers_coverage": 0.0,
                "kkt_count": 0,
                "kkt_share": 0.0,
                "criteria_matched": 0,
            }
        return candidates_data[inn]

    raw_results = {
        "staff": None,
        "suppliers": None,
        "buyers": None,
        "kkt": None,
    }

    # 1. Сотрудники
    if "Сотрудники" in frames and not frames["Сотрудники"].empty:
        try:
            staff_res = search_staff(frames["Сотрудники"], source_inn)
            raw_results["staff"] = staff_res
            if not staff_res["ranking"].empty:
                for _, row in staff_res["ranking"].iterrows():
                    inn = str(row["ИНН кандидата"]).strip()
                    entry = candidate_entry(inn)
                    entry["staff_count"] = int(row["Общие сотрудники"])
                    entry["staff_share"] = float(row["Доля, %"])
                    entry["staff_seq"] = int(row["Последовательные периоды в выгрузке"])
        except Exception as exc:
            warnings.append(f"Поиск по сотрудникам не выполнен: {exc}")
    else:
        warnings.append("Лист «Сотрудники» отсутствует или пуст.")

    # 2. Поставщики
    if "Покупки" in frames and "Продажи" in frames:
        try:
            supp_res = find_trade_candidates(frames, source_inn, "suppliers")
            raw_results["suppliers"] = supp_res
            if not supp_res["ranking"].empty:
                for _, row in supp_res["ranking"].iterrows():
                    inn = str(row["ИНН кандидата"]).strip()
                    entry = candidate_entry(inn)
                    entry["suppliers_count"] = int(row["Общих контрагентов"])
                    entry["suppliers_coverage"] = float(row["Покрытие оборота исходной, %"])
        except Exception as exc:
            warnings.append(f"Поиск по поставщикам не выполнен: {exc}")
    else:
        warnings.append("Для анализа поставщиков требуются листы «Покупки» и «Продажи».")

    # 3. Покупатели
    if "Покупки" in frames and "Продажи" in frames:
        try:
            buy_res = find_trade_candidates(frames, source_inn, "buyers")
            raw_results["buyers"] = buy_res
            if not buy_res["ranking"].empty:
                for _, row in buy_res["ranking"].iterrows():
                    inn = str(row["ИНН кандидата"]).strip()
                    entry = candidate_entry(inn)
                    entry["buyers_count"] = int(row["Общих контрагентов"])
                    entry["buyers_coverage"] = float(row["Покрытие оборота исходной, %"])
        except Exception as exc:
            warnings.append(f"Поиск по покупателям не выполнен: {exc}")

    # 4. ККТ
    if "ККТ" in frames and not frames["ККТ"].empty:
        try:
            kkt_rank, kkt_det, kkt_warn = search_kkt(frames, source_inn)
            raw_results["kkt"] = {"ranking": kkt_rank, "details": kkt_det, "warnings": kkt_warn}
            if not kkt_rank.empty:
                for _, row in kkt_rank.iterrows():
                    inn = str(row["ИНН кандидата"]).strip()
                    entry = candidate_entry(inn)
                    entry["kkt_count"] = int(row["Общие ККТ"])
                    entry["kkt_share"] = float(row["Доля совпадения, %"])
        except Exception as exc:
            warnings.append(f"Поиск по ККТ не выполнен: {exc}")
    else:
        warnings.append("Лист «ККТ» отсутствует или пуст.")

    rows = []
    for inn, data in candidates_data.items():
        matched = 0
        if data["staff_count"] > 0:
            matched += 1
        if data["suppliers_count"] > 0:
            matched += 1
        if data["buyers_count"] > 0:
            matched += 1
        if data["kkt_count"] > 0:
            matched += 1
        data["criteria_matched"] = matched

        rows.append({
            "ИНН кандидата": inn,
            "Критериев совпало": matched,
            "Сотрудники: общих": data["staff_count"],
            "Сотрудники: доля, %": data["staff_share"],
            "Сотрудники: последовательные": data["staff_seq"],
            "Поставщики: общих": data["suppliers_count"],
            "Поставщики: покрытие, %": data["suppliers_coverage"],
            "Покупатели: общих": data["buyers_count"],
            "Покупатели: покрытие, %": data["buyers_coverage"],
            "ККТ: общих": data["kkt_count"],
            "ККТ: доля, %": data["kkt_share"],
        })

    frame = pd.DataFrame(rows)
    return frame, raw_results, warnings
