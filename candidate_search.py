"""Общий поиск без Qt, CatBoost и языковой модели. Исходные модули не меняются."""
from datetime import datetime, timezone
from uuid import uuid4
import pandas as pd
from core import valid_inn
from staff import search_staff
from trade import find_trade_candidates
from kkt_search import search_kkt
from candidate_features import build_candidate_features

DIRECTIONS = {"staff": "Сотрудники", "suppliers": "Поставщики", "buyers": "Покупатели", "kkt": "ККТ"}
DEPENDENCIES = {"staff": ("Сотрудники",), "suppliers": ("Покупки", "Продажи"),
                "buyers": ("Продажи", "Покупки"), "kkt": ("ККТ",)}
SOURCE_SHEETS = {"staff": "Сотрудники", "suppliers": "Покупки", "buyers": "Продажи", "kkt": "ККТ"}


def run_candidate_search(frames, source_inn, progress=None):
    if not valid_inn(source_inn, (10,)):
        raise ValueError("Нужен корректный 10-значный ИНН исходной организации.")
    rankings = dict.fromkeys(DIRECTIONS)
    raw, states, warnings = {}, [], []
    metadata = {"analysis_id": str(uuid4()), "source_inn": source_inn,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "period": "Весь загруженный период; без ограничений дат во всех направлениях",
                "method": "Объединение полных результатов существующих поисков. Без ИИ и рейтинга.",
                "input_rows": {name: len(frame) for name, frame in frames.items()}}
    for group, title in DIRECTIONS.items():
        if progress:
            progress(title)
        state, message, count = "no_data", "", None
        try:
            absent = [name for name in DEPENDENCIES[group] if name not in frames or frames[name] is None or frames[name].empty]
            if absent:
                message = "Отсутствуют или пусты листы: " + ", ".join(absent)
            else:
                source = frames[SOURCE_SHEETS[group]]
                if "ИНН организации" not in source.columns:
                    raise ValueError("Отсутствует столбец «ИНН организации».")
                if not source["ИНН организации"].astype(str).str.strip().eq(source_inn).any():
                    message = "Нет записей исходной организации в листе «" + SOURCE_SHEETS[group] + "»."
                else:
                    if group == "staff":
                        result = search_staff(frames["Сотрудники"], source_inn)
                    elif group in ("suppliers", "buyers"):
                        result = find_trade_candidates(frames, source_inn, group)
                    else:
                        ranking, details, notes = search_kkt(frames, source_inn)
                        result = {"ranking": ranking, "details": details, "warnings": notes}
                    raw[group] = result
                    warnings.extend(title + ": " + str(note) for note in result.get("warnings", []))
                    unusable = group == "staff" and result.get("source_count", 0) == 0
                    if group in ("suppliers", "buyers"):
                        unusable = result["source_rows"].empty
                    if group == "kkt":
                        serials = source.loc[source["ИНН организации"].astype(str).str.strip().eq(source_inn), "Заводской номер"]
                        unusable = not serials.fillna("").astype(str).str.strip().ne("").any()
                    if unusable:
                        message = "У исходной организации нет пригодных записей для сравнения; см. ограничения."
                    else:
                        ranking = result["ranking"]
                        # Проверяем контракт каждого направления отдельно, чтобы ошибка одного
                        # модуля не уничтожила корректные результаты остальных.
                        build_candidate_features(source_inn, **{group: ranking})
                        rankings[group] = ranking
                        state, count = "completed", len(ranking)
                        message = "Проверено; совпадений нет." if ranking.empty else "Проверено; кандидаты найдены."
                        if group == "staff" and result.get("skipped"):
                            warnings.append(f"Сотрудники: пропущено строк — {result['skipped']}.")
        except Exception as exc:
            rankings[group] = None
            state, message = "error", f"{type(exc).__name__}: {exc}"
        states.append({"Направление": title, "Код": group, "Статус": state,
                       "Кандидатов": count, "Сообщение": message})
    features = build_candidate_features(source_inn, **rankings)
    metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    return {"features": features, "states": pd.DataFrame(states), "raw": raw,
            "warnings": warnings, "metadata": metadata}
