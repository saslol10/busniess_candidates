"""Добавление проверяемых организаций без подстановки баллов CatBoost."""
import copy
import re
import pandas as pd
from candidate_features import ALL_SPECS as SPECS, ALL_FEATURE_COLUMNS as FEATURE_COLUMNS
from catboost_ranker import score_candidates


def normal_inn(value):
    text = str(value).strip()
    if re.fullmatch(r"[0-9]{10}\.0", text):
        text = text[:-2]
    return text


def present_organizations(frames):
    found = set()
    for frame in frames.values():
        if isinstance(frame, pd.DataFrame) and "ИНН организации" in frame:
            found.update(normal_inn(x) for x in frame["ИНН организации"].dropna())
    return found


def augment_result(result, requested, model_path=None):
    output = dict(result)
    features = result["features"].copy(deep=True)
    source = str(result["metadata"]["source_inn"])
    states = result["states"].set_index("Код")["Статус"].to_dict()
    existing = set(features["candidate_inn"].astype(str))
    extra = []
    for inn in sorted(set(requested) - existing):
        if inn == source:
            continue
        row = {"source_inn": source, "candidate_inn": inn}
        for group, specs in SPECS.items():
            completed = states.get(group) == "completed"
            row[group + "_search_completed"] = float(completed)
            row[group + "_candidate_found"] = 0.0 if completed else float("nan")
            for feature, _, _ in specs:
                row[feature] = 0.0 if completed else float("nan")
        extra.append(row)
    if extra:
        features = pd.concat([features, pd.DataFrame(extra, columns=features.columns)], ignore_index=True)
    for column in FEATURE_COLUMNS:
        if column in features:
            features[column] = pd.to_numeric(features[column], errors="raise").astype(float)
    output["features"] = features.sort_values("candidate_inn", kind="stable").reset_index(drop=True)
    output["metadata"] = copy.deepcopy(result["metadata"])
    output["metadata"]["manually_requested_candidates"] = sorted(set(requested))
    output["warnings"] = list(result.get("warnings", []))
    if requested:
        output["warnings"].append("Ручное сравнение: " + ", ".join(sorted(requested)) +
                                  ". Нулевые совпадения относятся только к успешно проверенным направлениям загруженной книги; "
                                  "непроверенные направления остаются пропусками. Отсутствие совпадений не доказывает отсутствие связи.")
    # После изменения состава списка прежние оценки и места устарели.
    ranked, cb_info = score_candidates(output["features"], model_path=model_path)
    output["features"] = ranked
    output["metadata"]["manual_catboost"] = cb_info
    if cb_info.get("warning"):
        output["warnings"].append(
            "CatBoost при обновлении ручного сравнения: " + cb_info["warning"]
        )
    return output
