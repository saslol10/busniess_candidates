"""Обучение CatBoost. ИНН/даты/метки/комментарии не входят в признаки."""
from pathlib import Path
from datetime import datetime, timezone, date
from uuid import uuid4
import json
import os
import re
import tempfile
import numpy as np
import pandas as pd
from candidate_features import FEATURE_COLUMNS
from catboost_ranker import SCHEMA_VERSION, LABEL, prepare_features, digest


def _date(value):
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    s = str(value).strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        raise ValueError("Для размеченных строк заполните period_start/period_end в формате ГГГГ-ММ-ДД.")
    return date.fromisoformat(s).isoformat()


def read_labels(paths):
    parts, total, skipped = [], 0, 0
    for path in paths:
        with pd.ExcelFile(path, engine="openpyxl") as book:
            sheet = next((s for s in ("Для_разметки_CatBoost", "Разметка") if s in book.sheet_names), None)
            if sheet is None:
                raise ValueError(str(path) + ": нет листа Для_разметки_CatBoost / Разметка.")
            frame = pd.read_excel(book, sheet_name=sheet, dtype=object)
        label = LABEL if LABEL in frame.columns else "target"
        required = ["source_inn", "candidate_inn", "period_start", "period_end", label] + list(FEATURE_COLUMNS)
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise ValueError(str(path) + ": отсутствуют столбцы " + ", ".join(missing))
        total += len(frame)
        blank = frame[label].isna() | frame[label].astype(str).str.strip().eq("")
        skipped += int(blank.sum())
        frame = frame.loc[~blank].copy().reset_index(drop=True)
        if frame.empty:
            continue
        frame["_target"] = pd.to_numeric(frame[label], errors="raise")
        if not frame["_target"].isin([0, 1]).all():
            raise ValueError("Метки: только 0, 1 или пустая ячейка.")
        frame["_target"] = frame["_target"].astype(int)
        for c in ("source_inn", "candidate_inn"):
            if not frame[c].map(lambda v: isinstance(v, str) and bool(
                    re.fullmatch(r"[0-9]{10}", v.strip())) and v.strip() != "0" * 10).all():
                raise ValueError(c + ": ИНН должны быть текстовыми 10-значными идентификаторами.")
            frame[c] = frame[c].str.strip()
        if frame["source_inn"].eq(frame["candidate_inn"]).any():
            raise ValueError("Исходная организация не может совпадать с кандидатом.")
        for c in ("period_start", "period_end"):
            frame[c] = frame[c].map(_date)
        if (frame["period_start"] > frame["period_end"]).any():
            raise ValueError("Начало периода позже окончания.")
        numeric = prepare_features(frame)
        for c in FEATURE_COLUMNS:
            frame[c] = numeric[c]
        parts.append(frame)
    if not parts:
        raise ValueError("Нет строк с экспертными метками 0/1. Пустые метки пропущены.")
    data = pd.concat(parts, ignore_index=True)
    key = ["source_inn", "candidate_inn", "period_start", "period_end"]
    for _, group in data.groupby(key, sort=False):
        if len(group[["_target"] + list(FEATURE_COLUMNS)].drop_duplicates()) != 1:
            raise ValueError("Для одной пары и периода найдены противоречивые метки или признаки. Исправьте разметку.")
    before = len(data)
    data = data.drop_duplicates(key).reset_index(drop=True)
    if set(data["_target"]) != {0, 1}:
        raise ValueError("Для классификатора необходимы подтверждённые примеры ОБОИХ классов: 0 и 1.")
    return data, {"input_rows": total, "unlabeled_skipped": skipped, "duplicates_removed": before - len(data)}


def entity_groups(data):
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(data["source_inn"], data["candidate_inn"]):
        parent[find(b)] = find(a)
    return np.array([find(a) for a in data["source_inn"]])


def split_data(data, seed=42):
    from sklearn.model_selection import GroupShuffleSplit
    groups = entity_groups(data)
    n = len(set(groups))
    if n < 6:
        raise ValueError("Для раздельных обучения, валидации и теста нужно не менее 6 независимых групп организаций. "
                         "Общий ИНН у пар связывает их в одну группу. Нужны оба класса в каждой части. "
                         "Для технической проверки доступен только явный --allow-unvalidated.")
    y = data["_target"].to_numpy()
    for trainval, test in GroupShuffleSplit(n_splits=80, test_size=max(.2, 2 / n), random_state=seed).split(data, y,
                                                                                                            groups):
        if len(set(y[test])) != 2:
            continue
        remaining = groups[trainval]
        for tr, va in GroupShuffleSplit(n_splits=30, test_size=max(.25, 2 / len(set(remaining))),
                                        random_state=seed + 1).split(trainval, y[trainval], remaining):
            train, val = trainval[tr], trainval[va]
            if len(set(y[train])) == len(set(y[val])) == 2:
                return train, val, test, groups
    raise ValueError(
        "Не удалось получить непересекающиеся по организациям обучение, валидацию и тест с обоими классами. "
        "Соберите больше независимых подтверждённых случаев. Для технического опыта: --allow-unvalidated.")


def metrics(y, scores):
    from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
    return {"rows": int(len(y)), "positive": int(np.sum(y)),
            "roc_auc": float(roc_auc_score(y, scores)),
            "average_precision": float(average_precision_score(y, scores)),
            "log_loss": float(log_loss(y, scores, labels=[0, 1])),
            "brier_score": float(brier_score_loss(y, scores))}


def ranking_metrics(data, scores):
    q = data[["source_inn", "_target"]].copy()
    q["score"] = scores
    result = {}
    cases = [g.sort_values("score", ascending=False, kind="mergesort") for _, g in q.groupby("source_inn") if
             g["_target"].sum() > 0]
    result["source_cases_with_positives"] = len(cases)
    for k in (3, 5, 10):
        result[f"mean_recall_at_{k}"] = float(
            np.mean([g.head(k)["_target"].sum() / g["_target"].sum() for g in cases])) if cases else None
    return result


def train_and_save(paths, output_dir, allow_unvalidated=False, iterations=400, seed=42):
    from catboost import CatBoostClassifier, __version__
    if iterations < 10 or iterations > 10000:
        raise ValueError("iterations: допустимо от 10 до 10000.")
    data, counts = read_labels(paths)
    x, y = prepare_features(data), data["_target"].to_numpy()
    report = {"schema_version": SCHEMA_VERSION, **counts, "labeled_rows": len(data),
              "class_counts": {str(k): int((y == k).sum()) for k in (0, 1)},
              "calibrated": False, "positive_label": 1,
              "limitation": "Оценка положительного класса не является доказательством перехода бизнеса. Калибровка не выполнена."}
    try:
        train, val, test, groups = split_data(data, seed)
        report["validation_status"] = "held_out_test"
    except ValueError as exc:
        if not allow_unvalidated:
            raise
        train = np.arange(len(data))
        val, test, groups = np.array([], dtype=int), np.array([], dtype=int), entity_groups(data)
        report.update(validation_status="unvalidated", warning=str(exc))
    # Более сдержанная конфигурация для малой учебной выборки.
    # Не ограничивает оценки искусственно и не гарантирует заданный диапазон.
    model = CatBoostClassifier(iterations=iterations, depth=3, learning_rate=.03,
                               l2_leaf_reg=20.0, random_strength=1.0,
                               loss_function="Logloss", eval_metric="Logloss",
                               auto_class_weights="Balanced", random_seed=seed,
                               thread_count=2, allow_writing_files=False, verbose=False)
    fit_kwargs = {"eval_set": (x.iloc[val], y[val]), "early_stopping_rounds": 40, "use_best_model": True} if len(
        val) else {}
    model.fit(x.iloc[train], y[train], **fit_kwargs)
    report["split"] = {}
    for name, indices in (("train", train), ("validation", val), ("test", test)):
        report["split"][name] = {"rows": int(len(indices)), "entity_groups": len(set(groups[indices])),
                                 "positive": int(y[indices].sum()), "negative": int((y[indices] == 0).sum())}
    if len(test):
        test_scores = model.predict_proba(x.iloc[test])[:, 1]
        report["validation_metrics"] = metrics(y[val], model.predict_proba(x.iloc[val])[:, 1])
        report["test_metrics"] = metrics(y[test], test_scores)
        report["test_ranking_metrics"] = ranking_metrics(data.iloc[test], test_scores)
        report[
            "evaluation_note"] = "Валидация используется для ранней остановки; тест не используется при обучении. После просмотра теста не подбирайте по нему параметры."
    report["selection_metric"] = "Logloss"
    report["best_iteration"] = int(model.get_best_iteration())
    report["training_profile"] = "conservative_demo_v1"
    report["regularization"] = {"depth": 3, "learning_rate": 0.03,
                                "l2_leaf_reg": 20.0, "random_strength": 1.0}
    report["score_note"] = (
        "Некалиброванная оценка положительного класса. Более сильная регуляризация "
        "не гарантирует диапазон 0.8–0.9 и не является калибровкой. "
        "Взвешивание классов также влияет на интерпретацию оценки."
    )
    report["feature_importance"] = {c: float(v) for c, v in zip(FEATURE_COLUMNS, model.feature_importances_)}
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    report.update(model_version=version, catboost_version=__version__, seed=seed,
                  iterations_requested=iterations, trees=int(model.tree_count_))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        temp = Path(tmp)
        model.save_model(str(temp / "catboost_model.cbm"))
        manifest = {"schema_version": SCHEMA_VERSION, "model_version": version,
                    "features": list(FEATURE_COLUMNS), "positive_label": 1,
                    "model_sha256": digest(temp / "catboost_model.cbm"), "catboost_version": __version__,
                    "validation_status": report["validation_status"], "calibrated": False}
        (temp / "catboost_model.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (temp / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        for name in ("catboost_model.cbm", "training_report.json", "catboost_model.json"):
            os.replace(temp / name, out / name)
    return {"model_path": str((out / "catboost_model.cbm").resolve()), "report": report}
