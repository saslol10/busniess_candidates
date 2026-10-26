from __future__ import annotations

from collections import defaultdict
import pandas as pd


def _text(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _inn(value):
    value = _text(value)
    return value if value.isdigit() and len(value) == 12 else ""


def _empty_details():
    return pd.DataFrame(columns=["ИНН физлица", "ФИО", "В исходной организации", "В организации-кандидате", "Основание"])


def search_roles(frames, source_inn):
    """Compare people by personal INN across employees, founders and managers."""
    source_inn = _text(source_inn)
    staff = frames.get("Сотрудники")
    orgs = frames.get("Организации")
    warnings = []
    if staff is None:
        raise ValueError("В книге нет листа «Сотрудники».")
    if orgs is None:
        raise ValueError("В книге нет листа «Организации» с учредителями и руководителями.")
    if "ИНН сотрудника" not in staff.columns:
        raise ValueError("На листе «Сотрудники» нет колонки «ИНН сотрудника». Добавьте ИНН физических лиц.")
    required_roles = ["ИНН учредителя", "ИНН руководителя"]
    absent = [c for c in required_roles if c not in orgs.columns]
    if absent:
        raise ValueError("На листе «Организации» нет колонок: " + ", ".join(absent) + ".")
    if "ФИО сотрудника" not in staff.columns:
        warnings.append("Не найдена колонка «ФИО сотрудника»: в основаниях будут показаны только ИНН.")
    if "ФИО учредителя" not in orgs.columns or "ФИО руководителя" not in orgs.columns:
        warnings.append("Не найдены все колонки ФИО: сопоставление выполняется только по ИНН.")

    people_names = {}
    def remember(person, fio):
        person = _inn(person)
        fio = _text(fio)
        if person and fio and person not in people_names:
            people_names[person] = fio

    employees = defaultdict(set)
    for _, row in staff.iterrows():
        org = _text(row.get("ИНН организации"))
        person = _inn(row.get("ИНН сотрудника"))
        if org and person:
            employees[org].add(person)
            remember(person, row.get("ФИО сотрудника", ""))

    founders = defaultdict(set)
    managers = defaultdict(set)
    for _, row in orgs.iterrows():
        org = _text(row.get("ИНН организации"))
        if not org:
            continue
        founder = _inn(row.get("ИНН учредителя"))
        manager = _inn(row.get("ИНН руководителя"))
        if founder:
            founders[org].add(founder)
            remember(founder, row.get("ФИО учредителя", ""))
        if manager:
            managers[org].add(manager)
            remember(manager, row.get("ФИО руководителя", ""))

    if not employees.get(source_inn) and not founders.get(source_inn) and not managers.get(source_inn):
        raise ValueError("Для исходной организации не найдено ни сотрудников, ни учредителей, ни руководителей с корректным ИНН физлица.")

    source = {"employees": employees[source_inn], "founders": founders[source_inn], "managers": managers[source_inn]}
    organizations = set(employees) | set(founders) | set(managers)
    names = {}
    if "Наименование организации" in orgs.columns:
        for _, row in orgs.iterrows():
            org = _text(row.get("ИНН организации"))
            name = _text(row.get("Наименование организации"))
            if org and name:
                names.setdefault(org, name)
    for frame in frames.values():
        if "ИНН организации" in frame.columns:
            for _, row in frame.iterrows():
                org = _text(row.get("ИНН организации"))
                if org:
                    name = _text(row.get("Наименование организации", ""))
                    if name:
                        names.setdefault(org, name)

    rows, details = [], {}
    detail_columns = ["ИНН физлица", "ФИО", "В исходной организации", "В организации-кандидате", "Основание"]
    for candidate in sorted(organizations):
        if candidate == source_inn:
            continue
        groups = [
            ("Сотрудник", "Учредитель", source["employees"] & founders[candidate], "сотрудник исходной организации — учредитель кандидата"),
            ("Сотрудник", "Руководитель", source["employees"] & managers[candidate], "сотрудник исходной организации — руководитель кандидата"),
            ("Учредитель", "Сотрудник", source["founders"] & employees[candidate], "учредитель исходной организации — сотрудник кандидата"),
            ("Руководитель", "Сотрудник", source["managers"] & employees[candidate], "руководитель исходной организации — сотрудник кандидата"),
            ("Учредитель", "Учредитель", source["founders"] & founders[candidate], "общий учредитель"),
            ("Руководитель", "Руководитель", source["managers"] & managers[candidate], "общий руководитель"),
        ]
        drows = []
        counts = {}
        for source_role, candidate_role, people, reason in groups:
            key = source_role.lower().replace(" ", "_") + "_в_" + candidate_role.lower().replace(" ", "_")
            counts[key] = len(people)
            for person in sorted(people):
                drows.append({"ИНН физлица": person, "ФИО": people_names.get(person, ""),
                              "В исходной организации": source_role, "В организации-кандидате": candidate_role,
                              "Основание": reason})
        total = sum(counts.values())
        if not total:
            continue
        rows.append({"ИНН кандидата": candidate, "Наименование кандидата": names.get(candidate, ""),
                     "Сотрудник → учредитель": counts["сотрудник_в_учредитель"],
                     "Сотрудник → руководитель": counts["сотрудник_в_руководитель"],
                     "Учредитель → сотрудник": counts["учредитель_в_сотрудник"],
                     "Руководитель → сотрудник": counts["руководитель_в_сотрудник"],
                     "Общие учредители": counts["учредитель_в_учредитель"],
                     "Общие руководители": counts["руководитель_в_руководитель"],
                     "Всего совпадений": total})
        details[candidate] = pd.DataFrame(drows, columns=detail_columns)

    ranking_columns = ["ИНН кандидата", "Наименование кандидата", "Сотрудник → учредитель", "Сотрудник → руководитель",
                       "Учредитель → сотрудник", "Руководитель → сотрудник", "Общие учредители", "Общие руководители", "Всего совпадений"]
    ranking = pd.DataFrame(rows, columns=ranking_columns)
    if not ranking.empty:
        ranking = ranking.sort_values(["Всего совпадений", "ИНН кандидата"], ascending=[False, True], kind="stable").reset_index(drop=True)
    summary = (f"ИНН исходной организации: {source_inn} · сотрудников: {len(source['employees'])} · "
               f"учредителей: {len(source['founders'])} · руководителей: {len(source['managers'])} · "
               f"кандидатов с ролевыми совпадениями: {len(ranking)}")
    if not rows:
        warnings.append("Совпадений по ИНН физических лиц не найдено.")
    return {"ranking": ranking, "details": details, "empty_details": _empty_details(), "warnings": warnings, "summary": summary}
