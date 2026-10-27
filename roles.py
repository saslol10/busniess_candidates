"""Поиск кандидатов по учредителям и по руководителям исходной организации.

Лица исходной организации (учредители ИЛИ руководители) ищутся в других
организациях среди учредителей, руководителей и сотрудников. Совпадение
определяется только по ИНН физлица; совпадение одного ФИО не учитывается.
Модуль не меняет признаки CatBoost и контекст Qwen.
"""
import pandas as pd
from core import clean, valid_inn
from staff import month_number, month_label

FOUNDER, DIRECTOR, STAFF = "учредитель", "руководитель", "сотрудник"
MODES = {"founders": FOUNDER, "directors": DIRECTOR}
TITLES = {"founders": "Поиск по учредителям", "directors": "Поиск по руководителям"}
_INSTR = {FOUNDER: "учредителем", DIRECTOR: "руководителем", STAFF: "сотрудником"}
DETAIL_COLS = ["ИНН кандидата", "Наименование кандидата", "ИНН физлица", "ФИО",
               "Роль в исходной организации", "Роль в организации-кандидате",
               "Период (для сотрудников)", "Основание"]
RANK_COLS = ["Место", "ИНН кандидата", "Наименование кандидата", "Совпавших лиц",
             "Оснований", "Роли в кандидате"]


def role_records(frames):
    """Плоский список: организация, ИНН физлица, ФИО, роль, период."""
    rows, names = [], {}
    org = frames.get("Организации")
    if org is not None and not org.empty:
        for _, r in org.iterrows():
            o = clean(r.get("ИНН организации", ""))
            name = clean(r.get("Наименование организации", ""))
            if name:
                names.setdefault(o, name)
            for role, fio_col, inn_col in ((FOUNDER, "ФИО учредителя", "ИНН учредителя"),
                                           (DIRECTOR, "ФИО руководителя", "ИНН руководителя")):
                p = clean(r.get(inn_col, ""))
                if p:
                    rows.append({"org": o, "person": p, "fio": clean(r.get(fio_col, "")), "role": role, "period": ""})
    staff = frames.get("Сотрудники")
    if staff is not None and not staff.empty and "ИНН сотрудника" in staff.columns:
        s = staff.assign(_p=staff["ИНН сотрудника"].map(clean))
        s = s[s["_p"] != ""]
        for (o, p), g in s.groupby(["ИНН организации", "_p"]):
            nums = sorted(n for n in g["Месяц"].map(month_number) if n is not None)
            fio = next((clean(x) for x in g.get("ФИО сотрудника", []) if clean(x)), "")
            period = f"{month_label(nums[0])} – {month_label(nums[-1])}" if nums else ""
            rows.append({"org": clean(o), "person": p, "fio": fio, "role": STAFF, "period": period})
    frame = pd.DataFrame(rows, columns=["org", "person", "fio", "role", "period"]).drop_duplicates()
    return frame, names


def search_roles(frames, source_inn, mode):
    if mode not in MODES:
        raise ValueError("Неизвестный режим поиска: " + str(mode))
    src = clean(source_inn)
    if not valid_inn(src, (10,)):
        raise ValueError("Введите корректный 10-значный ИНН исходной организации.")
    if frames.get("Организации") is None:
        raise ValueError("В файле нет листа «Организации» с учредителями и руководителями.")
    role = MODES[mode]
    rec, names = role_records(frames)
    source = rec[(rec["org"] == src) & (rec["role"] == role)].drop_duplicates("person")
    empty = {"ranking": pd.DataFrame(columns=RANK_COLS), "details": pd.DataFrame(columns=DETAIL_COLS),
             "source_persons": len(source), "mode": mode}
    if source.empty:
        return {**empty, "message": f"У организации {src} в листе «Организации» не указан {role} с ИНН."}
    other = rec[(rec["org"] != src) & rec["person"].isin(source["person"])]
    details = []
    for _, s in source.iterrows():
        for _, c in other[other["person"] == s["person"]].iterrows():
            details.append({
                "ИНН кандидата": c["org"], "Наименование кандидата": names.get(c["org"], ""),
                "ИНН физлица": s["person"], "ФИО": s["fio"] or c["fio"],
                "Роль в исходной организации": role, "Роль в организации-кандидате": c["role"],
                "Период (для сотрудников)": c["period"],
                "Основание": f"{role.capitalize()} исходной организации является {_INSTR[c['role']]} кандидата",
            })
    det = pd.DataFrame(details, columns=DETAIL_COLS).drop_duplicates()
    if det.empty:
        return {**empty, "message": f"Лица с ролью «{role}» ({len(source)}) не найдены в других организациях."}
    rank = (det.groupby("ИНН кандидата")
            .agg(**{"Наименование кандидата": ("Наименование кандидата", "first"),
                    "Совпавших лиц": ("ИНН физлица", "nunique"),
                    "Оснований": ("Основание", "size"),
                    "Роли в кандидате": ("Роль в организации-кандидате", lambda x: ", ".join(sorted(set(x))))})
            .reset_index()
            .sort_values(["Совпавших лиц", "Оснований", "ИНН кандидата"], ascending=[False, False, True]))
    rank.insert(0, "Место", range(1, len(rank) + 1))
    det = det.sort_values(["ИНН кандидата", "ИНН физлица", "Роль в организации-кандидате"]).reset_index(drop=True)
    return {"ranking": rank.loc[:, RANK_COLS].reset_index(drop=True), "details": det,
            "source_persons": len(source), "mode": mode,
            "message": f"Лиц с ролью «{role}» в исходной организации: {len(source)}. Кандидатов: {len(rank)}."}
