"""Двусторонний поиск кандидатов по учредителям и руководителям.

founders: учредитель хотя бы на одной стороне связи.
directors: руководитель хотя бы на одной стороне связи.
На другой стороне допускаются учредитель, руководитель или сотрудник.
Совпадение определяется по ИНН физлица, а не по ФИО.
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
                if o and p:
                    rows.append({"org": o, "person": p, "fio": clean(r.get(fio_col, "")), "role": role, "period": ""})
    staff = frames.get("Сотрудники")
    if staff is not None and not staff.empty and "ИНН сотрудника" in staff.columns:
        s = staff.assign(_p=staff["ИНН сотрудника"].map(clean),
                         _o=staff["ИНН организации"].map(clean))
        s = s[(s["_p"] != "") & (s["_o"] != "")]
        for (o, p), g in s.groupby(["_o", "_p"]):
            nums = sorted(n for n in g["Месяц"].map(month_number) if n is not None)
            fio = next((clean(x) for x in g.get("ФИО сотрудника", []) if clean(x)), "")
            period = f"{month_label(nums[0])} – {month_label(nums[-1])}" if nums else ""
            rows.append({"org": o, "person": p, "fio": fio, "role": STAFF, "period": period})
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
    # Не отбираем только выбранную роль и не удаляем другие роли человека.
    # Иначе теряются обратные связи и отдельные основания.
    source = rec[rec["org"] == src]
    source_persons = source.loc[source["role"] == role, "person"].nunique()
    empty = {"ranking": pd.DataFrame(columns=RANK_COLS),
             "details": pd.DataFrame(columns=DETAIL_COLS),
             "source_persons": int(source_persons), "mode": mode}
    if source.empty:
        return {**empty, "message": f"У организации {src} не найдены учредители, руководители или сотрудники с ИНН физлица."}

    other = rec[(rec["org"] != src) & rec["person"].isin(source["person"])]
    pairs = source.merge(other, on="person", suffixes=("_source", "_candidate"))
    # Выбранная роль должна присутствовать хотя бы на одной стороне.
    pairs = pairs[(pairs["role_source"] == role) | (pairs["role_candidate"] == role)]
    details = []
    for _, pair in pairs.iterrows():
        source_role, candidate_role = pair["role_source"], pair["role_candidate"]
        periods = []
        if source_role == STAFF and pair["period_source"]:
            periods.append("Исходная организация: " + pair["period_source"])
        if candidate_role == STAFF and pair["period_candidate"]:
            periods.append("Кандидат: " + pair["period_candidate"])
        details.append({
            "ИНН кандидата": pair["org_candidate"],
            "Наименование кандидата": names.get(pair["org_candidate"], ""),
            "ИНН физлица": pair["person"],
            "ФИО": pair["fio_source"] or pair["fio_candidate"],
            "Роль в исходной организации": source_role,
            "Роль в организации-кандидате": candidate_role,
            "Период (для сотрудников)": "; ".join(periods),
            "Основание": f"{source_role.capitalize()} исходной организации является {_INSTR[candidate_role]} кандидата",
        })
    det = pd.DataFrame(details, columns=DETAIL_COLS).drop_duplicates()
    if det.empty:
        return {**empty, "message": f"Совпадения по ИНН физлица с ролью «{role}» хотя бы в одной из организаций не найдены."}

    rank = (det.groupby("ИНН кандидата")
            .agg(**{"Наименование кандидата": ("Наименование кандидата", "first"),
                    "Совпавших лиц": ("ИНН физлица", "nunique"),
                    "Оснований": ("Основание", "size"),
                    "Роли в кандидате": ("Роль в организации-кандидате", lambda x: ", ".join(sorted(set(x))))})
            .reset_index()
            .sort_values(["Совпавших лиц", "Оснований", "ИНН кандидата"], ascending=[False, False, True]))
    rank.insert(0, "Место", range(1, len(rank) + 1))
    det = det.sort_values(["ИНН кандидата", "ИНН физлица", "Роль в исходной организации",
                           "Роль в организации-кандидате"]).reset_index(drop=True)
    return {"ranking": rank.loc[:, RANK_COLS].reset_index(drop=True), "details": det,
            "source_persons": int(source_persons), "mode": mode,
            "message": (f"Лиц с ролью «{role}» в исходной организации: {source_persons}. "
                        f"Поиск выполнен в обе стороны. Кандидатов: {len(rank)}. Оснований: {len(det)}.")}
