"""Детерминированный поиск общих сотрудников в локальной выгрузке."""
from datetime import datetime
from decimal import Decimal
import pandas as pd
from core import clean, valid_inn, valid_amount, valid_date

RANK_COLUMNS = ["ИНН кандидата", "Общие сотрудники", "Уникальные сотрудники исходной", "Доля, %",
                "Регулярные у обеих организаций", "Последовательные периоды в выгрузке"]
DETAIL_COLUMNS = ["ID сотрудника", "Месяцы: исходная", "Месяцы: кандидат", "Последний месяц: исходная",
                  "Первый месяц: кандидат", "Разница месяцев", "Характер совпадения", "Регулярность: исходная",
                  "Регулярность: кандидат", "Строки Excel: исходная", "Строки Excel: кандидат"]


def month_number(value):
    for fmt in ("%Y-%m", "%m.%Y", "%Y-%m-%d", "%d.%m.%Y"):
        try:
            dt = datetime.strptime(value, fmt)
            return dt.year * 12 + dt.month - 1
        except ValueError:
            continue
    raise ValueError("Некорректный месяц: " + value)


def month_label(n):
    return f"{n // 12:04d}-{n % 12 + 1:02d}"


def regular(months):
    values = sorted(months)
    streak = 0;
    previous = None
    for n in values:
        streak = streak + 1 if previous is not None and n == previous + 1 else 1
        if streak >= 3:
            return True
        previous = n
    return False


def search_staff(frame, source_inn):
    if not valid_inn(source_inn, (10,)):
        raise ValueError("Некорректный ИНН исходной организации.")
    required = ["ИНН организации", "ID сотрудника", "Месяц", "Сумма дохода"]
    if frame is None or any(c not in frame.columns for c in required):
        raise ValueError("Нет листа «Сотрудники» с обязательными колонками.")
    records = {}
    skipped = 0
    for index, row in frame.iterrows():
        org, person, month, total = (clean(row[c]) for c in required)
        if not (valid_inn(org, (10,)) and person and valid_date(month, True) and valid_amount(total)):
            skipped += 1
            continue
        if Decimal(total.replace(' ', '').replace('\u00a0', '').replace(',', '.')) <= 0:
            skipped += 1
            continue
        n = month_number(month)
        entry = records.setdefault(org, {}).setdefault(person, {"months": set(), "rows": set()})
        entry['months'].add(n)
        entry['rows'].add(int(index) + 2)
    source = records.get(source_inn, {})
    ranking = []
    details = {}
    for org, people in records.items():
        if org == source_inn:
            continue
        common = sorted(set(source) & set(people))
        if not common:
            continue
        rows = []
        both_regular = 0
        sequential = 0
        for person in common:
            a = source[person]
            b = people[person]
            am = a['months']
            bm = b['months']
            ra = regular(am)
            rb = regular(bm)
            both_regular += int(ra and rb)
            if min(bm) > max(am):
                kind = "Выплаты кандидата после исходной (в выгрузке)"
                sequential += 1
            elif am & bm:
                kind = "Есть общие месяцы выплат"
            elif min(bm) < min(am):
                kind = "Выплаты кандидата начались раньше"
            else:
                kind = "Периоды пересекаются; направление не установлено"
            rows.append([person, ', '.join(map(month_label, sorted(am))), ', '.join(map(month_label, sorted(bm))),
                         month_label(max(am)), month_label(min(bm)), min(bm) - max(am), kind, "Да" if ra else "Нет",
                         "Да" if rb else "Нет", ', '.join(map(str, sorted(a['rows']))),
                         ', '.join(map(str, sorted(b['rows'])))])
        ranking.append(
            [org, len(common), len(source), round(100 * len(common) / len(source), 2), both_regular, sequential])
        details[org] = pd.DataFrame(rows, columns=DETAIL_COLUMNS)
    ranking = pd.DataFrame(ranking, columns=RANK_COLUMNS)
    if not ranking.empty:
        ranking = ranking.sort_values(["Общие сотрудники", "Регулярные у обеих организаций", "ИНН кандидата"],
                                      ascending=[False, False, True], kind='stable').reset_index(drop=True)
    return {"ranking": ranking, "details": details, "source_count": len(source), "skipped": skipped,
            "warnings": ["Общие сотрудники не доказывают переход бизнеса. Анализ ограничен загруженными месяцами.",
                         "Учитываются уникальные ID с положительным доходом. ID должны быть одинаковыми для одного человека во всей книге.",
                         "Регулярность: не менее трёх последовательных месяцев с положительным доходом. Разница месяцев — расстояние между месяцами, не число дней.",
                         "Номера строк относятся к листу «Сотрудники» исходного Excel. Пустые, некорректные и нулевые выплаты не учитываются."]}
