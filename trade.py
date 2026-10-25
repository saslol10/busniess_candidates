from collections import defaultdict
from datetime import datetime
from decimal import Decimal

import pandas as pd

from core import clean, valid_inn, valid_amount

MODES = {
    "suppliers": {
        "title": "Общие поставщики",
        "sheet": "Покупки",
        "partner_column": "ИНН поставщика",
    },
    "buyers": {
        "title": "Общие покупатели",
        "sheet": "Продажи",
        "partner_column": "ИНН покупателя",
    },
}

REQUIRED = {
    "Покупки": [
        "ИНН организации",
        "ИНН поставщика",
        "Дата",
        "Сумма",
        "Код операции",
    ],
    "Продажи": [
        "ИНН организации",
        "ИНН покупателя",
        "Дата",
        "Сумма",
        "Код операции",
    ],
}


def parse_date(value):
    text = clean(value)
    for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Некорректная дата: {text}")


def parse_amount(value):
    text = clean(value)
    if not valid_amount(text):
        raise ValueError(f"Некорректная сумма: {text}")
    return Decimal(
        text.replace(" ", "").replace("\u00a0", "").replace(",", ".")
    )


def read_rows(frame, sheet):
    """Проверяет строки и сохраняет номера строк исходного Excel."""
    if frame is None:
        raise ValueError(f"В книге отсутствует лист «{sheet}».")

    missing = [c for c in REQUIRED[sheet] if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{sheet}: отсутствуют столбцы: {', '.join(missing)}"
        )

    result = []
    partner_column = (
        "ИНН поставщика" if sheet == "Покупки"
        else "ИНН покупателя"
    )

    for position, (index, row) in enumerate(frame.iterrows(), start=2):
        # load_book сохраняет исходный индекс даже после удаления пустых строк.
        excel_row = (
            int(index) + 2
            if isinstance(index, int) or pd.api.types.is_integer(index)
            else position
        )

        values = {c: clean(row[c]) for c in REQUIRED[sheet]}
        if not any(values.values()):
            continue

        prefix = f"{sheet}, строка Excel {excel_row}: "
        org = values["ИНН организации"]
        partner = values[partner_column]

        if not valid_inn(org, (10,)):
            raise ValueError(prefix + "некорректный ИНН организации.")
        if not valid_inn(partner, (10, 12)):
            raise ValueError(prefix + "некорректный ИНН контрагента.")

        code = values["Код операции"]
        if not code.isdigit() or len(code) > 2:
            raise ValueError(prefix + "некорректный код операции.")

        try:
            amount = parse_amount(values["Сумма"])
            operation_date = parse_date(values["Дата"])
        except ValueError as exc:
            raise ValueError(prefix + str(exc)) from exc

        result.append({
            "org": org,
            "partner": partner,
            "date": operation_date,
            "amount": amount,
            "code": code.zfill(2),
            "sheet": sheet,
            "row": excel_row,
        })

    return result


def aggregate(rows, key_function):
    grouped = {}
    for row in rows:
        key = key_function(row)
        if key not in grouped:
            grouped[key] = {
                "amount": Decimal("0"),
                "first": row["date"],
                "last": row["date"],
                "rows": [],
            }

        item = grouped[key]
        item["amount"] += row["amount"]
        item["first"] = min(item["first"], row["date"])
        item["last"] = max(item["last"], row["date"])
        item["rows"].append(row["row"])

    return grouped


def raw_frame(rows):
    columns = [
        "Лист",
        "Строка Excel",
        "ИНН организации",
        "ИНН контрагента",
        "Дата",
        "Сумма с НДС",
        "Код операции",
    ]
    return pd.DataFrame([
        {
            "Лист": row["sheet"],
            "Строка Excel": row["row"],
            "ИНН организации": row["org"],
            "ИНН контрагента": row["partner"],
            "Дата": row["date"].isoformat(),
            "Сумма с НДС": float(row["amount"]),
            "Код операции": row["code"],
        }
        for row in rows
    ], columns=columns)


def find_trade_candidates(
        frames,
        source_inn,
        mode,
        source_from=None,
        source_to=None,
        candidate_from=None,
        candidate_to=None,
):
    """
    Поиск организаций с общими контрагентами.

    suppliers:
        Покупки исходной организации и кандидатов
        сопоставляются по ИНН поставщика.

    buyers:
        Продажи исходной организации и кандидатов
        сопоставляются по ИНН покупателя.

    Периоды исходной организации и кандидатов задаются независимо.
    None означает отсутствие соответствующей границы.
    """
    source_inn = source_inn.strip()

    if not valid_inn(source_inn, (10,)):
        raise ValueError("Нужен корректный 10-значный ИНН организации.")

    if mode not in MODES:
        raise ValueError("Неизвестное направление поиска.")

    for start, end in (
            (source_from, source_to),
            (candidate_from, candidate_to),
    ):
        if start and end and start > end:
            raise ValueError(
                "Начало периода позже его окончания."
            )

    config = MODES[mode]
    sheet = config["sheet"]

    all_rows = read_rows(frames.get(sheet), sheet)

    def in_period(row, start, end):
        return (
                (start is None or row["date"] >= start)
                and (end is None or row["date"] <= end)
        )

    source_rows = [
        row
        for row in all_rows
        if row["org"] == source_inn
           and row["code"] == "01"
           and in_period(row, source_from, source_to)
    ]

    # Для расчёта оборота учитываем только положительные операции.
    source_positive_rows = [
        row for row in source_rows
        if row["amount"] > 0
    ]

    source_groups = aggregate(
        source_positive_rows,
        lambda row: row["partner"],
    )

    partners = set(source_groups)

    source_total = sum(
        (group["amount"] for group in source_groups.values()),
        Decimal("0"),
    )

    # Кандидаты находятся на том же листе:
    # org — организация-кандидат,
    # partner — общий поставщик или общий покупатель.
    candidate_rows = [
        row
        for row in all_rows
        if row["org"] != source_inn
           and row["partner"] in partners
           and row["org"] != row["partner"]
           and row["code"] == "01"
           and row["amount"] > 0
           and in_period(row, candidate_from, candidate_to)
    ]

    candidate_groups = aggregate(
        candidate_rows,
        lambda row: (row["org"], row["partner"]),
    )

    candidates = defaultdict(list)
    details = []

    for (candidate, partner), candidate_group in candidate_groups.items():
        source_group = source_groups[partner]

        gap = (
                candidate_group["first"] - source_group["last"]
        ).days

        if gap > 0:
            observation = (
                "Первая операция кандидата позже "
                "последней операции исходной организации"
            )
        else:
            observation = (
                "Периоды операций пересекаются или "
                "не подтверждают последовательную смену"
            )

        details.append({
            "ИНН кандидата": candidate,
            "ИНН общего контрагента": partner,
            "Сумма исходной с НДС": float(
                source_group["amount"]
            ),
            "Доля в обороте исходной, %": round(
                float(
                    source_group["amount"]
                    / source_total
                    * 100
                ),
                2,
            ),
            "Сумма кандидата с НДС": float(
                candidate_group["amount"]
            ),
            "Первая дата исходной": (
                source_group["first"].isoformat()
            ),
            "Последняя дата исходной": (
                source_group["last"].isoformat()
            ),
            "Первая дата кандидата": (
                candidate_group["first"].isoformat()
            ),
            "Последняя дата кандидата": (
                candidate_group["last"].isoformat()
            ),
            "Разница дат, дней": gap,
            "Наблюдение": observation,
            "Лист исходной": sheet,
            "Строки исходной": ", ".join(
                map(str, source_group["rows"])
            ),
            "Лист кандидата": sheet,
            "Строки кандидата": ", ".join(
                map(str, candidate_group["rows"])
            ),
        })

        candidates[candidate].append(
            (partner, candidate_group)
        )

    ranking_rows = []

    for candidate, links in candidates.items():
        common_source_amount = sum(
            (
                source_groups[partner]["amount"]
                for partner, _ in links
            ),
            Decimal("0"),
        )

        candidate_amount = sum(
            (
                candidate_group["amount"]
                for _, candidate_group in links
            ),
            Decimal("0"),
        )

        ranking_rows.append({
            "ИНН кандидата": candidate,
            "Общих контрагентов": len(links),
            "Всего контрагентов исходной": len(partners),
            "Доля контрагентов исходной, %": round(
                len(links) / len(partners) * 100,
                2,
            ),
            "Оборот исходной с общими, с НДС": float(
                common_source_amount
            ),
            "Покрытие оборота исходной, %": round(
                float(
                    common_source_amount
                    / source_total
                    * 100
                ),
                2,
            ),
            "Оборот кандидата с общими, с НДС": float(
                candidate_amount
            ),
        })

    ranking = pd.DataFrame(ranking_rows)

    if not ranking.empty:
        ranking = ranking.sort_values(
            [
                "Покрытие оборота исходной, %",
                "Общих контрагентов",
                "Оборот кандидата с общими, с НДС",
                "ИНН кандидата",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
        ).reset_index(drop=True)

        ranking.insert(
            0,
            "Место",
            range(1, len(ranking) + 1),
        )

    warnings = [
        "Учитывается код операции 01.",
        "Суммы считаются суммами с НДС.",
        (
            "Общие связи и покрытие оборота "
            "не являются вероятностью перехода бизнеса."
        ),
        (
            "Даты отражают только загруженную выборку, "
            "а не доказанную дату прекращения отношений."
        ),
        (
            "Совпавшие строки не удаляются автоматически: "
            "без идентификатора документа нельзя подтвердить дубликаты."
        ),
        (
            "Наименования не выводятся: "
            "в текущей схеме нет справочника организаций."
        ),
        (
            "Доля в общем обороте кандидата не рассчитывается, "
            "поскольку выборка может быть неполной."
        ),
        (
            "Сопоставление выполняется на одном листе "
            "по ИНН поставщика или покупателя."
        ),
    ]

    if not source_groups:
        warnings.append(
            "Нет положительных операций исходной организации "
            "с кодом 01 в выбранном периоде."
        )
    elif ranking.empty:
        warnings.append(
            "Кандидаты не найдены. Проверьте наличие операций "
            "с теми же поставщиками или покупателями "
            "и границы периода."
        )

    return {
        "title": config["title"],
        "source_inn": source_inn,
        "ranking": ranking,
        "details": pd.DataFrame(details),
        "source_rows": raw_frame(source_rows),
        "candidate_rows": raw_frame(candidate_rows),
        "warnings": warnings,
    }
