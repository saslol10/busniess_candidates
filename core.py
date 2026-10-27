from pathlib import Path
from datetime import datetime, date
from decimal import Decimal, InvalidOperation
import re
import pandas as pd

SCHEMAS = {
    "Сотрудники": ["ИНН организации", "ID сотрудника", "Месяц", "Сумма дохода"],
    "Покупки": ["ИНН организации", "ИНН поставщика", "Дата", "Сумма", "Код операции"],
    "Продажи": ["ИНН организации", "ИНН покупателя", "Дата", "Сумма", "Код операции"],
    "ККТ": ["ИНН организации", "Модель", "Заводской номер", "Адрес", "Дата регистрации", "Дата снятия"],
    "Организации": ["ИНН организации"],
    "Банк": ["ИНН организации", "ИНН контрагента", "Дата", "Направление", "Сумма", "Назначение"],
}
# Необязательные колонки: читаются, если есть в файле, и могут быть пустыми.
OPTIONAL_COLS = {
    "Сотрудники": ["ФИО сотрудника", "ИНН сотрудника"],
    "Организации": ["Наименование организации", "ФИО учредителя", "ИНН учредителя",
                    "ФИО руководителя", "ИНН руководителя"],
}
INN_COLS = {"ИНН организации": (10,), "ИНН сотрудника": (12,), "ИНН учредителя": (10, 12),
            "ИНН руководителя": (12,), "ИНН поставщика": (10, 12), "ИНН покупателя": (10, 12),
            "ИНН контрагента": (10, 12)}


def clean(value):
    if pd.isna(value):
        return ""
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def valid_inn(value, lengths=(10, 12)):
    if not re.fullmatch(r"[0-9]+", value or "") or len(value) not in lengths:
        return False
    a = list(map(int, value))
    if not any(a):
        return False

    def digit(weights, digits):
        return sum(w * d for w, d in zip(weights, digits)) % 11 % 10

    if len(a) == 10:
        return digit([2, 4, 10, 3, 5, 9, 4, 6, 8], a[:9]) == a[9]
    return (digit([7, 2, 4, 10, 3, 5, 9, 4, 6, 8], a[:10]) == a[10]
            and digit([3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8], a[:11]) == a[11])


def valid_date(value, month=False):
    formats = ("%Y-%m", "%m.%Y", "%Y-%m-%d", "%d.%m.%Y") if month else ("%Y-%m-%d", "%d.%m.%Y")
    for fmt in formats:
        try:
            datetime.strptime(value, fmt)
            return True
        except ValueError:
            pass
    return False


def valid_amount(value):
    try:
        n = Decimal(value.replace(" ", "").replace("\u00a0", "").replace(",", "."))
        return n.is_finite() and n >= 0
    except (InvalidOperation, ValueError):
        return False


def load_book(path):
    if Path(path).suffix.lower() != ".xlsx":
        raise ValueError("Поддерживаются только файлы .xlsx. Пересохраните старый .xls в Excel.")
    frames = {}
    errors = []
    warnings = []
    total_errors = 0

    def error(message):
        nonlocal total_errors
        total_errors += 1
        if len(errors) < 500:
            errors.append(message)

    with pd.ExcelFile(path, engine="openpyxl") as book:
        known = [s for s in book.sheet_names if s in SCHEMAS]
        if not known:
            raise ValueError("Не найдено известных листов. Используйте приложенный шаблон Excel.")
        unknown = [s for s in book.sheet_names if s not in SCHEMAS]
        if unknown:
            warnings.append("Пропущены неизвестные листы: " + ", ".join(unknown))
        missing = [s for s in SCHEMAS if s not in known]
        if missing:
            warnings.append("Отсутствуют необязательные листы: " + ", ".join(missing))
        for sheet in known:
            frame = pd.read_excel(book, sheet_name=sheet, dtype=object, keep_default_na=False)
            frame.columns = [str(c).strip() for c in frame.columns]
            if len(set(frame.columns)) != len(frame.columns):
                error(f"{sheet}: повторяются названия колонок.")
                continue
            absent = [c for c in SCHEMAS[sheet] if c not in frame.columns]
            if absent:
                error(f"{sheet}: отсутствуют колонки: {', '.join(absent)}")
                continue
            frame = frame.loc[:, SCHEMAS[sheet] + [c for c in OPTIONAL_COLS.get(sheet, []) if c in frame.columns]].copy()
            for column in frame.columns:
                frame[column] = frame[column].map(clean)
            frame = frame.loc[frame.ne("").any(axis=1)]
            for index, row in frame.iterrows():
                prefix = f"{sheet}, строка Excel {index + 2}: "
                for col in frame.columns:
                    value = row[col]
                    if not value and col != "Дата снятия" and col not in OPTIONAL_COLS.get(sheet, ()):
                        error(prefix + f"не заполнено поле «{col}».")
                        continue
                    if not value:
                        continue
                    if col in INN_COLS and not valid_inn(value, INN_COLS[col]):
                        error(prefix + f"некорректный ИНН в поле «{col}».")
                    if col in ("Сумма", "Сумма дохода") and not valid_amount(value):
                        error(prefix + f"«{col}» должна быть неотрицательным числом.")
                    if col in ("Дата", "Дата регистрации", "Дата снятия", "Месяц") and not valid_date(value,
                                                                                                      col == "Месяц"):
                        error(prefix + f"некорректная дата в поле «{col}».")
                if sheet == "Банк" and row["Направление"] not in ("Входящий", "Исходящий"):
                    error(prefix + "направление должно быть «Входящий» или «Исходящий».")
                if sheet in ("Покупки", "Продажи"):
                    code = row["Код операции"]
                    if re.fullmatch(r"[0-9]{1,2}", code):
                        frame.at[index, "Код операции"] = code.zfill(2)
                    else:
                        error(prefix + "код операции должен содержать одну или две цифры.")
                if sheet == "ККТ" and row["Дата снятия"] and valid_date(row["Дата снятия"]) and valid_date(
                        row["Дата регистрации"]):
                    def dt(v):
                        return datetime.strptime(v, "%d.%m.%Y" if "." in v else "%Y-%m-%d")

                    if dt(row["Дата снятия"]) < dt(row["Дата регистрации"]):
                        error(prefix + "снятие ККТ предшествует регистрации.")
            dup = int(frame.duplicated().sum())
            if dup:
                warnings.append(
                    f"{sheet}: {dup} полностью совпавших строк. Не удалены: без ID документа нельзя подтвердить дублирование.")
            frames[sheet] = frame
    if not sum(len(f) for f in frames.values()):
        error("Нет строк данных: файл пустой либо структура листов не подходит.")
    if total_errors > len(errors):
        errors.append(f"Показаны первые 500 ошибок из {total_errors}.")
    warnings.append("Контрольная сумма ИНН проверена локально; существование организации в реестре не проверялось.")
    return {"frames": frames, "errors": errors, "error_count": total_errors, "warnings": warnings}
