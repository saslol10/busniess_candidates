"""Поиск по одинаковым заводским номерам ККТ. Интерфейс PySide6 импортируется отдельно."""
import re
from pathlib import Path
import pandas as pd

RANK_COLUMNS = ["ИНН кандидата", "Общие ККТ", "ККТ исходной организации", "Доля совпадения, %", "Заводские номера"]
DETAIL_COLUMNS = ["ИНН кандидата", "Роль записи", "Строка данных ККТ", "ИНН организации", "Заводской номер", "Модель",
                  "Адрес", "Дата регистрации", "Дата снятия"]


def text(value):
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def search_kkt(frames, source_inn):
    """Возвращает (рейтинг, основания, предупреждения). Входные DataFrame не изменяет.

    Номера сравниваются точно после удаления пробелов по краям.
    Даты не фильтруются и не используются для установления направления передачи.
    """
    source_inn = text(source_inn)
    if not re.fullmatch(r"(?:\d{10}|\d{12})", source_inn, flags=re.ASCII):
        raise ValueError("Введите ИНН: 10 или 12 цифр. Проверка существования организации не выполняется.")
    if "ККТ" not in frames:
        raise ValueError("В загруженной книге отсутствует лист «ККТ».")
    frame = frames["ККТ"].copy()
    missing = [c for c in ["ИНН организации", "Заводской номер"] if c not in frame.columns]
    if missing:
        raise ValueError("В листе «ККТ» отсутствуют столбцы: " + ", ".join(missing))

    frame["Строка данных ККТ"] = range(1, len(frame) + 1)
    for col in ["ИНН организации", "Заводской номер", "Модель", "Адрес", "Дата регистрации", "Дата снятия"]:
        if col not in frame.columns:
            frame[col] = ""
        frame[col] = frame[col].map(text)
    warnings = []
    valid_inn = frame["ИНН организации"].str.fullmatch(r"(?:[0-9]{10}|[0-9]{12})", na=False)
    valid_serial = frame["Заводской номер"].ne("")
    excluded = int((~(valid_inn & valid_serial)).sum())
    if excluded:
        warnings.append(f"Исключены строки с пустым заводским номером или неверным форматом ИНН: {excluded}.")
    frame = frame.loc[valid_inn & valid_serial].copy()
    source = frame.loc[frame["ИНН организации"].eq(source_inn)]
    serials = set(source["Заводской номер"])
    if not serials:
        warnings.append("У исходной организации не найдены пригодные для сравнения заводские номера.")
        return pd.DataFrame(columns=RANK_COLUMNS), pd.DataFrame(columns=DETAIL_COLUMNS), warnings
    candidates = frame.loc[frame["ИНН организации"].ne(source_inn) & frame["Заводской номер"].isin(serials)]
    rows, details = [], []
    for inn, group in candidates.groupby("ИНН организации", sort=True):
        common = sorted(set(group["Заводской номер"]))
        rows.append({"ИНН кандидата": inn, "Общие ККТ": len(common),
                     "ККТ исходной организации": len(serials),
                     "Доля совпадения, %": round(100 * len(common) / len(serials), 2),
                     "Заводские номера": "; ".join(common)})

        for role, records in [("Исходная организация", source.loc[source["Заводской номер"].isin(common)]),
                              ("Кандидат", group)]:
            evidence = records.copy()
            evidence["ИНН кандидата"] = inn
            evidence["Роль записи"] = role
            details.append(evidence[DETAIL_COLUMNS])
    ranking = pd.DataFrame(rows, columns=RANK_COLUMNS)
    if not ranking.empty:
        ranking = ranking.sort_values(["Общие ККТ", "ИНН кандидата"], ascending=[False, True]).reset_index(drop=True)
    evidence = pd.concat(details, ignore_index=True) if details else pd.DataFrame(columns=DETAIL_COLUMNS)
    warnings.append(
        "Повторяющиеся записи не увеличивают количество уникальных общих ККТ; в основаниях сохранены все записи.")
    warnings.append(
        "Совпадение адреса без совпадения номера не учитывается. Даты показаны как в источнике: направление передачи не установлено.")
    warnings.append(
        "Номер строки в основаниях — позиция в загруженной таблице ККТ, без заголовка; не обязательно номер строки исходного Excel.")
    return ranking, evidence, warnings


def export_kkt(path, ranking, evidence, warnings, source_inn):
    """Экспортирует снимок результатов; текст не превращается в формулы Excel."""

    def safe(frame):
        result = frame.copy()

        def cell(value):
            if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                return "'" + value
            return value

        for col in result.columns:
            result[col] = result[col].map(cell)
        return result

    meta = pd.DataFrame({"Параметр": ["ИНН исходной организации", "Критерий", "Период", "Ограничения"],
                         "Значение": [str(source_inn),
                                      "Точное совпадение заводского номера после обрезки пробелов по краям",
                                      "Все загруженные записи; фильтрация по датам не выполняется",
                                      "\n".join(warnings)]})
    with pd.ExcelWriter(Path(path), engine="openpyxl") as writer:
        for name, frame in [("Рейтинг ККТ", ranking), ("Основания ККТ", evidence), ("Параметры", meta)]:
            safe(frame).to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                sheet.column_dimensions[column[0].column_letter].width = min(65, max(18, len(str(column[0].value)) + 3))
                for cell_obj in column:
                    if isinstance(cell_obj.value, str):
                        cell_obj.number_format = "@"


def open_kkt_dialog(parent):
    """Вызывать из существующего Window; нужны parent.frames и, необязательно, parent.inn."""
    from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex
    from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                                   QPushButton, QLineEdit, QTabWidget, QTableView, QMessageBox, QFileDialog)

    class TableModel(QAbstractTableModel):
        def __init__(self, frame, owner):
            super().__init__(owner)
            self.frame = frame

        def rowCount(self, parent=QModelIndex()):
            return 0 if parent.isValid() else len(self.frame)

        def columnCount(self, parent=QModelIndex()):
            return 0 if parent.isValid() else len(self.frame.columns)

        def data(self, index, role=Qt.ItemDataRole.DisplayRole):
            if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
                return str(self.frame.iat[index.row(), index.column()])
            return None

        def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
            if role == Qt.ItemDataRole.DisplayRole:
                return str(self.frame.columns[section]) if orientation == Qt.Orientation.Horizontal else str(
                    section + 1)
            return None

    class KktDialog(QDialog):
        def __init__(self, owner):
            super().__init__(owner)
            self.setWindowTitle("Кандидаты по ККТ")
            self.resize(1150, 720)
            self.frames = owner.frames
            self.result = None
            layout = QVBoxLayout(self)
            intro = QLabel(
                "Совпадение заводских номеров ККТ. Доля — от уникальных касс исходной организации, не вероятность перехода бизнеса.")
            intro.setWordWrap(True)
            layout.addWidget(intro)
            row = QHBoxLayout()
            self.inn_edit = QLineEdit()
            self.inn_edit.setPlaceholderText("ИНН исходной организации")
            self.inn_edit.setMaxLength(12)
            if hasattr(owner, "inn"):
                self.inn_edit.setText(owner.inn.text().strip())
            run = QPushButton("Найти по ККТ")
            run.clicked.connect(self.search)
            self.inn_edit.returnPressed.connect(self.search)
            self.inn_edit.textChanged.connect(self.invalidate)
            self.save_button = QPushButton("Сохранить результаты в Excel…")
            self.save_button.setEnabled(False)
            self.save_button.clicked.connect(self.save)
            row.addWidget(self.inn_edit)
            row.addWidget(run)
            row.addWidget(self.save_button)
            layout.addLayout(row)
            self.status_label = QLabel("Введите ИНН и запустите поиск.")
            self.status_label.setWordWrap(True)
            layout.addWidget(self.status_label)
            self.tabs = QTabWidget()
            layout.addWidget(self.tabs)
            self.notes = QLabel()
            self.notes.setWordWrap(True)
            layout.addWidget(self.notes)

        def invalidate(self, *_):
            self.result = None
            self.save_button.setEnabled(False)
            if hasattr(self, "tabs"):
                self.tabs.clear()
                self.notes.clear()
                self.status_label.setText("Параметры изменены. Запустите поиск.")

        def search(self):
            self.invalidate()
            try:
                inn = self.inn_edit.text().strip()
                ranking, evidence, notes = search_kkt(self.frames, inn)
            except Exception as exc:
                QMessageBox.warning(self, "Поиск ККТ", str(exc))
                return
            self.result = (ranking, evidence, notes, inn)
            for title, frame in [("Кандидаты", ranking), ("Основания", evidence)]:
                table = QTableView()
                table.setModel(TableModel(frame, table))
                table.setAlternatingRowColors(True)
                table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
                for i in range(len(frame.columns)):
                    table.setColumnWidth(i, 190)
                self.tabs.addTab(table, f"{title} ({len(frame)})")
            self.status_label.setText(
                f"ИНН {inn}. Найдено кандидатов: {len(ranking)}." if not ranking.empty else f"ИНН {inn}. Совпадения не найдены.")
            self.notes.setText("\n".join(notes))
            self.save_button.setEnabled(True)

        def save(self):
            if self.result is None:
                return
            ranking, evidence, notes, inn = self.result
            path, _ = QFileDialog.getSaveFileName(self, "Сохранить результаты", f"kkt_{inn}.xlsx", "Excel (*.xlsx)")
            if not path:
                return
            if not path.lower().endswith(".xlsx"):
                path += ".xlsx"
            try:
                export_kkt(path, ranking, evidence, notes, inn)
            except Exception as exc:
                QMessageBox.warning(self, "Ошибка сохранения", str(exc))
                return
            QMessageBox.information(self, "Готово", "Рейтинг, основания и параметры сохранены.")

    if not getattr(parent, "frames", None):
        QMessageBox.information(parent, "Нет данных", "Сначала загрузите Excel в главном окне.")
        return
    if "ККТ" not in parent.frames:
        QMessageBox.information(parent, "Нет листа ККТ", "Добавьте в Excel лист «ККТ» и повторно загрузите книгу.")
        return
    KktDialog(parent).exec()
