from datetime import date
from pathlib import Path
import os
import tempfile

import pandas as pd

from PySide6.QtCore import (
    Qt, QAbstractTableModel, QModelIndex, QThread, Signal
)
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QLineEdit, QPushButton, QTabWidget,
    QTableView, QMessageBox, QFileDialog,
)

from trade import find_trade_candidates


class TradeTableModel(QAbstractTableModel):
    def __init__(self, frame, parent=None):
        super().__init__(parent)
        self.frame = frame.reset_index(drop=True)

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.frame)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.frame.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self.frame.iat[index.row(), index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            if isinstance(value, float):
                return f"{value:,.2f}".replace(",", " ")
            return str(value)
        if role == Qt.ItemDataRole.ToolTipRole:
            return str(value)
        return None

    def headerData(self, section, orientation,
                   role=Qt.ItemDataRole.DisplayRole):
        if role in (
            Qt.ItemDataRole.DisplayRole,
            Qt.ItemDataRole.ToolTipRole,
        ):
            if orientation == Qt.Orientation.Horizontal:
                return str(self.frame.columns[section])
            return str(section + 1)
        return None


class TradeWorker(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, frames, inn, mode, periods, parent=None):
        super().__init__(parent)
        self.frames = frames
        self.inn = inn
        self.mode = mode
        self.periods = periods

    def run(self):
        try:
            result = find_trade_candidates(
                self.frames, self.inn, self.mode, **self.periods
            )
            self.succeeded.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class TradeDialog(QDialog):
    def __init__(self, frames, source_inn="", parent=None):
        super().__init__(parent)
        self.frames = frames
        self.result = None
        self.worker = None

        self.setWindowTitle("Торговые связи")
        self.resize(1280, 800)
        layout = QVBoxLayout(self)

        note = QLabel(
            "Общие поставщики: покупки исходной и кандидатов "
            "сопоставляются по ИНН поставщика.\n"
            "Общие покупатели: продажи исходной и кандидатов "
            "сопоставляются по ИНН покупателя.\n"
            "Пустые границы периода означают все доступные даты."
        )

        note.setWordWrap(True)
        layout.addWidget(note)

        controls = QHBoxLayout()
        self.inn = QLineEdit(source_inn)
        self.inn.setPlaceholderText("ИНН исходной организации")
        self.inn.setMaxLength(10)

        self.mode = QComboBox()
        self.mode.addItem("Общие поставщики", "suppliers")
        self.mode.addItem("Общие покупатели", "buyers")

        self.search = QPushButton("Найти кандидатов")
        self.search.clicked.connect(self.start_search)

        self.export = QPushButton("Сохранить Excel…")
        self.export.setEnabled(False)
        self.export.clicked.connect(self.export_result)

        for widget in (self.inn, self.mode, self.search, self.export):
            controls.addWidget(widget)
        layout.addLayout(controls)

        self.period_fields = {}
        for title, keys in (
            ("Период исходной:", ("source_from", "source_to")),
            ("Период кандидатов:", ("candidate_from", "candidate_to")),
        ):
            row = QHBoxLayout()
            row.addWidget(QLabel(title))
            for key in keys:
                field = QLineEdit()
                field.setPlaceholderText("ГГГГ-ММ-ДД, необязательно")
                self.period_fields[key] = field
                row.addWidget(field)
            layout.addLayout(row)

        self.status = QLabel("Задайте ИНН и направление поиска.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        self.warning = QLabel(
            "Совпадения не подтверждают переход бизнеса сами по себе."
        )
        self.warning.setWordWrap(True)
        layout.addWidget(self.warning)

    def start_search(self):
        try:
            periods = {}
            for key, field in self.period_fields.items():
                text = field.text().strip()
                periods[key] = date.fromisoformat(text) if text else None
        except ValueError:
            QMessageBox.warning(
                self, "Период",
                "Введите даты в формате ГГГГ-ММ-ДД."
            )
            return

        self.result = None
        self.tabs.clear()
        self.export.setEnabled(False)
        self.search.setEnabled(False)
        self.status.setText("Расчёт торговых связей…")

        self.worker = TradeWorker(
            self.frames,
            self.inn.text().strip(),
            self.mode.currentData(),
            periods,
            self,
        )
        self.worker.succeeded.connect(self.show_result)
        self.worker.failed.connect(self.show_error)
        self.worker.finished.connect(
            lambda: self.search.setEnabled(True)
        )
        self.worker.start()

    def show_error(self, message):
        self.status.setText("Расчёт не выполнен.")
        QMessageBox.warning(self, "Ошибка расчёта", message)

    def add_table(self, title, frame):
        table = QTableView()
        table.setModel(TradeTableModel(frame, table))
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(
            QTableView.SelectionBehavior.SelectRows
        )
        table.setWordWrap(False)
        for column in range(len(frame.columns)):
            table.setColumnWidth(column, 220)
        self.tabs.addTab(table, title)

    def show_result(self, result):
        self.result = result
        self.tabs.clear()
        self.add_table("Кандидаты", result["ranking"])
        self.add_table("Основания", result["details"])
        self.add_table("Строки исходной", result["source_rows"])
        self.add_table("Строки кандидатов", result["candidate_rows"])

        self.status.setText(
            f"{result['title']} · ИНН {result['source_inn']} · "
            f"кандидатов: {len(result['ranking'])}"
        )
        self.warning.setText("\n".join(result["warnings"]))
        self.export.setEnabled(True)

    def export_result(self):
        if self.result is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить торговые связи",
            f"trade_{self.result['source_inn']}.xlsx",
            "Книга Excel (*.xlsx)"
        )
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"

        target = Path(path)
        temporary = None
        try:
            descriptor, temporary = tempfile.mkstemp(
                suffix=".xlsx", dir=str(target.parent)
            )
            os.close(descriptor)

            result = self.result
            with pd.ExcelWriter(
                temporary, engine="openpyxl"
            ) as writer:
                for title, key in (
                    ("Кандидаты", "ranking"),
                    ("Основания", "details"),
                    ("Строки исходной", "source_rows"),
                    ("Строки кандидатов", "candidate_rows"),
                ):
                    result[key].to_excel(
                        writer, sheet_name=title, index=False
                    )

                pd.DataFrame({
                    "Методика и ограничения": [
                        result["title"],
                        "Исходный ИНН: " + result["source_inn"],
                        *result["warnings"],
                    ]
                }).to_excel(
                    writer, sheet_name="Методика", index=False
                )

                for sheet in writer.book.worksheets:
                    sheet.freeze_panes = "A2"
                    sheet.auto_filter.ref = sheet.dimensions
                    for cells in sheet.columns:
                        letter = cells[0].column_letter
                        sheet.column_dimensions[letter].width = 28

            os.replace(temporary, target)
            temporary = None

            QMessageBox.information(
                self, "Готово", "Результат сохранён."
            )
        except Exception as exc:
            QMessageBox.warning(
                self, "Ошибка сохранения", str(exc)
            )
        finally:
            if temporary and os.path.exists(temporary):
                os.remove(temporary)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(
                self, "Расчёт",
                "Дождитесь завершения расчёта."
            )
            event.ignore()
        else:
            event.accept()