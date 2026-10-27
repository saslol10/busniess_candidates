"""Окно общего поиска. Qt-виджеты создаются только в основном потоке."""
import json
import os
import tempfile
from pathlib import Path
import pandas as pd
from PySide6.QtCore import Qt, QThread, Signal, QAbstractTableModel, QModelIndex, QTimer
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTabWidget, QTableView, QPlainTextEdit, QSplitter, QWidget, QMessageBox, QFileDialog)
from candidate_search import run_candidate_search, DIRECTIONS, ROLE_DIRECTIONS

STATUS = {"completed": "Проверено", "no_data": "Нет данных", "error": "Ошибка"}
HEADERS = {"source_inn": "ИНН исходной", "candidate_inn": "ИНН кандидата",
    "staff_common_count": "Общие сотрудники", "staff_source_share_pct": "Доля сотрудников исходной, %",
    "staff_regular_common_count": "Регулярные у обеих организаций",
    "staff_sequential_count": "Последовательные периоды в выгрузке",
    "suppliers_common_count": "Общие поставщики", "buyers_common_count": "Общие покупатели",
    "kkt_common_count": "Общие ККТ", "kkt_source_share_pct": "Доля ККТ исходной, %"}
ALL_DIRECTIONS = dict(DIRECTIONS)
ALL_DIRECTIONS.update(ROLE_DIRECTIONS)

for group, name in ALL_DIRECTIONS.items():
    HEADERS[group + "_search_completed"] = name + ": поиск выполнен"
    HEADERS[group + "_candidate_found"] = name + ": кандидат найден"
    if group in ("suppliers", "buyers"):
        for suffix, text in {
            "source_count_share_pct": "Доля контрагентов исходной, %",
            "source_turnover_share_pct": "Покрытие оборота исходной, %",
            "common_source_amount": "Оборот исходной с общими, с НДС",
            "common_candidate_amount": "Оборот кандидата с общими, с НДС"}.items():
            HEADERS[group + "_" + suffix] = name + ": " + text


class CandidateTableModel(QAbstractTableModel):
    def __init__(self, frame, parent=None, headers=False):
        super().__init__(parent)
        self.frame = frame.reset_index(drop=True)
        self.headers = headers

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.frame)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.frame.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self.frame.iat[index.row(), index.column()]
        col = str(self.frame.columns[index.column()])
        if role == Qt.ItemDataRole.ToolTipRole:
            return "Нет рассчитанного значения; проверьте статус направления." if pd.isna(value) else str(value)
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if pd.isna(value):
            return "—"
        if col == "Статус":
            return STATUS.get(str(value), str(value))
        if col.endswith(("_search_completed", "_candidate_found")):
            return "Да" if value == 1 else "Нет"
        if isinstance(value, float):
            return str(int(value)) if value.is_integer() else f"{value:.2f}"
        return str(value)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role not in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            return None
        if orientation == Qt.Orientation.Vertical:
            return str(section + 1)
        name = str(self.frame.columns[section])
        return HEADERS.get(name, name) if self.headers and role == Qt.ItemDataRole.DisplayRole else name


class CandidateWorker(QThread):
    succeeded = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, frames, inn, parent=None):
        super().__init__(parent)
        self.frames, self.inn = frames, inn

    def run(self):
        try:
            self.succeeded.emit(run_candidate_search(self.frames, self.inn, self.progress.emit))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class CandidateSearchDialog(QDialog):
    def __init__(self, frames, source_inn, parent=None):
        super().__init__(parent)
        self.frames, self.source_inn = frames, source_inn
        self.worker, self.result = None, None
        self.setWindowTitle("Общий поиск кандидатов · ИНН " + source_inn)
        self.resize(1280, 850)
        layout = QVBoxLayout(self)
        note = QLabel("Объединение сотрудников, поставщиков, покупателей и ККТ; учредители и руководители отображаются отдельными направлениями. "
            "Без ИИ, вероятности и итогового рейтинга. Кандидаты упорядочены по ИНН. "
            "Период: вся загруженная книга; фильтры других окон не используются. "
            "Совпадения не доказывают переход бизнеса. «—» означает отсутствие расчёта, а не ноль.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.status = QLabel("Подготовка поиска…")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.states = QTableView()
        self.states.setMaximumHeight(190)
        self.states.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.states)
        actions = QHBoxLayout()
        self.retry = QPushButton("Повторить поиск")
        self.retry.clicked.connect(self.start_search)
        self.export = QPushButton("Сохранить в Excel…")
        self.export.setEnabled(False)
        self.export.clicked.connect(self.export_result)
        self.close_button = QPushButton("Закрыть")
        self.close_button.clicked.connect(self.reject)
        actions.addWidget(self.retry)
        actions.addWidget(self.export)
        actions.addStretch()
        actions.addWidget(self.close_button)
        layout.addLayout(actions)
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.table = QTableView()
        self.table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        splitter.addWidget(self.table)
        self.details = QTabWidget()
        splitter.addWidget(self.details)
        splitter.setSizes([320, 220])
        layout.addWidget(splitter, 1)
        self.notes = QPlainTextEdit()
        self.notes.setReadOnly(True)
        self.notes.setMaximumHeight(115)
        layout.addWidget(self.notes)
        QTimer.singleShot(0, self.start_search)

    def running(self):
        return self.worker is not None and self.worker.isRunning()

    def start_search(self):
        if self.running():
            return
        if self.worker is not None:
            self.worker.deleteLater()
        self.result = None
        self.details.clear()
        self.table.setModel(CandidateTableModel(pd.DataFrame(), self.table))
        self.states.setModel(CandidateTableModel(pd.DataFrame(), self.states))
        self.notes.clear()
        self.retry.setEnabled(False)
        self.export.setEnabled(False)
        self.close_button.setEnabled(False)
        self.status.setText("Поиск выполняется в фоновом потоке…")
        self.worker = CandidateWorker(self.frames, self.source_inn, self)
        self.worker.progress.connect(self.show_progress)
        self.worker.succeeded.connect(self.show_result)
        self.worker.failed.connect(self.show_error)
        self.worker.finished.connect(self.search_finished)
        self.worker.start()

    def show_progress(self, title):
        self.status.setText("Выполняется поиск: " + title + "…")

    def search_finished(self):
        self.retry.setEnabled(True)
        self.close_button.setEnabled(True)
        self.export.setEnabled(self.result is not None)

    def show_error(self, message):
        self.status.setText("Общий поиск не выполнен.")
        self.notes.setPlainText(message)

    def show_result(self, result):
        self.result = result
        states = result["states"]
        self.states.setModel(CandidateTableModel(states, self.states))
        self.states.resizeColumnsToContents()
        features = result["features"]
        self.table.setModel(CandidateTableModel(features, self.table, headers=True))
        for c in range(len(features.columns)):
            self.table.setColumnWidth(c, 210 if c > 1 else 150)
        self.table.selectionModel().currentRowChanged.connect(self.select_candidate)
        failed = int((states["Статус"] == "error").sum())
        missing = int((states["Статус"] == "no_data").sum())
        self.status.setText(f"Кандидатов: {len(features)} · направлений с ошибками: {failed} · без данных: {missing}. "
            + ("Выберите кандидата для просмотра оснований, включая учредителей и руководителей." if len(features) else
               "Кандидаты не найдены в доступных данных. Это не доказывает отсутствие перехода."))
        self.notes.setPlainText("\n".join(result["warnings"]) or "Дополнительных предупреждений нет.")
        if not features.empty:
            self.table.selectRow(0)

    def select_candidate(self, current, previous=QModelIndex()):
        self.details.clear()
        if not current.isValid() or self.result is None:
            return
        inn = str(self.result["features"].iloc[current.row()]["candidate_inn"])
        for group, title in ALL_DIRECTIONS.items():
            matched = self.result["states"].loc[self.result["states"]["Код"] == group]
            if matched.empty:
                continue
            row = matched.iloc[0]
            frame = pd.DataFrame()
            if row["Статус"] == "completed":
                evidence = self.result["raw"][group].get("details")
                if isinstance(evidence, dict):
                    frame = evidence.get(inn, pd.DataFrame())
                elif isinstance(evidence, pd.DataFrame) and "ИНН кандидата" in evidence.columns:
                    frame = evidence.loc[evidence["ИНН кандидата"].astype(str) == inn]
            if not frame.empty:
                view = QTableView()
                view.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
                view.setModel(CandidateTableModel(frame, view))
                for c in range(len(frame.columns)):
                    view.setColumnWidth(c, 190)
                self.details.addTab(view, title + f" ({len(frame)})")
            else:
                label = QLabel(("Кандидат не найден в этом направлении." if row["Статус"] == "completed"
                                else STATUS[row["Статус"]] + ": " + row["Сообщение"]))
                label.setWordWrap(True)
                self.details.addTab(label, title)

    def export_result(self):
        if self.result is None or self.running():
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить общий поиск",
            f"candidates_{self.source_inn}.xlsx", "Книга Excel (*.xlsx)")
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        temp = None
        try:
            from openpyxl import load_workbook
            target = Path(path)
            fd, temp = tempfile.mkstemp(suffix=".xlsx", dir=target.parent)
            os.close(fd)
            with pd.ExcelWriter(temp, engine="openpyxl") as writer:
                self.result["features"].to_excel(writer, sheet_name="Признаки", index=False)
                states = self.result["states"].copy()
                states["Статус"] = states["Статус"].map(STATUS)
                states.to_excel(writer, sheet_name="Направления", index=False)
                metadata = {key: json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value
                            for key, value in self.result["metadata"].items()}
                pd.DataFrame(metadata.items(), columns=["Параметр", "Значение"]).to_excel(writer, sheet_name="Параметры", index=False)
                pd.DataFrame({"Предупреждения": self.result["warnings"]}).to_excel(writer, sheet_name="Ограничения", index=False)
                for group, result in self.result["raw"].items():
                    evidence = result.get("details")
                    if isinstance(evidence, dict):
                        parts = []
                        for inn, frame in evidence.items():
                            part = frame.copy()
                            part.insert(0, "ИНН кандидата", inn)
                            parts.append(part)
                        evidence = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
                    if isinstance(evidence, pd.DataFrame) and not evidence.empty:
                        evidence.to_excel(writer, sheet_name="Основания_" + group, index=False)
            book = load_workbook(temp)
            for sheet in book:
                sheet.freeze_panes = "A2"
                sheet.auto_filter.ref = sheet.dimensions
                for row in sheet:
                    for cell in row:
                        if isinstance(cell.value, str):
                            cell.data_type = "s"
            book.save(temp)
            book.close()
            os.replace(temp, target)
            QMessageBox.information(self, "Готово", "Признаки, статусы, основания и ограничения сохранены.")
        except Exception as exc:
            QMessageBox.warning(self, "Ошибка экспорта", str(exc))
        finally:
            if temp and os.path.exists(temp):
                os.unlink(temp)

    def reject(self):
        if self.running():
            QMessageBox.information(self, "Поиск", "Дождитесь завершения поиска перед закрытием окна.")
            return
        super().reject()

    def closeEvent(self, event):
        if self.running():
            event.ignore()
            QMessageBox.information(self, "Поиск", "Дождитесь завершения поиска перед закрытием окна.")
        else:
            event.accept()
