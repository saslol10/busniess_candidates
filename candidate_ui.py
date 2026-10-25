"""Окно общего поиска. Qt-виджеты создаются только в основном потоке."""
import json
import os
import tempfile
from pathlib import Path
import pandas as pd
from PySide6.QtCore import Qt, QThread, Signal, QAbstractTableModel, QModelIndex, QTimer
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QTabWidget, QTableView, QPlainTextEdit, QSplitter, QWidget, QMessageBox, QFileDialog)
from candidate_search import run_candidate_search, DIRECTIONS
from llm_advisor import generate_conclusion, DEFAULT_MODEL

from shiboken6 import isValid as _candidate_qt_is_valid


def _candidate_safe_running(worker):
    if worker is None:
        return False
    try:
        return bool(_candidate_qt_is_valid(worker) and worker.isRunning())
    except RuntimeError:
        return False


def _candidate_safe_delete(worker):
    if worker is None:
        return
    try:
        if _candidate_qt_is_valid(worker) and not worker.isRunning():
            worker.deleteLater()
    except RuntimeError:
        pass


STATUS = {"completed": "Проверено", "no_data": "Нет данных", "error": "Ошибка"}
HEADERS = {"source_inn": "ИНН исходной", "candidate_inn": "ИНН кандидата",
           "staff_common_count": "Общие сотрудники", "staff_source_share_pct": "Доля сотрудников исходной, %",
           "staff_regular_common_count": "Регулярные у обеих организаций",
           "staff_sequential_count": "Последовательные периоды в выгрузке",
           "suppliers_common_count": "Общие поставщики", "buyers_common_count": "Общие покупатели",
           "kkt_common_count": "Общие ККТ", "kkt_source_share_pct": "Доля ККТ исходной, %"}
for group, name in DIRECTIONS.items():
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
        if section < 0:
            return None
        if orientation == Qt.Orientation.Vertical:
            if section >= len(self.frame):
                return None
            return str(section + 1)
        if orientation == Qt.Orientation.Horizontal:
            if section >= len(self.frame.columns):
                return None
            name = str(self.frame.columns[section])
            return HEADERS.get(name, name) if self.headers and role == Qt.ItemDataRole.DisplayRole else name
        return None


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


class LLMWorker(QThread):
    """Генерация заключения в отдельном потоке, без блокировки Qt-интерфейса."""
    succeeded = Signal(str)
    failed = Signal(str)

    def __init__(self, source_inn, candidate_inn, candidate_row, warnings,
                 states, evidence_by_group, parent=None):
        super().__init__(parent)
        self.source_inn = source_inn
        self.candidate_inn = candidate_inn
        self.candidate_row = candidate_row
        self.warnings = warnings
        self.states = states
        self.evidence_by_group = evidence_by_group

    def run(self):
        try:
            text = generate_conclusion(
                self.source_inn,
                self.candidate_inn,
                self.candidate_row,
                self.warnings,
                states=self.states,
                evidence_by_group=self.evidence_by_group,
                model_name=DEFAULT_MODEL,
            )
            self.succeeded.emit(text)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class CandidateSearchDialog(QDialog):
    def __init__(self, frames, source_inn, parent=None):
        super().__init__(parent)
        self.frames, self.source_inn = frames, source_inn
        self.worker, self.result = None, None
        self.llm_worker = None
        self.setWindowTitle("Общий поиск кандидатов · ИНН " + source_inn)
        self.resize(1280, 850)
        layout = QVBoxLayout(self)
        note = QLabel("Объединение сотрудников, поставщиков, покупателей и ККТ. "
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
        self.ai_button = QPushButton("Сформировать заключение Qwen")
        self.ai_button.setEnabled(False)
        self.ai_button.setToolTip(f"Локальная модель Ollama: {DEFAULT_MODEL}")
        self.ai_button.clicked.connect(self.request_ai_conclusion)
        self.close_button = QPushButton("Закрыть")
        self.close_button.clicked.connect(self.reject)
        actions.addWidget(self.retry)
        actions.addWidget(self.export)
        actions.addWidget(self.ai_button)
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
        self.ai_status = QLabel("Заключение Qwen не запрашивалось.")
        self.ai_status.setWordWrap(True)
        layout.addWidget(self.ai_status)
        self.ai_text = QPlainTextEdit()
        self.ai_text.setReadOnly(True)
        self.ai_text.setPlaceholderText("Здесь появится заключение локальной модели Qwen по выбранному кандидату.")
        self.ai_text.setMaximumHeight(220)
        layout.addWidget(self.ai_text)
        self.notes = QPlainTextEdit()
        self.notes.setReadOnly(True)
        self.notes.setMaximumHeight(115)
        layout.addWidget(self.notes)
        QTimer.singleShot(0, self.start_search)

    def running(self):
        return self.worker is not None and _candidate_safe_running(self.worker)

    def start_search(self):
        if self.running():
            return
        if self.worker is not None:
            _candidate_safe_delete(self.worker)
        self.result = None
        self.details.clear()
        self.table.setModel(CandidateTableModel(pd.DataFrame(), self.table))
        self.states.setModel(CandidateTableModel(pd.DataFrame(), self.states))
        self.notes.clear()
        self.ai_text.clear()
        self.ai_status.setText("Заключение Qwen не запрашивалось.")
        self.ai_button.setEnabled(False)
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
        self.ai_button.setEnabled(self.result is not None and not self.result["features"].empty)

    def show_error(self, message):
        self.status.setText("Общий поиск не выполнен.")
        self.notes.setPlainText(message)
        self.ai_button.setEnabled(False)

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
                            + ("Выберите кандидата для просмотра оснований." if len(features) else
                               "Кандидаты не найдены в доступных данных. Это не доказывает отсутствие перехода."))
        self.notes.setPlainText("\n".join(result["warnings"]) or "Дополнительных предупреждений нет.")
        if not features.empty:
            self.table.selectRow(0)
        self.ai_button.setEnabled(not features.empty)

    def select_candidate(self, current, previous=QModelIndex()):
        self.details.clear()
        if not current.isValid() or self.result is None:
            self.ai_button.setEnabled(False)
            return
        inn = str(self.result["features"].iloc[current.row()]["candidate_inn"])
        for group, title in DIRECTIONS.items():
            row = self.result["states"].loc[self.result["states"]["Код"] == group].iloc[0]
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

    def _selected_candidate(self):
        if self.result is None:
            return None, None
        current = self.table.currentIndex()
        if not current.isValid():
            return None, None
        frame = self.result["features"]
        if current.row() >= len(frame):
            return None, None
        row = frame.iloc[current.row()]
        return str(row["candidate_inn"]), row

    def _evidence_for_candidate(self, candidate_inn):
        evidence = {}
        for group in DIRECTIONS:
            raw_result = self.result["raw"].get(group, {}) if self.result else {}
            details = raw_result.get("details") if isinstance(raw_result, dict) else None
            frame = pd.DataFrame()
            if isinstance(details, dict):
                frame = details.get(candidate_inn, pd.DataFrame())
            elif isinstance(details, pd.DataFrame) and "ИНН кандидата" in details.columns:
                frame = details.loc[details["ИНН кандидата"].astype(str) == candidate_inn]
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                evidence[group] = frame.copy()
        return evidence

    def request_ai_conclusion(self):
        if self.running():
            return
        if self.llm_worker is not None and _candidate_safe_running(self.llm_worker):
            return
        candidate_inn, row = self._selected_candidate()
        if row is None:
            QMessageBox.information(self, "Кандидат не выбран", "Выберите строку кандидата.")
            return
        states = self.result["states"].to_dict("records")
        evidence = self._evidence_for_candidate(candidate_inn)
        self.ai_button.setEnabled(False)
        self.ai_status.setText(f"Qwen ({DEFAULT_MODEL}) формирует заключение для ИНН {candidate_inn}…")
        self.ai_text.setPlainText("Ожидание ответа локальной модели…")
        self.llm_worker = LLMWorker(
            self.source_inn,
            candidate_inn,
            row.to_dict(),
            self.result.get("warnings", []),
            states,
            evidence,
            self,
        )
        self.llm_worker.succeeded.connect(self.show_ai_result)
        self.llm_worker.failed.connect(self.show_ai_error)
        self.llm_worker.finished.connect(self.ai_finished)
        self.llm_worker.start()

    def ai_finished(self):
        if self.result is not None and not self.result["features"].empty:
            self.ai_button.setEnabled(True)

    def show_ai_result(self, text):
        self.ai_status.setText(f"Заключение сформировано локальной моделью {DEFAULT_MODEL}.")
        self.ai_text.setPlainText(text)

    def show_ai_error(self, message):
        self.ai_status.setText("Не удалось сформировать заключение Qwen.")
        self.ai_text.setPlainText("Ошибка:\n\n" + message)

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
                pd.DataFrame(metadata.items(), columns=["Параметр", "Значение"]).to_excel(writer,
                                                                                          sheet_name="Параметры",
                                                                                          index=False)
                pd.DataFrame({"Предупреждения": self.result["warnings"]}).to_excel(writer, sheet_name="Ограничения",
                                                                                   index=False)
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
        if self.running() or (self.llm_worker is not None and _candidate_safe_running(self.llm_worker)):
            QMessageBox.information(self, "Поиск", "Дождитесь завершения фоновой операции перед закрытием окна.")
            return
        super().reject()

    def closeEvent(self, event):
        if self.running() or (self.llm_worker is not None and _candidate_safe_running(self.llm_worker)):
            event.ignore()
            QMessageBox.information(self, "Поиск", "Дождитесь завершения фоновой операции перед закрытием окна.")
        else:
            event.accept()


def _install_candidate_roomy_window():
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import (QDialog, QSplitter, QPlainTextEdit, QVBoxLayout,
                                   QHBoxLayout, QWidget, QPushButton, QLabel)
    def decorate(cls):
        original = cls.__init__

        def init(self, *args, **kwargs):
            original(self, *args, **kwargs)
            self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint
                                | Qt.WindowType.WindowMinimizeButtonHint)

            def arrange():
                if getattr(self, '_roomy_applied_v1', False):
                    return
                self._roomy_applied_v1 = True
                self.showMaximized()
                states = getattr(self, 'states', None)
                table = getattr(self, 'table', None)
                details = getattr(self, 'details', None)
                notes = getattr(self, 'notes', None)
                if states is not None:
                    def compact_states(*unused):
                        model = states.model()
                        rows = model.rowCount() if model is not None else 4
                        height = states.horizontalHeader().height() + sum(
                            states.rowHeight(i) for i in range(min(rows, 4))) + 6
                        states.setMinimumHeight(65)
                        states.setMaximumHeight(max(65, min(height, 160)))

                    compact_states()
                    QTimer.singleShot(1500, compact_states)
                    model = states.model()
                    if model is not None:
                        model.modelReset.connect(compact_states)
                        model.rowsInserted.connect(compact_states)
                if table is not None:
                    table.setMinimumHeight(135)
                    table.verticalHeader().setDefaultSectionSize(30)
                    table.horizontalHeader().setMinimumSectionSize(100)
                    table.horizontalHeader().setStretchLastSection(False)
                if details is not None:
                    details.setMinimumHeight(160)
                if notes is not None:
                    notes.setMinimumHeight(40)
                    notes.setMaximumHeight(75)
                splitter = None
                if table is not None:
                    p = table.parentWidget()
                    while p is not None and p is not self:
                        if isinstance(p, QSplitter) and p.orientation() == Qt.Orientation.Vertical:
                            splitter = p
                            break
                        p = p.parentWidget()
                if splitter is not None:
                    splitter.setChildrenCollapsible(False)
                    splitter.setHandleWidth(9)
                    editors = [w for w in self.findChildren(QPlainTextEdit)
                               if w is not notes and not splitter.isAncestorOf(w)]
                    editor = next((w for w in editors if any(word in
                                                             (
                                                                     w.placeholderText() + ' ' + w.toPlainText() + ' ' + w.objectName()).lower()
                                                             for word in ('qwen', 'заключен', 'llm'))), None)
                    if editor is not None:
                        old_layout = editor.parentWidget().layout()
                        if old_layout is not None:
                            old_layout.removeWidget(editor)
                        panel = QWidget()
                        box = QVBoxLayout(panel)
                        box.setContentsMargins(0, 0, 0, 0)
                        row = QHBoxLayout()
                        row.addWidget(QLabel('Заключение Qwen'))
                        toggle = QPushButton('Свернуть текст')
                        row.addStretch()
                        row.addWidget(toggle)
                        box.addLayout(row)
                        box.addWidget(editor)
                        editor.setMinimumHeight(65)
                        editor.setMaximumHeight(16777215)

                        def toggle_editor():
                            visible = editor.isHidden()
                            editor.setVisible(visible)
                            toggle.setText('Свернуть текст' if visible else 'Показать текст')

                        toggle.clicked.connect(toggle_editor)
                        splitter.addWidget(panel)
                    count = splitter.count()
                    if count:
                        splitter.setSizes([300, 270] + [150] * max(0, count - 2))
                        for i in range(count):
                            splitter.setStretchFactor(i, 3 if i < 2 else 1)
                self.resize(self.size())

            QTimer.singleShot(0, arrange)

        cls.__init__ = init

    for name in ['CandidateSearchDialog']:
        cls = globals().get(name)
        if isinstance(cls, type) and issubclass(cls, QDialog):
            decorate(cls)


_install_candidate_roomy_window()
