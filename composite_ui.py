"""Диалог комплексного анализа кандидатов (CatBoost + Qwen)."""
from pathlib import Path
import os
import tempfile
import pandas as pd

from PySide6.QtCore import Qt, QThread, Signal, QAbstractTableModel, QModelIndex
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableView, QTabWidget, QTextEdit, QSplitter, QMessageBox,
    QFileDialog, QWidget
)

from features import extract_candidate_features
from scorer import score_candidates
from llm_advisor import generate_conclusion, DEFAULT_MODEL


class TableModel(QAbstractTableModel):
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
        val = self.frame.iat[index.row(), index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            if isinstance(val, float):
                return f"{val:.2f}"
            return str(val)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self.frame.columns[section])
            return str(section + 1)
        return None


class LLMWorker(QThread):
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, source_inn, candidate_inn, candidate_row, warnings):
        super().__init__()
        self.source_inn = source_inn
        self.candidate_inn = candidate_inn
        self.candidate_row = candidate_row
        self.warnings = warnings

    def run(self):
        try:
            text = generate_conclusion(
                self.source_inn,
                self.candidate_inn,
                self.candidate_row,
                self.warnings,
            )
            self.finished.emit(text)
        except Exception as exc:
            self.failed.emit(str(exc))


class CompositeAnalysisDialog(QDialog):
    def __init__(self, frames, source_inn, parent=None):
        super().__init__(parent)
        self.frames = frames
        self.source_inn = source_inn
        self.llm_worker = None

        self.setWindowTitle(f"Комплексный анализ кандидатов · ИНН {source_inn}")
        self.resize(1300, 850)

        main_layout = QVBoxLayout(self)

        # 1. Расчёт признаков и скоринг
        raw_features, self.raw_results, self.warnings = extract_candidate_features(self.frames, self.source_inn)
        self.scored_df, self.mode_name = score_candidates(raw_features)

        header_box = QHBoxLayout()
        info_label = QLabel(
            f"<b>Исходная организация:</b> ИНН {source_inn} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Кандидатов:</b> {len(self.scored_df)} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Режим оценки:</b> {self.mode_name}"
        )
        header_box.addWidget(info_label)
        header_box.addStretch()

        self.export_btn = QPushButton("Экспорт датасета для CatBoost в Excel…")
        self.export_btn.clicked.connect(self.export_dataset)
        header_box.addWidget(self.export_btn)
        main_layout.addLayout(header_box)

        # 2. Основной сплиттер (Сводная таблица сверху, детали и ИИ снизу)
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Верхняя таблица кандидатов
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        table_title = QLabel("Рейтинг кандидатов по 4 критериям (выберите строку для просмотра оснований):")
        top_layout.addWidget(table_title)

        self.ranking_table = QTableView()
        self.ranking_table.setModel(TableModel(self.scored_df, self.ranking_table))
        self.ranking_table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.ranking_table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.ranking_table.setAlternatingRowColors(True)
        self.ranking_table.selectionModel().currentRowChanged.connect(self.on_candidate_selected)
        for i in range(len(self.scored_df.columns)):
            self.ranking_table.setColumnWidth(i, 160)
        top_layout.addWidget(self.ranking_table)
        splitter.addWidget(top_widget)

        # Нижняя часть: Сплиттер слева (детали по каналам), справа (ИИ-заключение)
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Детализация (вкладки)
        self.detail_tabs = QTabWidget()
        bottom_splitter.addWidget(self.detail_tabs)

        # Блок ИИ-заключения
        ai_box = QWidget()
        ai_layout = QVBoxLayout(ai_box)
        ai_layout.setContentsMargins(0, 0, 0, 0)

        ai_header = QHBoxLayout()
        self.ai_btn = QPushButton("Сформировать ИИ-заключение (Qwen)")
        self.ai_btn.clicked.connect(self.request_ai_conclusion)
        ai_header.addWidget(self.ai_btn)
        self.ai_status = QLabel("Готов к анализу.")
        ai_header.addWidget(self.ai_status)
        ai_header.addStretch()
        ai_layout.addLayout(ai_header)

        self.ai_text = QTextEdit()
        self.ai_text.setReadOnly(True)
        self.ai_text.setPlaceholderText(
            "Здесь появится заключение локальной языковой модели Qwen по выбранной организации.\n"
            "Модель сопоставит установленные факты, предложит альтернативные объяснения "
            "и сформирует контрольные мероприятия."
        )
        ai_layout.addWidget(self.ai_text)
        bottom_splitter.addWidget(ai_box)

        bottom_splitter.setSizes([650, 600])
        splitter.addWidget(bottom_splitter)
        splitter.setSizes([320, 480])
        main_layout.addWidget(splitter)

        if not self.scored_df.empty:
            self.ranking_table.selectRow(0)

    def current_candidate_row(self):
        indexes = self.ranking_table.selectionModel().selectedRows()
        if not indexes:
            return None
        row_idx = indexes[0].row()
        return self.scored_df.iloc[row_idx]

    def on_candidate_selected(self, current, _):
        if not current.isValid():
            return
        candidate_inn = str(self.scored_df.iat[current.row(), 0])
        self.ai_status.setText(f"Выбран кандидат: {candidate_inn}")
        self.update_detail_tabs(candidate_inn)

    def update_detail_tabs(self, candidate_inn):
        self.detail_tabs.clear()

        # Персонал
        staff_data = self.raw_results.get("staff")
        if staff_data and candidate_inn in staff_data.get("details", {}):
            df = staff_data["details"][candidate_inn]
            tv = QTableView()
            tv.setModel(TableModel(df, tv))
            self.detail_tabs.addTab(tv, f"Сотрудники ({len(df)})")

        # Поставщики
        supp_data = self.raw_results.get("suppliers")
        if supp_data and not supp_data["details"].empty:
            df = supp_data["details"].loc[supp_data["details"]["ИНН кандидата"] == candidate_inn]
            if not df.empty:
                tv = QTableView()
                tv.setModel(TableModel(df, tv))
                self.detail_tabs.addTab(tv, f"Поставщики ({len(df)})")

        # Покупатели
        buy_data = self.raw_results.get("buyers")
        if buy_data and not buy_data["details"].empty:
            df = buy_data["details"].loc[buy_data["details"]["ИНН кандидата"] == candidate_inn]
            if not df.empty:
                tv = QTableView()
                tv.setModel(TableModel(df, tv))
                self.detail_tabs.addTab(tv, f"Покупатели ({len(df)})")

        # ККТ
        kkt_data = self.raw_results.get("kkt")
        if kkt_data and not kkt_data["details"].empty:
            df = kkt_data["details"].loc[kkt_data["details"]["ИНН кандидата"] == candidate_inn]
            if not df.empty:
                tv = QTableView()
                tv.setModel(TableModel(df, tv))
                self.detail_tabs.addTab(tv, f"ККТ ({len(df)})")

        if self.detail_tabs.count() == 0:
            lbl = QLabel("Детальные основания по данному кандидату отсутствуют.")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.detail_tabs.addTab(lbl, "Нет данных")

    def request_ai_conclusion(self):
        row = self.current_candidate_row()
        if row is None:
            QMessageBox.information(self, "Кандидат не выбран", "Выберите строку кандидата в верхней таблице.")
            return

        candidate_inn = str(row["ИНН кандидата"])
        self.ai_btn.setEnabled(False)
        self.ai_status.setText("Генерация заключения Qwen (подождите 10-30 сек)…")
        self.ai_text.setPlainText("Локальная модель обрабатывает рассчитанные показатели...")

        self.llm_worker = LLMWorker(self.source_inn, candidate_inn, row.to_dict(), self.warnings)
        self.llm_worker.finished.connect(self.on_ai_finished)
        self.llm_worker.failed.connect(self.on_ai_failed)
        self.llm_worker.start()

    def on_ai_finished(self, text):
        self.ai_btn.setEnabled(True)
        self.ai_status.setText("Заключение успешно сформировано.")
        self.ai_text.setPlainText(text)

    def on_ai_failed(self, error_message):
        self.ai_btn.setEnabled(True)
        self.ai_status.setText("Ошибка формирования заключения.")
        self.ai_text.setPlainText(f"Не удалось получить ответ модели:\n\n{error_message}")
        QMessageBox.warning(self, "Ошибка ИИ", error_message)

    def export_dataset(self):
        """Выгружает таблицу признаков с пустой колонкой метки для разметки экспертами."""
        if self.scored_df.empty:
            QMessageBox.information(self, "Экспорт", "Таблица признаков пуста.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить датасет признаков для CatBoost",
            f"dataset_{self.source_inn}.xlsx",
            "Книга Excel (*.xlsx)",
        )
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"

        export_df = self.scored_df.copy()
        export_df.insert(0, "Исходный ИНН", self.source_inn)
        # Колонка для разметки экспертом: 1 = подтверждён переход, 0 = обычные отношения
        export_df.insert(3, "Метка_целевая (1/0)", "")

        try:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                export_df.to_excel(writer, sheet_name="Для_разметки_CatBoost", index=False)
                pd.DataFrame({
                    "Инструкция": [
                        "Заполните колонку 'Метка_целевая (1/0)':",
                        "1 — эксперт подтверждает совокупность признаков перехода бизнеса.",
                        "0 — совпадения признаны добросовестными/случайными.",
                        "Оставьте пустым, если данных недостаточно.",
                        "После разметки 30-50 строк запустите скрипт train_catboost.py."
                    ]
                }).to_excel(writer, sheet_name="Инструкция", index=False)
            QMessageBox.information(
                self, "Успешно",
                "Датасет выгружен. Разметьте колонку 'Метка_целевая (1/0)' для обучения CatBoost."
            )
        except Exception as exc:
            QMessageBox.warning(self, "Ошибка экспорта", str(exc))