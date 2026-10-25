"""Надстройка над имеющимся candidate_ui: старые поиск, основания и экспорт сохранены."""
from pathlib import Path
from PySide6.QtCore import QThread, Signal, QTimer
from PySide6.QtWidgets import QLabel, QPushButton, QHBoxLayout, QFileDialog, QMessageBox
import pandas as pd
import candidate_ui as existing
from candidate_search import run_candidate_search
from catboost_ranker import DEFAULT_MODEL, HEADERS, score_candidates, export_labeling

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


existing.HEADERS.update(HEADERS)


class SearchWorker(QThread):
    progress = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, frames, source_inn, model_path, parent=None):
        super().__init__(parent)
        self.frames, self.source_inn, self.model_path = frames, source_inn, model_path

    def run(self):
        try:
            result = run_candidate_search(self.frames, self.source_inn, self.progress.emit)
            self.progress.emit("Применение CatBoost")
            result["features"], info = score_candidates(result["features"], self.model_path)
            result["metadata"]["catboost"] = info
            result["metadata"]["method"] = "Объединение точных поисков; " + info["mode"]
            result["metadata"]["catboost_model_file"] = str(self.model_path or DEFAULT_MODEL)
            if info["warning"]:
                result["warnings"].append("CatBoost: " + info["warning"])
            self.succeeded.emit(result)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class TrainingWorker(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, paths, output_dir, parent=None):
        super().__init__(parent)
        self.paths, self.output_dir = paths, output_dir

    def run(self):
        try:
            from catboost_training import train_and_save
            self.succeeded.emit(train_and_save(self.paths, self.output_dir))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class CandidateSearchDialog(existing.CandidateSearchDialog):
    def __init__(self, frames, source_inn, parent=None):
        self.model_path, self.training_worker, self.refresh_after_training = None, None, False
        super().__init__(frames, source_inn, parent)
        controls = QHBoxLayout()
        self.model_button = QPushButton("Загрузить модель CatBoost…")
        self.default_button = QPushButton("Модель по умолчанию")
        self.label_button = QPushButton("Выгрузить для разметки 0/1…")
        self.train_button = QPushButton("Обучить CatBoost…")
        for b in (self.model_button, self.default_button, self.label_button, self.train_button):
            controls.addWidget(b)
        self.layout().insertLayout(3, controls)
        self.mode_label = QLabel("CatBoost: подготовка…")
        self.mode_label.setWordWrap(True)
        self.layout().insertWidget(4, self.mode_label)
        for i in range(self.layout().count()):
            w = self.layout().itemAt(i).widget()
            if isinstance(w, QLabel) and w.text().startswith("Объединение сотрудников"):
                w.setText(
                    "Объединение сотрудников, поставщиков, покупателей и ККТ. При наличии совместимой обученной модели "
                    "CatBoost ранжирует найденных кандидатов. Без модели оценки не подставляются. "
                    "Период: вся загруженная книга. Балл модели не является доказательством или калиброванной вероятностью.")
        self.model_button.clicked.connect(self.choose_model)
        self.default_button.clicked.connect(self.use_default)
        self.label_button.clicked.connect(self.export_labels)
        self.train_button.clicked.connect(self.train_model)
        self.label_button.setEnabled(False)

    def running(self):
        return super().running() or (self.training_worker is not None and _candidate_safe_running(self.training_worker))

    def set_ml_enabled(self, enabled):
        for b in (self.model_button, self.default_button, self.train_button):
            b.setEnabled(enabled)
        self.label_button.setEnabled(enabled and self.result is not None and not self.result["features"].empty)

    def start_search(self):
        if self.running():
            return
        if self.worker is not None:
            _candidate_safe_delete(self.worker)
        self.result = None
        self.details.clear()
        self.table.setModel(existing.CandidateTableModel(pd.DataFrame(), self.table))
        self.states.setModel(existing.CandidateTableModel(pd.DataFrame(), self.states))
        self.notes.clear()
        self.retry.setEnabled(False)
        self.export.setEnabled(False)
        self.close_button.setEnabled(False)
        self.set_ml_enabled(False)
        self.status.setText("Поиск и оценка выполняются в фоновом потоке…")
        self.mode_label.setText("CatBoost: ожидает завершения поиска")
        self.worker = SearchWorker(self.frames, self.source_inn, self.model_path, self)
        self.worker.progress.connect(self.show_progress)
        self.worker.succeeded.connect(self.show_result)
        self.worker.failed.connect(self.show_error)
        self.worker.finished.connect(self.search_finished)
        self.worker.start()

    def search_finished(self):
        super().search_finished()
        self.set_ml_enabled(True)

    def show_result(self, result):
        super().show_result(result)
        info = result["metadata"]["catboost"]
        version = " · версия " + info["version"] if info["version"] else ""
        self.mode_label.setText(info["mode"] + version + ". Оценка 0–1 — приоритет проверки, не доказательство.")

    def show_error(self, message):
        super().show_error(message)
        self.mode_label.setText("CatBoost не применён: поиск не завершён.")

    def choose_model(self):
        if self.running():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Выберите доверенную модель; рядом должен быть одноимённый JSON",
                                              str(DEFAULT_MODEL.parent), "CatBoost (*.cbm)")
        if path:
            self.model_path = path
            self.start_search()

    def use_default(self):
        if not self.running():
            self.model_path = None
            self.start_search()

    def export_labels(self):
        if self.running() or self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Данные для экспертной разметки", f"labels_{self.source_inn}.xlsx",
                                              "Excel (*.xlsx)")
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        try:
            export_labeling(self.result["features"], path, self.result["metadata"])
            QMessageBox.information(self, "Готово",
                                    "Заполните метки 0/1 и даты периода. Непроверенные случаи оставьте без метки. Инструкция находится на втором листе.")
        except Exception as exc:
            QMessageBox.warning(self, "Ошибка экспорта", str(exc))

    def train_model(self):
        if self.running():
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "Выберите размеченные файлы (можно несколько)", "",
                                                "Excel (*.xlsx)")
        if not paths:
            return
        output = QFileDialog.getExistingDirectory(self, "Куда сохранить модель и отчёт",
                                                  str(DEFAULT_MODEL.parent.parent))
        if not output:
            return
        self.set_ml_enabled(False)
        for b in (self.retry, self.export, self.close_button):
            b.setEnabled(False)
        self.status.setText(
            "Обучение CatBoost в фоновом потоке. Организации не пересекаются между обучением, валидацией и тестом…")
        if self.training_worker is not None:
            _candidate_safe_delete(self.training_worker)
        self.refresh_after_training = False
        self.training_worker = TrainingWorker(paths, output, self)
        self.training_worker.succeeded.connect(self.training_succeeded)
        self.training_worker.failed.connect(self.training_failed)
        self.training_worker.finished.connect(self.training_finished)
        self.training_worker.start()

    def training_succeeded(self, result):
        self.model_path = result["model_path"]
        self.refresh_after_training = True
        report = result["report"]
        test = report.get("test_metrics", {})
        text = "Модель сохранена. Результаты текущего поиска будут пересчитаны."
        if test:
            text += f"\nОтложенный тест: ROC-AUC {test['roc_auc']:.3f}, Average Precision {test['average_precision']:.3f}."
        text += "\nЭти показатели не гарантируют качество на других данных. Подробности — training_report.json."
        QMessageBox.information(self, "CatBoost обучен", text)

    def training_failed(self, message):
        self.status.setText("Обучение CatBoost не выполнено; предыдущая модель не заменена.")
        QMessageBox.warning(self, "Ошибка обучения", message)

    def training_finished(self):
        self.retry.setEnabled(True)
        self.export.setEnabled(self.result is not None)
        self.close_button.setEnabled(True)
        self.set_ml_enabled(True)
        if self.refresh_after_training:
            QTimer.singleShot(0, self.start_search)
