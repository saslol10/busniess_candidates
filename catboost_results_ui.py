from PySide6.QtWidgets import QDialog, QVBoxLayout, QPlainTextEdit
"""Дополнительная таблица CatBoost поверх существующего окна поиска."""
import importlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QThread, Signal, QAbstractTableModel
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QTableView, QFileDialog, QMessageBox, QHeaderView, QDoubleSpinBox)
from shiboken6 import isValid

try:
    from catboost_results_config import BASE_MODULE, BASE_CLASS
except ImportError:
    BASE_MODULE, BASE_CLASS = "candidate_ui", "CandidateSearchDialog"
BaseDialog = getattr(importlib.import_module(BASE_MODULE), BASE_CLASS)
COLUMNS = ["Место", "ИНН кандидата", "Оценка CatBoost, 0–1"]


def build_results(features, scores, low=0.20, high=0.70):
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(scores) != len(features) or not np.isfinite(scores).all():
        raise ValueError("Модель вернула неверное количество оценок или нечисловые оценки")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Оценки должны быть в диапазоне 0–1")
    frame = pd.DataFrame({"ИНН кандидата": features["candidate_inn"].astype(str).to_numpy(),
                          "Оценка CatBoost, 0–1": scores})
    frame = frame.sort_values(["Оценка CatBoost, 0–1", "ИНН кандидата"],
                              ascending=[False, True], kind="stable").reset_index(drop=True)
    frame.insert(0, "Место", frame["Оценка CatBoost, 0–1"].rank(method="min", ascending=False).astype(int))

    return frame[COLUMNS]


class ResultsModel(QAbstractTableModel):
    def __init__(self, frame, parent=None):
        super().__init__(parent)
        self.frame = frame

    def rowCount(self, parent=None):
        return 0 if parent is not None and parent.isValid() else len(self.frame)

    def columnCount(self, parent=None):
        return 0 if parent is not None and parent.isValid() else len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole or section < 0:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return COLUMNS[section] if section < len(COLUMNS) else None
        return str(section + 1) if section < len(self.frame) else None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self.frame) or not 0 <= index.column() < len(COLUMNS):
            return None
        value = self.frame.iat[index.row(), index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            return f"{float(value):.4f}" if index.column() == 2 else str(value)
        if role == Qt.ItemDataRole.TextAlignmentRole and index.column() in (0, 2):
            return Qt.AlignmentFlag.AlignCenter
        return None


class PredictWorker(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, path, features, parent=None):
        super().__init__(parent)
        self.path, self.features = path, features.copy(deep=True)

    def run(self):
        try:
            from catboost import CatBoostClassifier
            model = CatBoostClassifier()
            model.load_model(self.path)
            classes = list(model.classes_)
            if len(classes) != 2 or {str(c) for c in classes} not in ({"0", "1"}, {"0.0", "1.0"}):
                raise ValueError("Нужна бинарная модель с классами 0 и 1")
            names = list(model.feature_names_)
            if not names or all(str(n).isdigit() for n in names):
                schema = Path(self.path).with_suffix(".features.json")
                if not schema.exists():
                    raise ValueError("В модели нет названий признаков. Нужен файл " + schema.name +
                                     ' с точным списком признаков в порядке обучения: ["staff_common_count", ...]. Порядок угадываться не будет.')
                names = json.loads(schema.read_text(encoding="utf-8-sig"))
                if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
                    raise ValueError("Некорректный JSON со схемой признаков")
            if len(set(names)) != len(names) or not names:
                raise ValueError("Пустая схема признаков или повторяющиеся названия")
            if {"source_inn", "candidate_inn"}.intersection(names):
                raise ValueError("Модель использует ИНН как признаки. Переобучите её без идентификаторов")
            missing = [n for n in names if n not in self.features.columns]
            if missing:
                raise ValueError("Отсутствуют признаки модели: " + ", ".join(missing))
            x = self.features.loc[:, names].apply(pd.to_numeric, errors="raise").astype(float)
            if np.isinf(x.to_numpy()).any():
                raise ValueError("В признаках есть бесконечные значения")
            positive = next(i for i, c in enumerate(classes) if str(c) in ("1", "1.0"))
            scores = model.predict_proba(x)[:, positive]
            self.succeeded.emit((self.features, scores))
        except Exception as exc:
            self.failed.emit(str(exc))


class CandidateSearchDialog(BaseDialog):
    def __init__(self, *args, **kwargs):
        self._cb_worker = None
        self._cb_path = None
        self._cb_payload = None
        self._cb_frame = pd.DataFrame(columns=COLUMNS)
        super().__init__(*args, **kwargs)
        panel = QWidget(self)
        self._cb_panel = panel
        panel.setObjectName("catboostResultsPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("<b>Результат работы CatBoost</b>"))
        buttons = QHBoxLayout()
        self._cb_load = QPushButton("Выбрать модель .cbm…")
        self._cb_run = QPushButton("Рассчитать оценки")
        self._cb_save = QPushButton("Сохранить результат CatBoost…")
        self._cb_train = QPushButton("Обучить CatBoost…")
        self._cb_train.clicked.connect(self._train_model)
        self._cb_qwen = QPushButton("Сформировать заключение Qwen")
        self._cb_qwen.clicked.connect(self._generate_qwen)
        self._cb_save.setEnabled(False)
        self._cb_low, self._cb_high = QDoubleSpinBox(), QDoubleSpinBox()
        for spin, value in ((self._cb_low, .20), (self._cb_high, .70)):
            spin.setRange(0, 1)
            spin.setDecimals(2)
            spin.setSingleStep(.05)
            spin.setValue(value)
            spin.valueChanged.connect(self._cb_render)
        for button in (self._cb_load, self._cb_run, self._cb_save):
            buttons.addWidget(button)
        buttons.addStretch()
        layout.addLayout(buttons)
        self._cb_status = QLabel("Модель не выбрана. Демонстрационные оценки не подставляются.")
        self._cb_status.setWordWrap(True)
        layout.addWidget(self._cb_status)
        self._cb_table = QTableView()
        self._cb_table.setMinimumHeight(0)
        self._cb_table.setWordWrap(True)
        self._cb_table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self._cb_table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._cb_table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self._cb_table.verticalHeader().setVisible(False)
        self._cb_table.setStyleSheet(
            "QTableView { background:white; color:#111; gridline-color:#777; } QHeaderView::section { background:#f4f4f4; color:#111; padding:5px; border:1px solid #aaa; }")
        layout.addWidget(self._cb_table, 1)
        note = QLabel("Балл класса 1 — не доказательство перехода бизнеса и не подтверждённая калиброванная вероятность. Отображение: 4 знака; Excel сохраняет исходную точность.")
        note.setWordWrap(True)
        layout.addWidget(note)
        root = self.layout()
        if hasattr(root, "insertWidget"):
            root.insertWidget(max(0, root.count() - 1), panel)
        else:
            root.addWidget(panel)
        self._cb_load.clicked.connect(self._cb_choose)
        self._cb_run.clicked.connect(self._cb_predict)
        self._cb_save.clicked.connect(self._cb_export)
        self._cb_display(self._cb_frame)

    def _cb_busy(self):
        try:
            return self._cb_worker is not None and isValid(self._cb_worker) and self._cb_worker.isRunning()
        except RuntimeError:
            return False

    def running(self):
        return self._cb_busy() or super().running()

    def start_search(self):
        if self._cb_busy():
            QMessageBox.information(self, "CatBoost", "Дождитесь завершения оценки модели.")
            return
        if hasattr(self, "_cb_table"):
            self._cb_payload = None
            self._cb_display(pd.DataFrame(columns=COLUMNS))
            self._cb_status.setText("Поиск обновляется; предыдущие оценки сброшены.")
        return super().start_search()

    def show_result(self, result):
        super().show_result(result)
        if hasattr(self, "_cb_table"):
            self._cb_payload = None
            self._cb_display(pd.DataFrame(columns=COLUMNS))
            self._cb_status.setText("Кандидаты обновлены. Нажмите «Рассчитать оценки».")

    def _cb_choose(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выбрать модель CatBoost", "", "CatBoost (*.cbm)")
        if path:
            self._cb_path = path
            self._cb_payload = None
            self._cb_display(pd.DataFrame(columns=COLUMNS))
            self._cb_status.setText("Выбрана модель: " + Path(path).name)
            self._cb_status.setText("Выбрана модель: " + Path(path).name + ". Нажмите «Рассчитать оценки».")

    def _cb_predict(self):
        if self._cb_busy():
            return
        if not self._cb_path:
            QMessageBox.information(self, "CatBoost", "Сначала выберите обученную модель .cbm.")
            return
        result = getattr(self, "result", None)
        features = result.get("features") if isinstance(result, dict) else None
        if not isinstance(features, pd.DataFrame) or features.empty:
            self._cb_status.setText("Нет рассчитанных кандидатов. Завершите общий поиск.")
            return
        if "candidate_inn" not in features:
            QMessageBox.warning(self, "CatBoost", "В признаках нет candidate_inn.")
            return
        self._cb_payload = None
        self._cb_display(pd.DataFrame(columns=COLUMNS))
        self._cb_load.setEnabled(False)
        self._cb_run.setEnabled(False)
        self._cb_status.setText("CatBoost рассчитывает оценки в фоновом потоке…")
        worker = PredictWorker(self._cb_path, features, self)
        self._cb_worker = worker
        worker.succeeded.connect(self._cb_success)
        worker.failed.connect(self._cb_error)
        worker.finished.connect(self._cb_finished)
        worker.start()

    def _cb_success(self, payload):
        # One prediction supplies both tables. Never run a second scorer here.
        features, scores = payload
        current = getattr(self, "result", None)
        if not isinstance(current, dict):
            self._cb_error("Основной результат поиска недоступен.")
            return
        live = current.get("features")
        keys = ["source_inn", "candidate_inn"]
        if (not isinstance(live, pd.DataFrame) or
            len(live) != len(features) or
            live[keys].astype(str).values.tolist() != features[keys].astype(str).values.tolist()):
            self._cb_error("Список кандидатов изменился. Нажмите «Рассчитать оценки» повторно.")
            return
        scores = np.asarray(scores, dtype=float).reshape(-1)
        if len(scores) != len(features) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            self._cb_error("Модель вернула некорректные оценки.")
            return
        ranked = features.copy(deep=True)
        ranked["catboost_score"] = scores
        ranked["catboost_rank"] = ranked["catboost_score"].rank(method="min", ascending=False).astype(int)
        import hashlib
        fingerprint = hashlib.sha256(Path(self._cb_path).read_bytes()).hexdigest()[:12]
        version = Path(self._cb_path).name + " · SHA256 " + fingerprint
        mode = "CatBoost · балл класса 1; калибровка не подтверждена"
        ranked["catboost_mode"] = mode
        ranked["catboost_version"] = version
        ranked = ranked.sort_values(["catboost_score", "candidate_inn"], ascending=[False, True], kind="stable").reset_index(drop=True)
        updated = dict(current)
        updated["features"] = ranked
        updated["metadata"] = dict(current.get("metadata", {}))
        updated["metadata"]["catboost_active_prediction"] = {
            "applied": True, "model_path": str(self._cb_path), "version": version,
            "mode": mode, "validation_status": "not_verified_by_this_adapter"}
        # Call the original renderer, not self.show_result(): prevents recursion,
        # loss of manually added rows and repeated prediction.
        BaseDialog.show_result(self, updated)
        self.result = updated
        if hasattr(self, "status"):
            self.status.setText("Рейтинг CatBoost обновлён в обеих таблицах. Балл не является доказательством или калиброванной вероятностью.")
        self._cb_payload = (features.copy(deep=True), scores.copy())
        self._cb_render()

    def _cb_error(self, message):
        self._cb_status.setText("Оценка не выполнена: " + message)

    def _cb_finished(self):
        self._cb_load.setEnabled(True)
        self._cb_run.setEnabled(True)
        old = self._cb_worker
        self._cb_worker = None
        if old is not None and isValid(old):
            old.deleteLater()

    def _cb_render(self, *args):
        if self._cb_payload is None:
            return
        try:
            frame = build_results(*self._cb_payload)
            self._cb_display(frame)
            self._cb_status.setText("Модель: " + Path(self._cb_path).name + " · кандидатов: " + str(len(frame)) +
                                    " · сортировка по убыванию оценки класса 1")
        except Exception as exc:
            self._cb_display(pd.DataFrame(columns=COLUMNS))
            self._cb_status.setText(str(exc))

    def _cb_display(self, frame):
        self._cb_frame = frame
        old = self._cb_table.model()
        self._cb_table.setModel(ResultsModel(frame, self._cb_table))
        if old is not None:
            old.deleteLater()
        header = self._cb_table.horizontalHeader()
        for col in range(3):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._cb_table.resizeRowsToContents()
        self._cb_table.selectionModel().currentRowChanged.connect(self._cb_select)
        self._cb_save.setEnabled(not frame.empty)

    def _cb_select(self, current, previous):
        if not current.isValid() or self.result is None:
            return
        inn = str(self._cb_frame.iloc[current.row()]["ИНН кандидата"])
        # Исходная таблица не переставляется: основания остаются привязаны к правильной строке.
        model = self.table.model()
        frame = getattr(model, "frame", None)
        if isinstance(frame, pd.DataFrame) and "candidate_inn" in frame:
            matches = np.flatnonzero(frame["candidate_inn"].astype(str).to_numpy() == inn)
            if len(matches):
                self.table.selectRow(int(matches[0]))
                self.table.scrollTo(model.index(int(matches[0]), 0))

    def _cb_export(self):
        if self._cb_frame.empty:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить оценки CatBoost", "catboost_results.xlsx", "Excel (*.xlsx)")
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        import tempfile, os
        temp = None
        try:
            from openpyxl.styles import Font
            fd, temp = tempfile.mkstemp(suffix=".xlsx", dir=Path(path).parent)
            os.close(fd)
            data = self._cb_frame.copy()
            features, scores = self._cb_payload
            if "source_inn" in features:
                sources = features[["candidate_inn", "source_inn"]].copy()
                sources["candidate_inn"] = sources["candidate_inn"].astype(str)
                if sources["candidate_inn"].duplicated().any():
                    raise ValueError("Повторяющиеся ИНН кандидатов: сохранение остановлено.")
                lookup = sources.set_index("candidate_inn")["source_inn"]
                data.insert(1, "ИНН исходный", data["ИНН кандидата"].map(lookup).astype(str))
            meta = getattr(self, "result", {}).get("metadata", {}).get("catboost_active_prediction", {})
            with pd.ExcelWriter(temp, engine="openpyxl") as writer:
                data.to_excel(writer, sheet_name="Оценки CatBoost", index=False)
                pd.DataFrame({"Параметр": ["Модель", "Версия / SHA256", "Точность", "Ограничение"],
                    "Значение": [str(self._cb_path), meta.get("version", ""),
                    "В ячейках сохранены исходные числовые оценки; отображаются 4 знака",
                    "Балл не является доказательством или подтверждённой калиброванной вероятностью"]}).to_excel(writer, sheet_name="Параметры", index=False)
                ws = writer.sheets["Оценки CatBoost"]
                ws.freeze_panes = "A2"
                ws.auto_filter.ref = ws.dimensions
                for i, name in enumerate(data.columns, 1):
                    from openpyxl.utils import get_column_letter
                    ws.column_dimensions[get_column_letter(i)].width = 29 if "Оценка" in name else 21
                    ws.cell(1, i).font = Font(bold=True)
                    for row in range(2, ws.max_row + 1):
                        cell = ws.cell(row, i)
                        if "Оценка" in name:
                            cell.number_format = "0.0000"
                        elif "ИНН" in name:
                            cell.number_format = "@"
                            cell.data_type = "s"
                for sheet in writer.sheets.values():
                    for row in sheet:
                        for cell in row:
                            if isinstance(cell.value, str):
                                cell.data_type = "s"
            os.replace(temp, path)
            temp = None
            QMessageBox.information(self, "Готово", "Оценки CatBoost сохранены без округления исходных значений.")
        except Exception as exc:
            QMessageBox.warning(self, "Ошибка экспорта", str(exc))
        finally:
            if temp and os.path.exists(temp):
                os.unlink(temp)


# --- Added active-window handlers (the class is imported by manual_candidates_ui.py) ---
def _fixed_train_model(self):
    paths, _ = QFileDialog.getOpenFileNames(
        self, "Выберите Excel-файлы для обучения CatBoost", "", "Excel (*.xlsx *.xls)")
    if not paths:
        return
    directory = QFileDialog.getExistingDirectory(
        self, "Папка для сохранения новой модели CatBoost")
    if not directory:
        return
    try:
        from catboost_training import train_and_save
        self._cb_train.setEnabled(False)
        self._cb_status.setText("CatBoost обучается. Это может занять некоторое время…")
        # Обучение выполняется через тот же официальный модуль, что и отдельная панель CatBoost.
        result = train_and_save(paths, directory, allow_unvalidated=True, iterations=300)
        model_path = result.get("model_path", "") if isinstance(result, dict) else ""
        self._cb_status.setText("Модель CatBoost обучена и сохранена: " + str(model_path))
        QMessageBox.information(self, "CatBoost обучен",
            "Новая модель сохранена.\n\n" + str(model_path) +
            "\n\nРядом сохранены catboost_model.json и training_report.json.")
    except Exception as exc:
        self._cb_status.setText("Обучение CatBoost не выполнено.")
        QMessageBox.warning(self, "Ошибка обучения CatBoost", str(exc))
    finally:
        self._cb_train.setEnabled(True)


def _fixed_generate_qwen(self):
    if getattr(self, "result", None) is None:
        QMessageBox.information(self, "Qwen", "Сначала дождитесь завершения общего поиска.")
        return
    index = self.table.currentIndex()
    if not index.isValid():
        QMessageBox.information(self, "Qwen", "Выберите кандидата в таблице.")
        return
    try:
        import pandas as pd
        from llm_advisor import generate_conclusion
        row = self.result["features"].iloc[index.row()]
        states = self.result.get("states", pd.DataFrame()).to_dict("records")
        evidence = {}
        for group, payload in self.result.get("raw", {}).items():
            details = payload.get("details") if isinstance(payload, dict) else None
            if isinstance(details, dict):
                candidate = str(row.get("candidate_inn", ""))
                evidence[group] = details.get(candidate, pd.DataFrame())
            elif isinstance(details, pd.DataFrame):
                evidence[group] = details
        text = generate_conclusion(
            str(row.get("source_inn", getattr(self, "source_inn", ""))),
            str(row.get("candidate_inn", "")), row.to_dict(),
            self.result.get("warnings", []), states, evidence)
        dialog = QDialog(self)
        dialog.setWindowTitle("Заключение Qwen")
        dialog.resize(900, 650)
        box = QVBoxLayout(dialog)
        editor = QPlainTextEdit(); editor.setReadOnly(True); editor.setPlainText(text)
        box.addWidget(editor)
        close = QPushButton("Закрыть"); close.clicked.connect(dialog.accept); box.addWidget(close)
        dialog.exec()
    except Exception as exc:
        QMessageBox.warning(self, "Qwen", str(exc))


try:
    CandidateSearchDialog._train_model = _fixed_train_model
    CandidateSearchDialog._generate_qwen = _fixed_generate_qwen
except NameError:
    pass
