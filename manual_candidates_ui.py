"""Надстройка для существующего PySide6-окна общего поиска и CatBoost."""
import re
import pandas as pd
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QLabel, QInputDialog, QMessageBox
from core import valid_inn
from catboost_results_ui import CandidateSearchDialog as BaseDialog
from manual_candidate_logic import augment_result, present_organizations


class CandidateSearchDialog(BaseDialog):
    def __init__(self, *args, **kwargs):
        self._manual_requested = set()
        self._manual_base = None
        self._manual_panel = None
        super().__init__(*args, **kwargs)
        QTimer.singleShot(0, self._manual_mount)

    def _manual_mount(self):
        if self._manual_panel is not None:
            return
        parent = self.table.parentWidget()
        layout = parent.layout() if parent is not None else None
        if layout is None or not hasattr(layout, "insertWidget"):
            parent, layout = self, self.layout()
        panel = QWidget(parent)
        bar = QHBoxLayout(panel)
        bar.setContentsMargins(0, 4, 0, 4)
        add = QPushButton("Добавить организации для проверки…")
        remove = QPushButton("Убрать ручные организации…")
        self._manual_label = QLabel()
        self._manual_label.setWordWrap(True)
        add.clicked.connect(self._manual_add)
        remove.clicked.connect(self._manual_remove)
        bar.addWidget(add)
        bar.addWidget(remove)
        bar.addWidget(self._manual_label, 1)
        index = layout.indexOf(self.table)
        layout.insertWidget(max(0, index), panel)
        self._manual_panel = panel
        self._manual_update_label()

    def _manual_update_label(self):
        if hasattr(self, "_manual_label"):
            self._manual_label.setText("Ручное сравнение: " +
                                       (", ".join(sorted(self._manual_requested)) or "организации не добавлены") +
                                       ". Баллы рассчитывает выбранная модель, не задаются вручную.")

    def _manual_available(self):
        if self.running():
            QMessageBox.information(self, "Подождите", "Дождитесь завершения поиска и расчёта CatBoost.")
            return False
        if self._manual_base is None:
            QMessageBox.information(self, "Сначала поиск", "Выполните общий поиск по загруженной книге.")
            return False
        return True

    def _manual_add(self):
        if not self._manual_available():
            return
        states = self._manual_base["states"]
        if not states["Статус"].eq("completed").any():
            QMessageBox.warning(self, "Нет расчёта",
                                "Ни одно направление не проверено. Сначала исправьте загрузку данных.")
            return
        text, ok = QInputDialog.getMultiLineText(self, "Добавить в сравнение",
                                                 "Введите ИНН организаций через пробел, запятую или с новой строки.\n"
                                                 "Нужны записи этих организаций в загруженной книге.\n"
                                                 "Каждый ИНН добавится в общий список и будет оценён выбранной моделью:")
        if not ok or not text.strip():
            return
        inns = set(re.split(r"[\s,;]+", text.strip()))
        bad = sorted(x for x in inns if not valid_inn(x, (10,)))
        if bad:
            QMessageBox.warning(self, "Некорректный ИНН", "Проверьте формат и контрольную сумму: " + ", ".join(bad))
            return
        if self.source_inn in inns:
            QMessageBox.warning(self, "Исходная организация", "Исходную организацию нельзя добавить как кандидата.")
            return
        absent = inns - present_organizations(self.frames)
        if absent:
            QMessageBox.warning(self, "Нет исходных записей",
                                "Эти организации не найдены в столбце «ИНН организации» загруженной книги: " +
                                ", ".join(sorted(
                                    absent)) + ".\nДобавьте их реальные исходные данные и загрузите книгу заново. "
                                               "Одного упоминания в качестве контрагента недостаточно. Пустую строку с нулевыми признаками не создаём.")
            return
        self._manual_requested.update(inns)
        self._manual_refresh()

    def _manual_remove(self):
        if not self._manual_available() or not self._manual_requested:
            return
        selected, ok = QInputDialog.getItem(self, "Убрать из ручного сравнения", "Организация:",
                                            ["Все ручные организации"] + sorted(self._manual_requested), 0, False)
        if not ok:
            return
        if selected == "Все ручные организации":
            self._manual_requested.clear()
        else:
            self._manual_requested.discard(selected)
        self._manual_refresh()

    def _manual_refresh(self):
        updated = augment_result(
            self._manual_base,
            self._manual_requested,
            model_path=getattr(self, "_cb_path", None),
        )

        super().show_result(updated)
        self._manual_update_label()
        cb_info = updated["metadata"].get("manual_catboost", {})
        if cb_info.get("warning"):
            self.status.setText(self.status.text() + " CatBoost: " + cb_info["warning"])
        elif not cb_info.get("applied"):
            self.status.setText(
                self.status.text() + " " + cb_info.get("mode", "CatBoost не применён.")
            )

    def show_result(self, result):
        self._manual_base = result
        # При повторном поиске данные заново рассчитаны штатными модулями.
        self._manual_requested.intersection_update(present_organizations(self.frames))
        self._manual_refresh()
