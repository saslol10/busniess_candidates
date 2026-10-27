"""Окно результатов и кнопки «Учредители / Руководители» для обоих окон."""
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QWidget,
                               QTableWidget, QTableWidgetItem, QSplitter, QFileDialog, QMessageBox,
                               QHeaderView, QAbstractItemView)
from roles import search_roles, TITLES


def _fill(table, frame):
    table.clear()
    table.setRowCount(len(frame))
    table.setColumnCount(len(frame.columns))
    table.setHorizontalHeaderLabels([str(c) for c in frame.columns])
    for r, row in enumerate(frame.itertuples(index=False)):
        for c, v in enumerate(row):
            table.setItem(r, c, QTableWidgetItem("" if pd.isna(v) else str(v)))
    table.resizeColumnsToContents()
    table.horizontalHeader().setStretchLastSection(True)


def _table():
    t = QTableWidget()
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    t.setAlternatingRowColors(True)
    t.verticalHeader().setVisible(False)
    t.setWordWrap(True)
    return t


class RoleDialog(QDialog):
    def __init__(self, result, source, parent=None):
        super().__init__(parent)
        self.result, self.source = result, source
        self.setWindowTitle(f"{TITLES[result['mode']]} — {source}")
        self.resize(1100, 680)
        lay = QVBoxLayout(self)
        info = QLabel(result.get("message", ""))
        info.setWordWrap(True)
        info.setStyleSheet("background:#eaf1f9;color:#173556;padding:10px;border-radius:6px")
        lay.addWidget(info)
        split = QSplitter(Qt.Orientation.Vertical)
        self.rank, self.det = _table(), _table()
        _fill(self.rank, result["ranking"])
        _fill(self.det, result["details"])
        top, bottom = QWidget(), QWidget()
        tl, bl = QVBoxLayout(top), QVBoxLayout(bottom)
        tl.addWidget(QLabel("Кандидаты"))
        tl.addWidget(self.rank)
        self.det_label = QLabel("Основания: все кандидаты")
        bl.addWidget(self.det_label)
        bl.addWidget(self.det)
        split.addWidget(top)
        split.addWidget(bottom)
        lay.addWidget(split, 1)
        row = QHBoxLayout()
        export = QPushButton("Сохранить в Excel…")
        export.setEnabled(not result["ranking"].empty)
        export.clicked.connect(self.export)
        close = QPushButton("Закрыть")
        close.clicked.connect(self.accept)
        row.addStretch()
        row.addWidget(export)
        row.addWidget(close)
        lay.addLayout(row)
        self.rank.itemSelectionChanged.connect(self.select)

    def select(self):
        rows = self.rank.selectionModel().selectedRows()
        frame = self.result["details"]
        if rows:
            inn = self.rank.item(rows[0].row(), 1).text()
            frame = frame[frame["ИНН кандидата"] == inn]
            self.det_label.setText(f"Основания по кандидату {inn}")
        _fill(self.det, frame)

    def export(self):
        name = f"{self.result['mode']}_{self.source}.xlsx"
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить", name, "Excel (*.xlsx)")
        if not path:
            return
        try:
            with pd.ExcelWriter(path, engine="openpyxl") as w:
                self.result["ranking"].to_excel(w, sheet_name="Кандидаты", index=False)
                self.result["details"].to_excel(w, sheet_name="Основания", index=False)
            QMessageBox.information(self, "Сохранено", path)
        except Exception as exc:
            QMessageBox.warning(self, "Ошибка сохранения", str(exc))


def open_role_search(parent, frames, source_inn, mode):
    try:
        result = search_roles(frames or {}, source_inn, mode)
    except Exception as exc:
        QMessageBox.warning(parent, TITLES.get(mode, "Поиск"), str(exc))
        return
    RoleDialog(result, str(source_inn).strip(), parent).exec()


def mount_role_buttons(dialog, frames, source_inn):
    """Добавляет две кнопки в верх окна CatBoost/Qwen. Сам расчёт CatBoost не трогает."""
    bar = QWidget(dialog)
    row = QHBoxLayout(bar)
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(QLabel("Связи по лицам:"))
    for mode, text in (("founders", "Поиск по учредителям…"), ("directors", "Поиск по руководителям…")):
        b = QPushButton(text)
        b.setToolTip("Учредители/руководители исходной организации среди учредителей, руководителей и сотрудников других организаций")
        b.clicked.connect(lambda _=False, m=mode: open_role_search(dialog, frames, source_inn, m))
        row.addWidget(b)
    row.addStretch()
    root = dialog.layout()
    if hasattr(root, "insertWidget"):
        root.insertWidget(0, bar)
    elif root is not None:
        root.addWidget(bar)
    dialog._role_bar = bar
    return bar
