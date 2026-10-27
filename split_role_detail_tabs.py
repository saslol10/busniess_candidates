"""Разделение ролевых оснований в существующем окне. Расчеты не меняются."""
import re
from functools import wraps
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import QTableView, QLabel, QAbstractItemView


def _has_role(value, stem):
    return any(word.startswith(stem) for word in re.findall(r"[а-яё]+", str(value).lower()))


def partition_rows(headers, rows):
    """Связь попадает в раздел, если роль есть хотя бы с одной стороны."""
    role_columns = [i for i, name in enumerate(headers)
                    if ("исходной" in name.lower() or "кандидате" in name.lower())
                    and ("организаци" in name.lower() or "роль" in name.lower())]
    if len(role_columns) < 2:
        return None
    return {title: [row for row in rows if any(_has_role(row[i], stem) for i in role_columns)]
            for title, stem in (("Учредители", "учредител"), ("Руководители", "руководител"))}


def split_detail_tabs(dialog):
    tabs = getattr(dialog, "details", None)
    if tabs is None or getattr(dialog, "_role_tabs_busy", False):
        return
    dialog._role_tabs_busy = True
    try:
        for index in range(tabs.count() - 1, -1, -1):
            title = tabs.tabText(index).lower()
            if "учредител" not in title or "руководител" not in title:
                continue
            old = tabs.widget(index)
            view = old if isinstance(old, QTableView) else old.findChild(QTableView)
            if view is None or view.model() is None:
                # Не подменяем сообщения об ошибках/отсутствии исходных данных.
                continue
            model = view.model()
            headers = [str(model.headerData(c, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole) or "")
                       for c in range(model.columnCount())]
            rows = [[str(model.data(model.index(r, c), Qt.ItemDataRole.DisplayRole) or "")
                     for c in range(model.columnCount())] for r in range(model.rowCount())]
            groups = partition_rows(headers, rows)
            if groups is None:
                continue
            selected = tabs.currentWidget() is old
            tabs.removeTab(index)
            for offset, (name, values) in enumerate(groups.items()):
                new_view = QTableView(tabs)
                new_model = QStandardItemModel(new_view)
                new_model.setHorizontalHeaderLabels(headers)
                for row in values:
                    new_model.appendRow([QStandardItem(value) for value in row])
                new_view.setModel(new_model)
                new_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
                new_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
                new_view.setAlternatingRowColors(True)
                new_view.setSortingEnabled(True)
                for col in range(len(headers)):
                    new_view.setColumnWidth(col, max(160, min(view.columnWidth(col), 420)))
                new_view.setToolTip("Совпадение учитывается, если выбранная роль есть у человека "
                                    "в исходной организации или у кандидата. Пустая таблица означает, "
                                    "что в существующей расшифровке таких строк нет.")
                tabs.insertTab(index + offset, new_view, f"{name} ({len(values)})")
            if selected:
                tabs.setCurrentIndex(index)
            old.deleteLater()
    finally:
        dialog._role_tabs_busy = False


def attach_split_role_tabs(dialog_class):
    if dialog_class.__dict__.get("_split_role_tabs_installed", False):
        return dialog_class
    original = dialog_class.select_candidate
    @wraps(original)
    def select_candidate(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        split_detail_tabs(self)
        # Поддержка надстроек, дополняющих вкладки после базового обработчика.
        def later():
            from shiboken6 import isValid
            if isValid(self):
                split_detail_tabs(self)
        QTimer.singleShot(0, later)
        return result
    dialog_class.select_candidate = select_candidate
    dialog_class._split_role_tabs_installed = True
    return dialog_class
