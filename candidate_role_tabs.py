"""Учредители и руководители как вкладки оснований в окне «Общий поиск кандидатов».
Отдельные кнопки «Поиск по учредителям/руководителям» скрываются.
«Обучить CatBoost…» переносится в ряд «Повторить поиск / Сохранить в Excel / Qwen», «Закрыть» убирается.
Расчёты сотрудников, поставщиков, покупателей, ККТ и CatBoost не меняются."""
import pandas as pd
from PySide6.QtCore import Qt, QTimer, QAbstractTableModel, QModelIndex
from PySide6.QtWidgets import (QPushButton, QLabel, QTabWidget, QTableView, QAbstractItemView,
                               QHBoxLayout, QWidget)
try:
    from roles import search_roles
except Exception:  # старая версия проекта
    from org_roles import search_roles

ROLE_TABS = (("founders", "Учредители"), ("directors", "Руководители"))
HIDE_TEXT = ("Поиск по учредителям", "Поиск по руководителям", "Связи по лицам")
HIDE_COLS = ("ИНН кандидата", "Наименование кандидата")


class _Model(QAbstractTableModel):
    def __init__(self, frame, parent=None):
        super().__init__(parent)
        self.f = frame.reset_index(drop=True)
    def rowCount(self, p=QModelIndex()):
        return len(self.f)
    def columnCount(self, p=QModelIndex()):
        return len(self.f.columns)
    def data(self, i, role=Qt.ItemDataRole.DisplayRole):
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            v = self.f.iat[i.row(), i.column()]
            return "" if pd.isna(v) else str(v)
    def headerData(self, s, o, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            return str(self.f.columns[s]) if o == Qt.Orientation.Horizontal else str(s + 1)


def _find(layout, widget):
    if layout is None:
        return None
    for i in range(layout.count()):
        it = layout.itemAt(i)
        if it.widget() is widget:
            return layout, i
        if it.layout() is not None:
            hit = _find(it.layout(), widget)
            if hit:
                return hit
    return None


def _owner(widget):
    w = widget.parentWidget()
    while w is not None:
        hit = _find(w.layout(), widget)
        if hit:
            return hit
        w = w.parentWidget()
    return None


def _role_details(dialog):
    """Ролевые основания считаются один раз на исходный ИНН и набор данных."""
    key = (id(getattr(dialog, "frames", None)), str(getattr(dialog, "source_inn", "")))
    cache = getattr(dialog, "_role_cache", None)
    if cache and cache[0] == key:
        return cache[1]
    out = {}
    for mode, title in ROLE_TABS:
        try:
            out[title] = search_roles(dialog.frames or {}, dialog.source_inn, mode)["details"]
        except Exception as exc:
            out[title] = str(exc)
    dialog._role_cache = (key, out)
    return out


def _details_tabs(dialog):
    tabs = getattr(dialog, "details", None)
    if isinstance(tabs, QTabWidget):
        return tabs
    for t in dialog.findChildren(QTabWidget):
        if any(t.tabText(i).startswith(("Сотрудники", "Поставщики", "Покупатели", "ККТ")) for i in range(t.count())):
            return t
    return None


def _current_inn(dialog, current):
    try:
        return str(dialog.result["features"].iloc[current.row()]["candidate_inn"]).strip()
    except Exception:
        return ""


def add_role_tabs(dialog, inn):
    _remove_role_search_tabs(dialog)
    tabs = _details_tabs(dialog)
    if tabs is None or not inn:
        return
    for i in range(tabs.count() - 1, -1, -1):  # убираем прежние ролевые вкладки, без задвоения
        if tabs.tabText(i).startswith(("Учредител", "Руководител")):
            w = tabs.widget(i); tabs.removeTab(i); w.deleteLater()
    for title, data in _role_details(dialog).items():
        if isinstance(data, str):
            lab = QLabel(data); lab.setWordWrap(True)
            tabs.addTab(lab, title)
            continue
        frame = data[data["ИНН кандидата"].astype(str).str.strip() == inn]
        frame = frame.drop(columns=[c for c in HIDE_COLS if c in frame.columns]).drop_duplicates()
        if frame.empty:
            lab = QLabel(f"Совпадений по {title.lower()[:-1]}ям не найдено в загруженных данных.")
            tabs.addTab(lab, f"{title} (0)")
            continue
        view = QTableView(tabs)
        view.setModel(_Model(frame, view))
        view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        view.setAlternatingRowColors(True)
        view.resizeColumnsToContents()
        view.horizontalHeader().setStretchLastSection(True)
        tabs.addTab(view, f"{title} ({len(frame)})")



def _remove_role_search_tabs(dialog):
    """Удаляет дублирующие вкладки поиска ролей из окна кандидатов.

    Кнопки «Поиск по учредителям» и «Поиск по руководителям»
    остаются в главном окне; здесь остаются только вкладки оснований
    «Учредители» и «Руководители».
    """
    prefixes = ("поиск по учредителям", "поиск по руководителям")
    for tabs in dialog.findChildren(QTabWidget):
        for index in range(tabs.count() - 1, -1, -1):
            title = tabs.tabText(index).strip().lower()
            if title.startswith(prefixes):
                widget = tabs.widget(index)
                tabs.removeTab(index)
                if widget is not None:
                    widget.deleteLater()


def arrange(dialog):
    _remove_role_search_tabs(dialog)
    # 1. Отдельные кнопки «Поиск по учредителям/руководителям» не нужны.
    bar = getattr(dialog, "_role_bar", None)
    if bar is not None:
        bar.hide()
    for w in dialog.findChildren(QPushButton) + dialog.findChildren(QLabel):
        if w.text().strip().startswith(HIDE_TEXT):
            w.hide()
    # 2. «Закрыть» убираем (окно закрывается крестиком).
    for b in dialog.findChildren(QPushButton):
        if b.text().strip() == "Закрыть":
            b.hide()
    # 3. «Обучить CatBoost…» — в ряд с «Повторить поиск».
    retry = getattr(dialog, "retry", None)
    train = next((b for b in dialog.findChildren(QPushButton) if b.text().strip().startswith("Обучить CatBoost")), None)
    if retry is None or train is None or getattr(dialog, "_train_moved", False):
        return
    row = _owner(retry)
    old = _owner(train)
    if not row:
        return
    old_parent = train.parentWidget()
    if old:
        old[0].removeWidget(train)
    layout = row[0]
    ai = getattr(dialog, "ai_button", None)
    pos = _find(layout, ai)
    idx = pos[1] + 1 if pos and pos[0] is layout else row[1] + 1
    train.setParent(retry.parentWidget())
    train.setSizePolicy(retry.sizePolicy())
    layout.insertWidget(idx, train)
    train.show()
    # Если от старого места осталась пустая полоса — прячем её.
    if old_parent is not None and old_parent is not dialog and old_parent is not retry.parentWidget():
        if not any(c.isVisibleTo(old_parent) for c in old_parent.findChildren(QWidget)):
            old_parent.hide()
    dialog._train_moved = True


def install(cls):
    if getattr(cls, "_role_tabs_installed", False):
        return cls
    base_show = cls.showEvent
    base_select = getattr(cls, "select_candidate", None)

    def showEvent(self, event):
        base_show(self, event)
        for delay in (0, 300, 1200):
            QTimer.singleShot(delay, lambda d=self: arrange(d))
    cls.showEvent = showEvent

    if base_select is not None:
        def select_candidate(self, current, previous=QModelIndex()):
            out = base_select(self, current, previous)
            if current is not None and current.isValid():
                add_role_tabs(self, _current_inn(self, current))
            return out
        cls.select_candidate = select_candidate
    cls._role_tabs_installed = True
    return cls
