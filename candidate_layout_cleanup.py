"""Компоновка окна «Общий поиск кандидатов»: без таблицы статусов и лишних кнопок,
ручные организации — под таблицей CatBoost. Только интерфейс, логика расчёта не меняется."""
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QLabel

HIDE_BUTTONS = ("Загрузить модель CatBoost", "Модель по умолчанию", "Выгрузить для разметки")
HIDE_LABELS = ("CatBoost не применён: модель не загружена",)
MANUAL_BUTTONS = ("Добавить организации для проверки", "Убрать ручные организации")


def _find(layout, widget):
    """Возвращает (layout, index), где лежит widget, с обходом вложенных layout."""
    if layout is None:
        return None
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item.widget() is widget:
            return layout, i
        if item.layout() is not None:
            hit = _find(item.layout(), widget)
            if hit:
                return hit
    return None


def _owner(widget, root):
    w = widget.parentWidget()
    while w is not None:
        hit = _find(w.layout(), widget)
        if hit:
            return hit
        if w is root:
            break
        w = w.parentWidget()
    return None


def _detach(widget, root):
    hit = _owner(widget, root)
    if hit:
        hit[0].removeWidget(widget)


def cleanup(dialog):
    # 1. Первая таблица (статусы направлений) не нужна. Модель у неё остаётся,
    #    поэтому код поиска, который вызывает self.states.setModel, работает как раньше.
    states = getattr(dialog, "states", None)
    if states is not None:
        states.hide()
    # 2. Убираем кнопки загрузки модели, модели по умолчанию и выгрузки для разметки.
    for button in dialog.findChildren(QPushButton):
        if button.text().strip().startswith(HIDE_BUTTONS):
            button.hide()
            button.setEnabled(False)
    for label in dialog.findChildren(QLabel):
        if label is getattr(dialog, "status", None) or label is getattr(dialog, "_cb_status", None):
            continue
        if label.text().strip().startswith(HIDE_LABELS):
            label.hide()
    # 3. Ручные организации — под расширенной таблицей CatBoost.
    table = getattr(dialog, "_cb_table", None)
    if table is None or getattr(dialog, "_manual_row_moved", False):
        return
    buttons = [b for b in dialog.findChildren(QPushButton) if b.text().strip().startswith(MANUAL_BUTTONS)]
    if not buttons:
        return
    target = _owner(table, dialog)
    if not target:
        return
    old_parent = buttons[0].parentWidget()
    labels = [l for l in dialog.findChildren(QLabel) if l.text().strip().startswith("Ручное сравнение")]
    row = QWidget(table.parentWidget())
    box = QHBoxLayout(row)
    box.setContentsMargins(0, 4, 0, 0)
    for w in buttons + labels:
        _detach(w, dialog)
        box.addWidget(w, 1 if isinstance(w, QLabel) else 0)
        if isinstance(w, QLabel):
            w.setWordWrap(True)
    layout, index = target
    layout.insertWidget(index + 1, row)
    # Пустую старую строку скрываем, чтобы не оставалось отступа сверху.
    if old_parent is not None and old_parent is not dialog and old_parent is not row:
        if not any(c.isVisibleTo(old_parent) for c in old_parent.findChildren(QWidget)):
            old_parent.hide()
    dialog._manual_row_moved = True


def install(cls):
    if getattr(cls, "_layout_cleanup_installed", False):
        return cls
    base_show = cls.showEvent
    base_result = getattr(cls, "show_result", None)

    def showEvent(self, event):
        base_show(self, event)
        # Часть панелей (Qwen, обучение) добавляется после показа окна.
        for delay in (0, 250, 1000):
            QTimer.singleShot(delay, lambda d=self: cleanup(d))

    cls.showEvent = showEvent
    if base_result is not None:
        def show_result(self, *args, **kwargs):
            out = base_result(self, *args, **kwargs)
            cleanup(self)
            return out
        cls.show_result = show_result
    cls._layout_cleanup_installed = True
    return cls
