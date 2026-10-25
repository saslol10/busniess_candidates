"""Добавляет только колонку «Действие» в существующую таблицу CatBoost."""
import math
from PySide6.QtCore import Qt, QModelIndex, QTimer
from PySide6.QtWidgets import QTableView, QHeaderView


def action_for_score(value):
    try:
        score = float(str(value).replace(',', '.').strip())
    except (TypeError, ValueError):
        return '—'
    if not math.isfinite(score) or not 0 <= score <= 1:
        return '—'
    if score >= 0.70:
        return 'Проверить в первую очередь'
    if score >= 0.30:
        return 'Проверить после кандидата с более высокой оценкой'
    return 'Можно исключить из проверки'


def _headers(model):
    return [str(model.headerData(i, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole) or '')
            for i in range(model.columnCount())]


def add_action_column(table):
    model = table.model()
    if model is None or getattr(model, '_action_only_installed', False):
        return False
    headers = _headers(model)
    # Только компактная таблица результатов, не таблица признаков и не основания.
    if not 3 <= len(headers) <= 4:
        return False
    if not any('ИНН кандидата' == h.strip() for h in headers):
        return False
    score_cols = [i for i, h in enumerate(headers) if 'оценка' in h.lower() and 'catboost' in h.lower()]
    if len(score_cols) != 1 or not any('место' in h.lower() for h in headers):
        return False
    score_col = score_cols[0]
    existing = next((i for i, h in enumerate(headers) if h.strip() == 'Действие'), None)
    action_col = len(headers) if existing is None else existing
    added = existing is None
    base = type(model)

    def score_at(self, row):
        frame = getattr(self, 'frame', None)
        if frame is not None:
            try:
                return frame.iat[row, score_col]
            except (AttributeError, IndexError):
                pass
        return base.data(self, self.index(row, score_col), Qt.ItemDataRole.DisplayRole)

    class WithAction(base):
        def columnCount(self, parent=QModelIndex()):
            count = base.columnCount(self, parent)
            return count + (1 if added and not parent.isValid() else 0)

        def index(self, row, column, parent=QModelIndex()):
            if column == action_col and added:
                if not parent.isValid() and 0 <= row < self.rowCount() :
                    return self.createIndex(row, column)
                return QModelIndex()
            return base.index(self, row, column, parent)

        def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
            if orientation == Qt.Orientation.Horizontal and section == action_col:
                if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
                    return 'Действие'
                return None
            return base.headerData(self, section, orientation, role)

        def data(self, index, role=Qt.ItemDataRole.DisplayRole):
            if index.isValid() and index.column() == action_col:
                if role == Qt.ItemDataRole.DisplayRole:
                    return action_for_score(score_at(self, index.row()))
                if role == Qt.ItemDataRole.ToolTipRole:
                    return (action_for_score(score_at(self, index.row())) +
                            '\nРекомендация по порогам 0.30 / 0.70. Не подтверждает переход бизнеса. '
                            'Исключение не автоматическое: проверьте полноту данных.')
                if role == Qt.ItemDataRole.TextAlignmentRole:
                    return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                return None
            return base.data(self, index, role)

        def flags(self, index):
            if index.isValid() and index.column() == action_col:
                return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            return base.flags(self, index)

    # Сохраняем тот же объект модели и selectionModel: обработчики выбора не теряются.
    if added:
        model.beginInsertColumns(QModelIndex(), action_col, action_col)
    model.__class__ = WithAction
    model._action_only_installed = True
    if added:
        model.endInsertColumns()
    else:
        model.headerDataChanged.emit(Qt.Orientation.Horizontal, action_col, action_col)
    if model.rowCount():
        model.dataChanged.emit(model.index(0, action_col), model.index(model.rowCount()-1, action_col),
                               [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole])

    def refresh_actions(*args):
        if model.rowCount():
            model.dataChanged.emit(model.index(0, action_col), model.index(model.rowCount()-1, action_col),
                                   [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole])
    def on_data_changed(first, last, roles=None):
        if first.column() <= score_col <= last.column():
            refresh_actions()
    model.dataChanged.connect(on_data_changed)
    model.modelReset.connect(refresh_actions)
    header = table.horizontalHeader()
    # Перестаёт растягиваться колонка оценки, свободное место занимает текст действия.
    header.setStretchLastSection(False)
    for i, h in enumerate(headers):
        if i != action_col:
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.Interactive)
            table.setColumnWidth(i, 65 if 'место' in h.lower() else 155 if i == score_col else 145)
    header.setSectionResizeMode(action_col, QHeaderView.ResizeMode.Stretch)
    header.setMinimumSectionSize(55)
    table.setWordWrap(True)
    table.resizeRowsToContents()
    return True


def attach_action_column(dialog_class):
    if getattr(dialog_class, '_action_column_wrapper', False):
        return dialog_class
    class DialogWithAction(dialog_class):
        _action_column_wrapper = True
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            def scan():
                for table in self.findChildren(QTableView):
                    add_action_column(table)
            self._action_column_timer = QTimer(self)
            self._action_column_timer.setInterval(400)
            self._action_column_timer.timeout.connect(scan)
            self._action_column_timer.start()
            scan()
    DialogWithAction.__name__ = dialog_class.__name__
    return DialogWithAction
