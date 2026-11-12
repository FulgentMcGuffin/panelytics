from __future__ import annotations

import re
import sys
import warnings
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import polars as pl
from matplotlib.ticker import AutoMinorLocator
from plotnine.exceptions import PlotnineError, PlotnineWarning
from PySide6.QtCore import (
    QModelIndex,
    QPoint,
    QSize,
    QStringListModel,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QBrush,
    QCloseEvent,
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QImage,
    QKeySequence,
    QShowEvent,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QCompleter,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from gui.data_table_dialog import DataTableDialog, eye_icon
from gui.dialogs import EntityConfigurationDialog, UserFunctionsDialog
from gui.gui_constants import Theme, get_plot_bg, get_theme, window_style
from gui.gui_settings import load_settings, save_settings
from gui.gui_utils import (
    DateRangeSlider,
    FramelessResizeHelper,
    PanelFrame,
    RoundedShell,
    TitleBar,
)
from gui.panelyzer_bridge import (
    DataSourceRecord,
    build_autocomplete_tokens,
    create_source_from_clipboard,
    create_source_from_path,
    evaluate_formula_rows,
    formula_uses_source,
    load_user_eval_globals,
    parse_clipboard_dataframe,
    populate_source_metadata,
)
from gui.plotnine_wrapper import (
    coerce_date_columns,
    create_ggplot_from_df,
    format_display_date,
    infer_plot_columns,
    process_df_for_ggplot,
)
from gui.settings_dialog import SettingsDialog
from gui.table_and_plot_widget import PlotWidget

MAX_FORMULA_ROWS = 10

_TOKEN_PREFIX_RE = re.compile(r"(\{[A-Za-z0-9_.:-]*|[A-Za-z_][A-Za-z0-9_.]*)$")


class FormulaLineEdit(QLineEdit):
    """QLineEdit with token-aware QCompleter for panelyzer expressions."""

    def __init__(self, tokens: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tokens = list(tokens)
        self._model = QStringListModel(self._tokens, self)
        self._completer = QCompleter(self._model, self)
        self._completer.setWidget(self)
        self._completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self._completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self._completer.activated.connect(self._insert_completion)
        self.textEdited.connect(self._on_text_edited)

    def _current_token_span(self) -> tuple[int, int, str]:
        cursor = self.cursorPosition()
        left_text = self.text()[:cursor]
        match = _TOKEN_PREFIX_RE.search(left_text)
        if not match:
            return cursor, cursor, ""
        return match.start(1), cursor, match.group(1)

    def _on_text_edited(self, _text: str) -> None:
        _start, _end, prefix = self._current_token_span()
        if not prefix:
            self._completer.popup().hide()
            return
        self._completer.setCompletionPrefix(prefix)
        popup = self._completer.popup()
        popup.setCurrentIndex(self._completer.completionModel().index(0, 0))
        cr = self.cursorRect()
        cr.setWidth(
            max(
                220,
                popup.sizeHintForColumn(0)
                + popup.verticalScrollBar().sizeHint().width()
                + 20,
            )
        )
        self._completer.complete(cr)

    def _insert_completion(self, completion: str) -> None:
        start, end, _prefix = self._current_token_span()
        full = self.text()
        # Avoid doubling closing brace if one already sits right at `end`
        if completion.endswith("}") and end < len(full) and full[end] == "}":
            end += 1
        new_text = full[:start] + completion + full[end:]
        self.setText(new_text)
        self.setCursorPosition(start + len(completion))


class FormulaDelegate(QStyledItemDelegate):
    """Table item delegate providing FormulaLineEdit with live autocomplete tokens."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tokens: list[str] = []

    def set_tokens(self, tokens: list[str]) -> None:
        """Update the list of autocomplete suggestions."""
        self._tokens = list(tokens)

    def createEditor(
        self, parent: QWidget, _option, _index: QModelIndex
    ) -> QWidget:
        editor = FormulaLineEdit(self._tokens, parent)
        editor.setPlaceholderText(
            "e.g. {USA:Y010p0} - {DEU:Y010p0}  or  romean({USA:Y010p0}, 20)"
        )
        return editor

    def setEditorData(self, editor: QWidget, index: QModelIndex) -> None:
        if isinstance(editor, QLineEdit):
            editor.setText(str(index.data(Qt.ItemDataRole.EditRole) or ""))

    def setModelData(
        self, editor: QWidget, model, index: QModelIndex
    ) -> None:
        if isinstance(editor, QLineEdit):
            model.setData(
                index, editor.text().strip(), Qt.ItemDataRole.EditRole
            )


class DataSourcesListWidget(QListWidget):
    """Draggable, drop-enabled list widget for Panelytics Data Sources."""

    files_dropped = Signal(list)
    modify_requested = Signal(int)
    delete_requested = Signal(int)
    paste_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)
        self.itemDoubleClicked.connect(self._on_double_clicked)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        if event.mimeData().hasUrls():
            paths = [
                url.toLocalFile()
                for url in event.mimeData().urls()
                if url.isLocalFile()
            ]
            if paths:
                self.files_dropped.emit(paths)
                event.acceptProposedAction()
                return
        super().dropEvent(event)

    def _on_double_clicked(self, item: QListWidgetItem) -> None:
        row = self.row(item)
        if row >= 0:
            self.modify_requested.emit(row)

    def _on_context_menu(self, pos: QPoint) -> None:
        item = self.itemAt(pos)
        menu = QMenu(self)

        if item is not None:
            row = self.row(item)
            modify_action = QAction("Modify", self)
            delete_action = QAction("Delete", self)
            modify_action.triggered.connect(lambda: self.modify_requested.emit(row))
            delete_action.triggered.connect(lambda: self.delete_requested.emit(row))
            menu.addAction(modify_action)
            menu.addAction(delete_action)
        else:
            clip_text = QApplication.clipboard().text()
            is_valid = parse_clipboard_dataframe(clip_text) is not None
            paste_action = QAction("Paste", self)
            paste_action.setEnabled(is_valid)
            paste_action.triggered.connect(self.paste_requested.emit)
            menu.addAction(paste_action)

        menu.exec(self.viewport().mapToGlobal(pos))


class CollapsibleDataSourcesPane(QWidget):
    """Left collapsible Data Sources section (takes ~22% of width when expanded)."""

    toggled = Signal(bool)
    add_file_requested = Signal()

    def __init__(
        self,
        list_widget: DataSourcesListWidget,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._list_widget = list_widget
        self._expanded = True

        self.setObjectName("panelFrame")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(10, 10, 10, 10)
        self._layout.setSpacing(8)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(6)

        self._toggle_btn = QPushButton("◀  DATA SOURCES")
        self._toggle_btn.setObjectName("collapsibleToggle")
        self._toggle_btn.setToolTip("Collapse / Expand Data Sources")
        self._toggle_btn.clicked.connect(self.toggle)
        header_row.addWidget(self._toggle_btn, stretch=1)

        self._add_btn = QPushButton("+")
        self._add_btn.setObjectName("smallBtn")
        self._add_btn.setFixedWidth(28)
        self._add_btn.setToolTip("Add Data Source from File…")
        self._add_btn.clicked.connect(self.add_file_requested.emit)
        header_row.addWidget(self._add_btn)

        self._layout.addLayout(header_row)

        self._hint_label = QLabel(
            "Drag files here or right-click empty space to Paste clipboard data."
        )
        self._hint_label.setObjectName("statusLabel")
        self._hint_label.setWordWrap(True)
        self._layout.addWidget(self._hint_label)

        self._layout.addWidget(self._list_widget, stretch=1)
        self._layout.addStretch(0)

    @property
    def is_expanded(self) -> bool:
        """Return True if the Data Sources panel is currently expanded."""
        return self._expanded

    def toggle(self) -> None:
        """Toggle collapsed/expanded state."""
        self.set_expanded(not self._expanded)

    def set_expanded(self, expanded: bool) -> None:
        """Set expanded or collapsed state explicitly."""
        self._expanded = bool(expanded)
        self._list_widget.setVisible(self._expanded)
        self._hint_label.setVisible(self._expanded)
        self._add_btn.setVisible(self._expanded)
        self._layout.setStretch(3, 0 if self._expanded else 1)
        if self._expanded:
            self._toggle_btn.setText("◀  DATA SOURCES")
            self._toggle_btn.setToolTip("Collapse Data Sources")
            self.setMinimumWidth(160)
            self.setMaximumWidth(16777215)
        else:
            self._toggle_btn.setText("▶")
            self._toggle_btn.setToolTip("Expand Data Sources")
            self.setFixedWidth(48)
        self.toggled.emit(self._expanded)


class PanelyticsWindow(QMainWindow):
    """Main 4-section Panelytics desktop GUI window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._settings = load_settings()
        self._theme: Theme = get_theme(self._settings.get("ui_theme"))
        self._sources: list[DataSourceRecord] = []
        self._extra_eval_globals: dict[str, Any] = {}
        self._last_dataframe: pl.DataFrame | None = None
        self._last_regime_col: str | None = None
        self._row_errors: dict[int, str] = {}
        self._updating_formula_table = False
        self._initial_layout_applied = False
        self._saved_left_width = 260

        self.setWindowTitle("panelytics")
        self.resize(1280, 820)
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowMinimizeButtonHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        outer = QWidget()
        outer.setObjectName("outerRoot")
        outer.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setCentralWidget(outer)
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(16, 16, 16, 16)

        shell = RoundedShell()
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)

        # ── Section (1): Top Toolbar with GUI Controls, Add/Remove Functions, Square Gears Settings
        toolbar_trailing = QWidget()
        toolbar_trailing.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        tb_layout = QHBoxLayout(toolbar_trailing)
        tb_layout.setContentsMargins(0, 0, 0, 0)
        tb_layout.setSpacing(8)

        self._status_pill = QLabel("Ready")
        self._status_pill.setObjectName("statusPill")
        self._status_pill.setProperty("busy", False)
        tb_layout.addWidget(self._status_pill)

        self._eval_btn = QPushButton("Evaluate && Plot")
        self._eval_btn.setObjectName("sendBtn")
        self._eval_btn.setToolTip("Evaluate all formula rows and render the plot (F5)")
        self._eval_btn.setShortcut(QKeySequence("F5"))
        self._eval_btn.clicked.connect(self._evaluate_and_plot)
        tb_layout.addWidget(self._eval_btn)

        self._add_source_btn = QPushButton("Add Source…")
        self._add_source_btn.setObjectName("toolbarBtn")
        self._add_source_btn.setToolTip("Add a CSV, Parquet, SQLite, or DuckDB data source")
        self._add_source_btn.clicked.connect(self._browse_add_source)
        tb_layout.addWidget(self._add_source_btn)

        self._download_btn = QPushButton("Download Data")
        self._download_btn.setObjectName("toolbarBtn")
        self._download_btn.setEnabled(False)
        self._download_btn.clicked.connect(self._download_data)
        tb_layout.addWidget(self._download_btn)

        self._copy_chart_btn = QPushButton("Copy Chart")
        self._copy_chart_btn.setObjectName("toolbarBtn")
        self._copy_chart_btn.setEnabled(False)
        self._copy_chart_btn.clicked.connect(self._copy_chart_image)
        tb_layout.addWidget(self._copy_chart_btn)

        # To the left of the settings button: "Add/Remove Functions" button
        self._functions_btn = QPushButton("Add/Remove Functions")
        self._functions_btn.setObjectName("toolbarBtn")
        self._functions_btn.setToolTip(
            "Configure Python directories and imports available in panelyzer formulas"
        )
        self._functions_btn.clicked.connect(self._open_functions_dialog)
        tb_layout.addWidget(self._functions_btn)

        # Far right of toolbar controls: Square Settings button with gears icon
        self._settings_btn = QPushButton("⚙")
        self._settings_btn.setObjectName("gearBtn")
        self._settings_btn.setFixedSize(34, 34)
        self._settings_btn.setToolTip("Settings")
        self._settings_btn.clicked.connect(self._open_settings)
        tb_layout.addWidget(self._settings_btn)

        shell_layout.addWidget(
            TitleBar(
                self,
                "panelytics",
                shell,
                trailing=toolbar_trailing,
                subtitle="Panel expression analytics & visualization",
                minimizable=True,
            )
        )

        # ── Main Landscape Body (Divided into 3 sections below toolbar) ──────
        content = QWidget()
        content.setObjectName("dialogContent")
        content.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(16, 14, 16, 16)
        content_layout.setSpacing(10)

        self._body_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._body_splitter.setObjectName("bodySplitter")
        self._body_splitter.setChildrenCollapsible(False)

        # ── Section (2): Left Collapsible "Data Sources" (22% of width) ──────
        self._sources_list = DataSourcesListWidget()
        self._sources_list.files_dropped.connect(self._on_files_dropped)
        self._sources_list.modify_requested.connect(self._on_modify_source)
        self._sources_list.delete_requested.connect(self._on_delete_source)
        self._sources_list.paste_requested.connect(self._on_paste_source)
        self._sources_list.model().rowsMoved.connect(self._on_sources_reordered)

        self._sources_pane = CollapsibleDataSourcesPane(self._sources_list)
        self._sources_pane.toggled.connect(self._on_sources_pane_toggled)
        self._sources_pane.add_file_requested.connect(self._browse_add_source)
        self._body_splitter.addWidget(self._sources_pane)

        # ── Right Space (78% of width): Vertically divided into Top (25%) & Bottom (75%)
        self._right_splitter = QSplitter(Qt.Orientation.Vertical)
        self._right_splitter.setObjectName("rightSplitter")
        self._right_splitter.setChildrenCollapsible(False)

        # ── Section (3): Top Formula Sheet (fixed 2 columns, max 10 rows) ────
        formula_header_controls = QWidget()
        formula_header_controls.setAttribute(
            Qt.WidgetAttribute.WA_TranslucentBackground
        )
        fh_layout = QHBoxLayout(formula_header_controls)
        fh_layout.setContentsMargins(0, 0, 0, 0)
        fh_layout.setSpacing(6)

        self._row_count_label = QLabel("1 / 10 rows")
        self._row_count_label.setObjectName("statusLabel")
        fh_layout.addWidget(self._row_count_label)

        self._add_row_btn = QPushButton("+ Add Row")
        self._add_row_btn.setObjectName("smallBtn")
        self._add_row_btn.setToolTip("Add a new formula row (maximum 10 rows)")
        self._add_row_btn.clicked.connect(self._add_formula_row)
        fh_layout.addWidget(self._add_row_btn)

        self._del_row_btn = QPushButton("− Delete Row")
        self._del_row_btn.setObjectName("smallBtn")
        self._del_row_btn.setToolTip("Delete the selected formula row")
        self._del_row_btn.clicked.connect(self._delete_selected_formula_row)
        fh_layout.addWidget(self._del_row_btn)

        formula_panel = PanelFrame(
            "PANELYZER FORMULAS", header_trailing=formula_header_controls
        )

        self._formula_table = QTableWidget(0, 2)
        self._formula_table.setObjectName("formulaTable")
        self._formula_table.setHorizontalHeaderLabels(
            ["Regime", "Panelyzer Formula"]
        )
        self._formula_table.verticalHeader().setDefaultSectionSize(30)
        self._formula_table.verticalHeader().setVisible(True)
        self._formula_table.horizontalHeader().setFixedHeight(28)
        self._formula_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Fixed
        )
        self._formula_table.setColumnWidth(0, 68)
        self._formula_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self._formula_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self._formula_table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection
        )
        self._formula_table.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._formula_table.customContextMenuRequested.connect(
            self._on_formula_table_context_menu
        )

        self._formula_delegate = FormulaDelegate(self._formula_table)
        self._formula_table.setItemDelegateForColumn(1, self._formula_delegate)
        self._formula_table.itemChanged.connect(self._on_formula_item_changed)

        formula_panel.set_content(self._formula_table)
        self._right_splitter.addWidget(formula_panel)

        # ── Section (4): Bottom Visualization Area (~75% of right height) ────
        viz_header_controls = QWidget()
        viz_header_controls.setAttribute(
            Qt.WidgetAttribute.WA_TranslucentBackground
        )
        vh_layout = QHBoxLayout(viz_header_controls)
        vh_layout.setContentsMargins(0, 0, 0, 0)
        vh_layout.setSpacing(10)

        self._view_data_btn = QToolButton()
        self._view_data_btn.setObjectName("ViewDataButton")
        self._view_data_btn.setIcon(eye_icon(self._theme))
        self._view_data_btn.setIconSize(QSize(18, 18))
        self._view_data_btn.setFixedSize(28, 28)
        self._view_data_btn.setToolTip("View data for the current graph")
        self._view_data_btn.setEnabled(False)
        self._view_data_btn.clicked.connect(self._show_graph_data)
        vh_layout.addWidget(self._view_data_btn)

        self._date_range_slider = DateRangeSlider()
        self._date_range_slider.setMinimumWidth(280)
        self._date_range_slider.range_changed.connect(
            self._on_date_range_slider_changed
        )
        vh_layout.addWidget(self._date_range_slider, stretch=1)

        plot_panel = PanelFrame(
            "VISUALIZATION", header_trailing=viz_header_controls
        )
        self._plot = PlotWidget()
        self._plot.setObjectName("dataPlot")
        self._plot.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        plot_panel.set_content(self._plot)
        self._right_splitter.addWidget(plot_panel)

        self._right_splitter.setStretchFactor(0, 1)
        self._right_splitter.setStretchFactor(1, 3)

        self._body_splitter.addWidget(self._right_splitter)
        self._body_splitter.setStretchFactor(0, 22)
        self._body_splitter.setStretchFactor(1, 78)

        content_layout.addWidget(self._body_splitter, stretch=1)
        shell_layout.addWidget(content, stretch=1)
        outer_layout.addWidget(shell)

        self._resize_helper = FramelessResizeHelper(self)
        self._resize_helper.install()

        self._apply_theme()
        self._restore_persisted_state()

    def eventFilter(self, watched, event) -> bool:
        if self._resize_helper.handle_event_filter(watched, event):
            return True
        return super().eventFilter(watched, event)

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if not self._initial_layout_applied:
            self._initial_layout_applied = True
            self._apply_initial_splitter_ratios()

    def _apply_initial_splitter_ratios(self) -> None:
        total_w = max(self._body_splitter.width(), self.width() - 64)
        left_w = int(total_w * 0.22)
        right_w = max(100, total_w - left_w)
        self._saved_left_width = left_w
        self._body_splitter.setSizes([left_w, right_w])

        total_h = max(self._right_splitter.height(), self.height() - 100)
        top_h = int(total_h * 0.25)
        bot_h = max(100, total_h - top_h)
        self._right_splitter.setSizes([top_h, bot_h])

    def _on_sources_pane_toggled(self, expanded: bool) -> None:
        total_w = sum(self._body_splitter.sizes())
        if not expanded:
            current_sizes = self._body_splitter.sizes()
            if current_sizes and current_sizes[0] > 60:
                self._saved_left_width = current_sizes[0]
            self._body_splitter.setSizes([48, max(100, total_w - 48)])
        else:
            restore_w = max(160, self._saved_left_width or int(total_w * 0.22))
            self._body_splitter.setSizes([restore_w, max(100, total_w - restore_w)])

    # ── Persistence & Initialization ─────────────────────────────────────────

    def _restore_persisted_state(self) -> None:
        # 1. User functions
        uf = self._settings.get("user_functions") or {}
        pythonpath = list(uf.get("pythonpath") or [])
        imports = list(uf.get("imports") or [])
        self._extra_eval_globals, _errs = load_user_eval_globals(pythonpath, imports)

        # 2. Data sources
        saved_sources = self._settings.get("data_sources") or []
        self._sources = []
        for raw in saved_sources:
            if not isinstance(raw, dict):
                continue
            try:
                rec = DataSourceRecord(
                    name=str(raw.get("name", "source")),
                    source_type=str(raw.get("source_type", "csv")),
                    path=raw.get("path"),
                    table_name=raw.get("table_name"),
                    clipboard_text=raw.get("clipboard_text"),
                    date_col=str(raw.get("date_col", "date")),
                    entity_col=str(raw.get("entity_col", "entity")),
                    default_feature=str(raw.get("default_feature", "")),
                    date_col_type=str(raw.get("date_col_type", "date")),
                )
                if rec.source_type != "clipboard" and rec.path and not Path(rec.path).exists():
                    continue
                populate_source_metadata(rec, estimate_missing=False)
                self._sources.append(rec)
            except Exception:
                continue

        self._refresh_sources_list_ui()

        # 3. Formula rows
        saved_rows = self._settings.get("formula_rows") or [
            {"is_regime": False, "formula": ""}
        ]
        self._load_formula_rows_into_table(saved_rows)
        self._refresh_autocomplete()

    def _sync_state_to_settings(self) -> None:
        self._settings["data_sources"] = [s.to_dict() for s in self._sources]
        self._settings["formula_rows"] = self._collect_formula_rows()
        save_settings(self._settings)

    # ── Data Sources Management (Section 2) ──────────────────────────────────

    def _refresh_sources_list_ui(self) -> None:
        self._sources_list.clear()
        for src in self._sources:
            item = QListWidgetItem(src.name)
            item.setSizeHint(QSize(0, 34))
            tooltip_lines = [
                f"Name: {src.name}",
                f"Type: {src.source_type.upper()}",
            ]
            if src.path:
                tooltip_lines.append(f"Path: {src.path}")
            if src.table_name:
                tooltip_lines.append(f"Table: {src.table_name}")
            tooltip_lines.extend(
                [
                    f"date_col: {src.date_col} ({src.date_col_type})",
                    f"entity_col: {src.entity_col} ({len(src.entities)} entities)",
                    f"default_feature: {src.default_feature}",
                    f"Features: {', '.join(src.features[:12])}"
                    + ("…" if len(src.features) > 12 else ""),
                ]
            )
            item.setToolTip("\n".join(tooltip_lines))
            item.setData(Qt.ItemDataRole.UserRole, src.name)
            self._sources_list.addItem(item)

    def _existing_source_names(self) -> set[str]:
        return {s.name for s in self._sources}

    def _browse_add_source(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Data Source(s)",
            str(Path.home()),
            "Supported Data Files (*.csv *.parquet *.pq *.db *.sqlite *.sqlite3 *.duckdb);;All Files (*.*)",
        )
        if files:
            self._on_files_dropped(files)

    def _on_files_dropped(self, file_paths: list[str]) -> None:
        added_any = False
        for path in file_paths:
            try:
                rec = create_source_from_path(path, self._existing_source_names())
                self._sources.append(rec)
                added_any = True
            except Exception as exc:
                QMessageBox.warning(
                    self, "Could Not Add Data Source", f"{path}:\n{exc}"
                )

        if added_any:
            self._refresh_sources_list_ui()
            self._refresh_autocomplete()
            self._sync_state_to_settings()
            self._status_pill.setText(f"{len(self._sources)} source(s) loaded")

    def _on_paste_source(self) -> None:
        clip_text = QApplication.clipboard().text()
        try:
            rec = create_source_from_clipboard(
                clip_text, self._existing_source_names()
            )
        except Exception as exc:
            QMessageBox.warning(self, "Paste Data Source", str(exc))
            return

        self._sources.append(rec)
        self._refresh_sources_list_ui()
        self._refresh_autocomplete()
        self._sync_state_to_settings()
        self._status_pill.setText(f"Pasted source '{rec.name}'")

    def _on_modify_source(self, row: int) -> None:
        if not (0 <= row < len(self._sources)):
            return
        rec = self._sources[row]
        dlg = EntityConfigurationDialog(rec, parent=self, theme=self._theme)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._refresh_sources_list_ui()
            self._refresh_autocomplete()
            self._sync_state_to_settings()
            if self._has_any_formula():
                self._evaluate_and_plot()

    def _on_delete_source(self, row: int) -> None:
        if not (0 <= row < len(self._sources)):
            return
        removed = self._sources.pop(row)
        remaining_rows = [
            r
            for r in self._collect_formula_rows()
            if not formula_uses_source(r.get("formula", ""), removed)
        ]
        self._load_formula_rows_into_table(remaining_rows)
        self._refresh_sources_list_ui()
        self._refresh_autocomplete()
        self._sync_state_to_settings()
        self._status_pill.setText(f"Deleted source '{removed.name}'")
        if self._sources and self._has_any_formula():
            self._evaluate_and_plot()
        else:
            self._clear_visualization()

    def _on_sources_reordered(self) -> None:
        by_name = {s.name: s for s in self._sources}
        new_order: list[DataSourceRecord] = []
        for i in range(self._sources_list.count()):
            name = self._sources_list.item(i).data(Qt.ItemDataRole.UserRole)
            if name in by_name:
                new_order.append(by_name[name])
        if len(new_order) == len(self._sources):
            self._sources = new_order
            self._sync_state_to_settings()

    # ── Formula Table Management (Section 3) ─────────────────────────────────

    def _load_formula_rows_into_table(self, rows: list[dict[str, Any]]) -> None:
        self._row_errors.clear()
        self._updating_formula_table = True
        try:
            self._formula_table.setRowCount(0)
            regime_assigned = False
            for r in rows[:MAX_FORMULA_ROWS]:
                is_reg = bool(r.get("is_regime", False)) and not regime_assigned
                if is_reg:
                    regime_assigned = True
                self._insert_formula_row_widgets(
                    is_regime=is_reg,
                    formula=str(r.get("formula", "")),
                )
            if self._formula_table.rowCount() == 0:
                self._insert_formula_row_widgets(is_regime=False, formula="")
        finally:
            self._updating_formula_table = False
        self._update_formula_row_buttons()

    def _insert_formula_row_widgets(self, *, is_regime: bool, formula: str) -> int:
        row_idx = self._formula_table.rowCount()
        self._formula_table.insertRow(row_idx)

        # Column 0: Centered Checkbox
        cb_container = QWidget()
        cb_container.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        cb_layout = QHBoxLayout(cb_container)
        cb_layout.setContentsMargins(0, 0, 0, 0)
        cb_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        cb = QCheckBox()
        cb.setObjectName("regimeCheckBox")
        cb.setChecked(is_regime)
        cb.setToolTip(
            "Use this row's evaluated values as the plot regime background shading"
        )
        cb.toggled.connect(
            lambda checked, btn=cb: self._on_regime_checkbox_toggled(btn, checked)
        )
        cb_layout.addWidget(cb)
        self._formula_table.setCellWidget(row_idx, 0, cb_container)

        # Column 1: Panelyzer formula cell
        item = QTableWidgetItem(formula)
        self._formula_table.setItem(row_idx, 1, item)
        return row_idx

    def _get_row_checkbox(self, row: int) -> QCheckBox | None:
        container = self._formula_table.cellWidget(row, 0)
        if container is None:
            return None
        return container.findChild(QCheckBox)

    def _on_regime_checkbox_toggled(
        self, sender_cb: QCheckBox, checked: bool
    ) -> None:
        if self._updating_formula_table:
            return
        self._updating_formula_table = True
        try:
            if checked:
                # Enforce at most 1 checkbox ticked across all rows
                for r in range(self._formula_table.rowCount()):
                    cb = self._get_row_checkbox(r)
                    if cb is not None and cb is not sender_cb and cb.isChecked():
                        cb.setChecked(False)
        finally:
            self._updating_formula_table = False

        self._sync_state_to_settings()
        if self._has_any_formula() and self._sources:
            self._evaluate_and_plot()

    def _on_formula_item_changed(self, _item: QTableWidgetItem) -> None:
        if self._updating_formula_table:
            return
        self._sync_state_to_settings()
        if self._sources and self._has_any_formula():
            self._evaluate_and_plot()

    def _add_formula_row(self) -> None:
        if self._row_errors:
            first_err_row = min(self._row_errors)
            self._formula_table.selectRow(first_err_row)
            self._formula_table.setCurrentCell(first_err_row, 1)
            return
        if self._formula_table.rowCount() >= MAX_FORMULA_ROWS:
            return
        self._updating_formula_table = True
        try:
            new_row = self._insert_formula_row_widgets(is_regime=False, formula="")
        finally:
            self._updating_formula_table = False
        self._update_formula_row_buttons()
        self._sync_state_to_settings()
        self._formula_table.setCurrentCell(new_row, 1)
        self._formula_table.editItem(self._formula_table.item(new_row, 1))

    def _delete_selected_formula_row(self) -> None:
        self._row_errors.clear()
        row_count = self._formula_table.rowCount()
        if row_count <= 1:
            # Clear the single remaining row instead of leaving 0 rows
            self._updating_formula_table = True
            try:
                cb = self._get_row_checkbox(0)
                if cb:
                    cb.setChecked(False)
                item = self._formula_table.item(0, 1)
                if item:
                    item.setText("")
            finally:
                self._updating_formula_table = False
            self._apply_row_error_highlights()
            self._sync_state_to_settings()
            self._clear_visualization()
            return

        target_row = self._formula_table.currentRow()
        if target_row < 0:
            target_row = row_count - 1
        self._formula_table.removeRow(target_row)
        self._apply_row_error_highlights()
        self._sync_state_to_settings()
        if self._sources and self._has_any_formula():
            self._evaluate_and_plot()
        else:
            self._clear_visualization()

    def _on_formula_table_context_menu(self, pos: QPoint) -> None:
        menu = QMenu(self)
        add_act = QAction("Add Row", self)
        add_act.setEnabled(
            self._formula_table.rowCount() < MAX_FORMULA_ROWS
            and not self._row_errors
        )
        add_act.triggered.connect(self._add_formula_row)
        menu.addAction(add_act)

        del_act = QAction("Delete Row", self)
        del_act.triggered.connect(self._delete_selected_formula_row)
        menu.addAction(del_act)

        eval_act = QAction("Evaluate & Plot", self)
        eval_act.triggered.connect(self._evaluate_and_plot)
        menu.addSeparator()
        menu.addAction(eval_act)

        menu.exec(self._formula_table.viewport().mapToGlobal(pos))

    def _update_formula_row_buttons(self) -> None:
        n = self._formula_table.rowCount()
        self._row_count_label.setText(f"{n} / {MAX_FORMULA_ROWS} rows")
        self._add_row_btn.setEnabled(n < MAX_FORMULA_ROWS and not self._row_errors)
        self._del_row_btn.setEnabled(n >= 1)

    def _apply_row_error_highlights(self) -> None:
        self._updating_formula_table = True
        try:
            err_color = QColor(self._theme.total_err)
            err_bg = QColor(err_color)
            err_bg.setAlpha(65)
            for r in range(self._formula_table.rowCount()):
                item = self._formula_table.item(r, 1)
                container = self._formula_table.cellWidget(r, 0)
                err_msg = self._row_errors.get(r)
                if item is not None:
                    if err_msg:
                        item.setBackground(QBrush(err_bg))
                        item.setForeground(QBrush(err_color))
                        item.setToolTip(
                            f"Formula Error (fix or delete row {r + 1}): {err_msg}"
                        )
                    else:
                        item.setData(Qt.ItemDataRole.BackgroundRole, None)
                        item.setData(Qt.ItemDataRole.ForegroundRole, None)
                        item.setToolTip("")
                if container is not None:
                    container.setStyleSheet(
                        f"background-color: rgba({err_color.red()}, {err_color.green()}, {err_color.blue()}, 65);"
                        if err_msg
                        else "background: transparent;"
                    )
        finally:
            self._updating_formula_table = False
        self._update_formula_row_buttons()

    def _collect_formula_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for r in range(self._formula_table.rowCount()):
            cb = self._get_row_checkbox(r)
            item = self._formula_table.item(r, 1)
            rows.append(
                {
                    "is_regime": bool(cb.isChecked()) if cb else False,
                    "formula": item.text().strip() if item else "",
                }
            )
        return rows

    def _has_any_formula(self) -> bool:
        return any(r["formula"] for r in self._collect_formula_rows())

    def _refresh_autocomplete(self) -> None:
        tokens = build_autocomplete_tokens(
            self._sources, self._extra_eval_globals
        )
        self._formula_delegate.set_tokens(tokens)

    # ── Add/Remove Functions & Settings Dialogs (Section 1) ──────────────────

    def _open_functions_dialog(self) -> None:
        uf = self._settings.get("user_functions") or {}
        dlg = UserFunctionsDialog(
            pythonpath=list(uf.get("pythonpath") or []),
            imports=list(uf.get("imports") or []),
            parent=self,
            theme=self._theme,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._settings["user_functions"] = {
                "pythonpath": dlg.pythonpath,
                "imports": dlg.imports,
            }
            self._extra_eval_globals, errs = load_user_eval_globals(
                dlg.pythonpath, dlg.imports
            )
            self._refresh_autocomplete()
            self._sync_state_to_settings()
            if errs:
                self._status_pill.setText(f"Import warning: {errs[0]}")
            else:
                self._status_pill.setText(
                    f"Loaded {len(self._extra_eval_globals)} user symbol(s)"
                )
            if self._sources and self._has_any_formula():
                self._evaluate_and_plot()

    def _preview_ui_theme(self, theme_name: str) -> None:
        self._settings["ui_theme"] = theme_name
        self._apply_theme()

    def _open_settings(self) -> None:
        original_theme = self._settings.get("ui_theme")
        original_plot_settings = dict(self._settings.get("plot_settings", {}))
        original_download_settings = dict(
            self._settings.get("download_settings", {})
        )

        dlg = SettingsDialog(
            plot_settings=self._settings.get("plot_settings"),
            download_settings=self._settings.get("download_settings"),
            parent=self,
            theme=self._theme.name,
        )
        dlg.theme_changed.connect(self._preview_ui_theme)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            self._settings["ui_theme"] = original_theme
            self._settings["plot_settings"] = original_plot_settings
            self._settings["download_settings"] = original_download_settings
            self._apply_theme()
            return

        new_plot_settings = dict(dlg.plot_settings)
        plot_settings_changed = new_plot_settings != original_plot_settings

        self._settings["ui_theme"] = dlg.ui_theme_name
        self._settings["plot_settings"] = new_plot_settings
        self._settings["download_settings"] = dlg.download_settings
        self._sync_state_to_settings()
        self._apply_theme()

        if (
            plot_settings_changed
            and self._last_dataframe is not None
            and not self._last_dataframe.is_empty()
        ):
            disp_df = self._displayed_dataframe()
            if disp_df is not None:
                self._render_plot(disp_df, self._last_regime_col)

    def _apply_theme(self) -> None:
        self._theme = get_theme(self._settings.get("ui_theme"))
        self.setStyleSheet(window_style(self._theme))
        self._view_data_btn.setIcon(eye_icon(self._theme))
        primary, primary_hover = self._theme.accents[0]
        self._date_range_slider.update_theme(
            self._theme.slider_groove,
            primary,
            primary_hover,
            self._theme.slider_handle,
        )
        self._apply_row_error_highlights()
        if self._plot.has_plot:
            plot_bg = get_plot_bg(self._theme.name)
            self._plot.figure.patch.set_facecolor(plot_bg)
            if self._plot.ax is not None:
                self._plot.ax.set_facecolor(plot_bg)
            disp_df = self._displayed_dataframe()
            if disp_df is not None and self._last_regime_col:
                self._apply_regime_overlay(disp_df, self._last_regime_col)
            self._plot.canvas.draw_idle()
        self.update()
        QApplication.processEvents()

    # ── Formula Evaluation & Visualization (Section 4) ───────────────────────

    def _clear_visualization(self) -> None:
        self._last_dataframe = None
        self._last_regime_col = None
        self._date_range_slider.set_dates([])
        self._plot.clear_plot()
        self._view_data_btn.setEnabled(False)
        self._download_btn.setEnabled(False)
        self._copy_chart_btn.setEnabled(False)

    def _displayed_dataframe(self) -> pl.DataFrame | None:
        """Return the slice of `_last_dataframe` within the active date range slider."""
        if self._last_dataframe is None or self._last_dataframe.is_empty():
            return self._last_dataframe
        low, high = self._date_range_slider.selected_indices
        low = max(0, min(low, self._last_dataframe.height - 1))
        high = max(low, min(high, self._last_dataframe.height - 1))
        return self._last_dataframe.slice(low, high - low + 1)

    def _on_date_range_slider_changed(self, _low: int, _high: int) -> None:
        disp_df = self._displayed_dataframe()
        if disp_df is None or disp_df.is_empty():
            return
        self._render_plot(disp_df, self._last_regime_col)

    def _show_graph_data(self) -> None:
        """Open a cloned, filterable table of the current graph's data."""
        disp_df = self._displayed_dataframe()
        if disp_df is None or disp_df.is_empty():
            return
        DataTableDialog(
            disp_df,
            title="Visualization",
            theme=self._theme,
            parent=self,
        ).exec()

    def _evaluate_and_plot(self) -> None:
        rows = self._collect_formula_rows()
        row_errors: dict[int, str] = {}
        df, regime_col, messages = evaluate_formula_rows(
            self._sources,
            rows,
            self._extra_eval_globals,
            row_errors_out=row_errors,
        )
        self._row_errors = row_errors
        self._apply_row_error_highlights()

        if row_errors:
            self._clear_visualization()
            first_err_row = min(row_errors)
            first_err_msg = row_errors[first_err_row]
            self._formula_table.selectRow(first_err_row)
            self._formula_table.setCurrentCell(first_err_row, 1)
            self._status_pill.setText(
                f"Error in Row {first_err_row + 1} (fix or delete row): {first_err_msg}"
            )
            return

        if df is None or df.is_empty():
            self._clear_visualization()
            msg = messages[0] if messages else "No data returned from formulas."
            self._status_pill.setText(msg)
            return

        df = coerce_date_columns(df)
        self._last_dataframe = df
        self._last_regime_col = regime_col
        x_col = df.columns[0]
        date_labels = [format_display_date(v) or str(v) for v in df[x_col].to_list()]
        self._date_range_slider.set_dates(date_labels)
        disp_df = self._displayed_dataframe()
        self._render_plot(disp_df if disp_df is not None else df, regime_col)

        if messages:
            self._status_pill.setText(f"Plotted ({df.height} rows) · {messages[0]}")
        else:
            regime_note = f" [Regime: {regime_col}]" if regime_col else ""
            self._status_pill.setText(
                f"Plotted {df.height} rows × {len(df.columns) - 1} series{regime_note}"
            )

    def _render_plot(
        self, df: pl.DataFrame, regime_col: str | None = None
    ) -> None:
        if df.is_empty():
            self._plot.clear_plot()
            return

        x_col, all_y_cols = infer_plot_columns(df)
        # Exclude the regime column from plotted y-series if other non-regime series exist
        non_regime_y = [c for c in all_y_cols if c != regime_col]
        y_cols_to_plot = non_regime_y if non_regime_y else all_y_cols
        if not y_cols_to_plot:
            self._plot.clear_plot()
            return

        sub_df = df.select([x_col, *y_cols_to_plot])
        plot_df, y_cols, has_multiple_series = process_df_for_ggplot(
            sub_df, x_col
        )
        plot_settings = self._settings.get("plot_settings", {})
        theme_str = str(plot_settings.get("theme_str", "538"))
        flip_coord = bool(plot_settings.get("flip_coord", False))

        plot = create_ggplot_from_df(
            plot_df,
            x_col,
            y_cols,
            "",
            None,
            has_multiple_series,
            plot_settings.get("geoms", "LP"),
            plot_settings.get("color", True),
            plot_settings.get("alpha", 1.0),
            plot_settings.get("size", False),
            plot_settings.get("shape", False),
            plot_settings.get("group", False),
            theme_str,
            flip_coord,
            plot_settings.get("y_scale", "linear"),
            plot_settings.get("pos_legend", "bottom"),
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", PlotnineWarning)
                figure = plot.draw()
        except (IndexError, ValueError, TypeError, PlotnineError):
            plot = create_ggplot_from_df(
                plot_df,
                x_col,
                y_cols,
                "",
                None,
                has_multiple_series,
                plot_settings.get("geoms", "LP"),
                plot_settings.get("color", True),
                plot_settings.get("alpha", 1.0),
                plot_settings.get("size", False),
                plot_settings.get("shape", False),
                plot_settings.get("group", False),
                theme_str,
                flip_coord,
                "linear",
                plot_settings.get("pos_legend", "bottom"),
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", PlotnineWarning)
                try:
                    figure = plot.draw()
                except PlotnineError as exc:
                    self._status_pill.setText(f"Plot error: {exc}")
                    return

        for axis in figure.axes:
            axis.xaxis.set_minor_locator(AutoMinorLocator(3))
            axis.yaxis.set_minor_locator(AutoMinorLocator(3))

        self._plot.display_figure(
            figure,
            series_names=y_cols,
            flip_coord=flip_coord,
            x_values=sub_df[x_col].to_list(),
        )

        if regime_col and regime_col in df.columns:
            self._apply_regime_overlay(df, regime_col)

        self._view_data_btn.setEnabled(True)
        self._download_btn.setEnabled(True)
        self._copy_chart_btn.setEnabled(True)

    def _apply_regime_overlay(self, df: pl.DataFrame, regime_col: str) -> None:
        if regime_col not in df.columns or df.is_empty():
            return
        x_col = df.columns[0]
        plot_settings = self._settings.get("plot_settings", {})
        self._plot.apply_regime_shading(
            df[x_col].to_list(),
            df[regime_col].to_list(),
            plot_theme_str=str(plot_settings.get("theme_str", "538")),
            ui_theme=self._theme,
            flip_coord=bool(plot_settings.get("flip_coord", False)),
        )

    # ── Export & Clipboard Actions ───────────────────────────────────────────

    def _download_data(self) -> None:
        disp_df = self._displayed_dataframe()
        if disp_df is None or disp_df.is_empty():
            QMessageBox.information(
                self, "No Data", "No evaluated data available to download."
            )
            return

        download_settings = self._settings.get("download_settings", {})
        file_format = download_settings.get("format", "csv")
        include_headers = bool(download_settings.get("headers", True))
        directory = download_settings.get("directory") or str(Path.home())

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"panelytics_{timestamp}.{file_format}"

        if file_format == "csv":
            file_filter = "CSV Files (*.csv);;All Files (*.*)"
        else:
            file_filter = "Parquet Files (*.parquet);;All Files (*.*)"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Data",
            str(Path(directory) / default_name),
            file_filter,
        )
        if not file_path:
            return

        try:
            if file_format == "csv":
                disp_df.write_csv(
                    file_path, include_header=include_headers
                )
            else:
                disp_df.write_parquet(file_path)

            QMessageBox.information(
                self, "Download Complete", f"Data saved to:\n{file_path}"
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "Download Failed", f"Could not save file:\n{exc}"
            )

    def _copy_chart_image(self) -> None:
        if not self._plot.has_plot or self._plot.figure is None:
            return

        buffer = BytesIO()
        self._plot.figure.savefig(
            buffer,
            format="png",
            dpi=150,
            bbox_inches="tight",
            facecolor=self._plot.figure.get_facecolor(),
            edgecolor="none",
        )
        buffer.seek(0)

        image = QImage()
        image.loadFromData(buffer.getvalue())
        QApplication.clipboard().setImage(image)
        self._status_pill.setText("Chart copied to clipboard")

    def closeEvent(self, event: QCloseEvent) -> None:
        self._sync_state_to_settings()
        self._resize_helper.uninstall()
        super().closeEvent(event)


def main() -> None:
    """Launch the Panelytics PySide6 GUI application."""
    qt_app = QApplication.instance() or QApplication(sys.argv)
    font = QFont("Segoe UI", 10)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    qt_app.setFont(font)

    window = PanelyticsWindow()
    window.show()
    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()
