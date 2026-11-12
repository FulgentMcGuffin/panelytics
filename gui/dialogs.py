from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui.gui_constants import Theme, dialog_style, get_theme
from gui.gui_utils import TitleBar
from gui.panelyzer_bridge import (
    DataSourceRecord,
    estimate_source_columns,
    load_user_eval_globals,
    parse_import_statement,
    populate_source_metadata,
    sample_database_table,
)


class EntityConfigurationDialog(QDialog):
    """Dialog to configure a DataSourceRecord as a panelyzer EntityConfiguration."""

    def __init__(
        self,
        record: DataSourceRecord,
        parent: QWidget | None = None,
        theme: str | Theme | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = get_theme(theme)
        self._record = record

        self.setStyleSheet(dialog_style(self._theme))
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Dialog
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowTitle("Configure Data Source")
        self.setMinimumWidth(400)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        shell = QWidget()
        shell.setObjectName("dialogShell")
        shell.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)

        shell_layout.addWidget(TitleBar(self, "Configure EntityConfiguration", shell))

        content = QWidget()
        content.setObjectName("dialogContent")
        content.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 12, 16, 16)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setSpacing(10)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._name_input = QLineEdit(record.name)
        form.addRow("Source Name:", self._name_input)

        type_label = QLabel(record.source_type.upper())
        type_label.setObjectName("statusLabel")
        form.addRow("Source Type:", type_label)

        self._table_combo: QComboBox | None = None
        if record.source_type in ("sqlite", "duckdb"):
            self._table_combo = QComboBox()
            self._table_combo.addItems(record.available_tables)
            if record.table_name:
                idx = self._table_combo.findText(record.table_name)
                if idx >= 0:
                    self._table_combo.setCurrentIndex(idx)
            self._table_combo.currentTextChanged.connect(self._on_table_changed)
            form.addRow("Table:", self._table_combo)

        self._date_col_combo = QComboBox()
        self._entity_col_combo = QComboBox()
        self._default_feature_combo = QComboBox()
        self._date_type_combo = QComboBox()
        self._date_type_combo.addItems(["date", "datetime"])

        self._populate_column_combos(
            record.available_columns,
            record.date_col,
            record.entity_col,
            record.default_feature,
            record.date_col_type,
        )

        form.addRow("date_col:", self._date_col_combo)
        form.addRow("entity_col:", self._entity_col_combo)
        form.addRow("default_feature:", self._default_feature_combo)
        form.addRow("date_col_type:", self._date_type_combo)

        layout.addLayout(form)

        info_text = (
            f"Entities detected: {len(record.entities)}  |  "
            f"Features: {len(record.features)}"
        )
        self._info_label = QLabel(info_text)
        self._info_label.setObjectName("statusLabel")
        layout.addWidget(self._info_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        shell_layout.addWidget(content)
        outer.addWidget(shell)

    def _populate_column_combos(
        self,
        columns: list[str],
        date_col: str,
        entity_col: str,
        default_feature: str,
        date_col_type: str,
    ) -> None:
        for combo, selected in (
            (self._date_col_combo, date_col),
            (self._entity_col_combo, entity_col),
            (self._default_feature_combo, default_feature),
        ):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(columns)
            idx = combo.findText(selected)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.blockSignals(False)

        dtype_idx = self._date_type_combo.findText(date_col_type)
        if dtype_idx >= 0:
            self._date_type_combo.setCurrentIndex(dtype_idx)

    def _on_table_changed(self, table_name: str) -> None:
        if not table_name or not self._record.path:
            return
        try:
            df = sample_database_table(
                self._record.path, self._record.source_type, table_name
            )
            cols = list(df.columns)
            est_date, est_ent, est_feat, est_dtype = estimate_source_columns(df)
            self._populate_column_combos(cols, est_date, est_ent, est_feat, est_dtype)
        except Exception as exc:
            QMessageBox.warning(self, "Table Read Error", str(exc))

    def _on_accept(self) -> None:
        name = self._name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Invalid Name", "Source name cannot be empty.")
            return

        date_col = self._date_col_combo.currentText()
        entity_col = self._entity_col_combo.currentText()
        default_feature = self._default_feature_combo.currentText()
        date_col_type = self._date_type_combo.currentText()

        if len({date_col, entity_col, default_feature}) < 3 and len(
            self._record.available_columns
        ) >= 3:
            QMessageBox.warning(
                self,
                "Column Selection",
                "Please choose distinct columns for date_col, entity_col, and default_feature.",
            )
            return

        self._record.name = name
        if self._table_combo is not None:
            self._record.table_name = self._table_combo.currentText()
        self._record.date_col = date_col
        self._record.entity_col = entity_col
        self._record.default_feature = default_feature
        self._record.date_col_type = date_col_type

        try:
            populate_source_metadata(self._record, estimate_missing=False)
        except Exception as exc:
            QMessageBox.critical(
                self, "Configuration Error", f"Failed to configure source:\n{exc}"
            )
            return

        self.accept()


class UserFunctionsDialog(QDialog):
    """Dialog to add or remove Python package directories and import statements for panelyzer."""

    def __init__(
        self,
        pythonpath: list[str],
        imports: list[str],
        parent: QWidget | None = None,
        theme: str | Theme | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = get_theme(theme)
        self.pythonpath = list(pythonpath)
        self.imports = list(imports)

        self.setStyleSheet(dialog_style(self._theme))
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Dialog
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowTitle("Add/Remove Functions")
        self.setMinimumWidth(520)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        shell = QWidget()
        shell.setObjectName("dialogShell")
        shell.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)

        shell_layout.addWidget(
            TitleBar(
                self,
                "Add/Remove Functions",
                shell,
                subtitle="Expose Python packages & functions in panelyzer formulas",
            )
        )

        content = QWidget()
        content.setObjectName("dialogContent")
        content.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(12)

        # 1. Python Directories (pythonpath)
        dir_header = QHBoxLayout()
        dir_label = QLabel("PYTHON DIRECTORIES (PYTHONPATH)")
        dir_label.setObjectName("sectionLabel")
        dir_header.addWidget(dir_label)
        dir_header.addStretch()

        add_dir_btn = QPushButton("+ Add Directory…")
        add_dir_btn.setObjectName("smallBtn")
        add_dir_btn.clicked.connect(self._on_add_directory)
        dir_header.addWidget(add_dir_btn)

        rem_dir_btn = QPushButton("Remove Selected")
        rem_dir_btn.setObjectName("smallBtn")
        rem_dir_btn.clicked.connect(self._on_remove_directory)
        dir_header.addWidget(rem_dir_btn)
        layout.addLayout(dir_header)

        self._dir_list = QListWidget()
        self._dir_list.setMaximumHeight(110)
        for p in self.pythonpath:
            self._dir_list.addItem(p)
        layout.addWidget(self._dir_list)

        # 2. Python Import Statements
        imp_label = QLabel("PYTHON IMPORT STATEMENTS")
        imp_label.setObjectName("sectionLabel")
        layout.addWidget(imp_label)

        imp_entry_row = QHBoxLayout()
        self._imp_input = QLineEdit()
        self._imp_input.setPlaceholderText(
            "e.g. import scipy.stats as stats  or  from my_module import soft_clip"
        )
        self._imp_input.returnPressed.connect(self._on_add_import)
        imp_entry_row.addWidget(self._imp_input, stretch=1)

        add_imp_btn = QPushButton("Add Import")
        add_imp_btn.setObjectName("actionBtn")
        add_imp_btn.clicked.connect(self._on_add_import)
        imp_entry_row.addWidget(add_imp_btn)

        rem_imp_btn = QPushButton("Remove Selected")
        rem_imp_btn.setObjectName("actionBtn")
        rem_imp_btn.clicked.connect(self._on_remove_import)
        imp_entry_row.addWidget(rem_imp_btn)
        layout.addLayout(imp_entry_row)

        self._imp_list = QListWidget()
        self._imp_list.setMinimumHeight(130)
        for stmt in self.imports:
            self._imp_list.addItem(stmt)
        layout.addWidget(self._imp_list)

        self._status_label = QLabel("")
        self._status_label.setObjectName("statusLabel")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)
        self._refresh_status()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        shell_layout.addWidget(content)
        outer.addWidget(shell)

    def _collect_current(self) -> tuple[list[str], list[str]]:
        dirs = [self._dir_list.item(i).text() for i in range(self._dir_list.count())]
        imps = [self._imp_list.item(i).text() for i in range(self._imp_list.count())]
        return dirs, imps

    def _refresh_status(self) -> None:
        dirs, imps = self._collect_current()
        ns, errs = load_user_eval_globals(dirs, imps)
        if errs:
            self._status_label.setText(f"⚠ {errs[0]}")
        elif ns:
            names = ", ".join(sorted(ns.keys()))
            self._status_label.setText(f"Available in formulas: {names}")
        else:
            self._status_label.setText(
                "Built-in in formulas: np, sp, pl, m, romean, rostd, rozscore, robeta, rocorr, …"
            )

    def _on_add_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Python Package / Module Directory",
            str(Path.cwd()),
        )
        if not directory:
            return
        resolved = str(Path(directory).resolve())
        existing = [
            self._dir_list.item(i).text() for i in range(self._dir_list.count())
        ]
        if resolved not in existing:
            self._dir_list.addItem(resolved)

        # Auto-discover top-level .py modules or packages in that directory and offer quick add
        dir_path = Path(resolved)
        discovered: list[str] = []
        for child in sorted(dir_path.iterdir()):
            if (
                child.is_file()
                and child.suffix == ".py"
                and not child.name.startswith("_")
            ):
                discovered.append(f"import {child.stem}")
            elif (
                child.is_dir()
                and (child / "__init__.py").is_file()
                and not child.name.startswith("_")
            ):
                discovered.append(f"import {child.name}")

        existing_imps = {
            self._imp_list.item(i).text() for i in range(self._imp_list.count())
        }
        for stmt in discovered:
            if stmt not in existing_imps:
                self._imp_list.addItem(stmt)
                existing_imps.add(stmt)

        self._refresh_status()

    def _on_remove_directory(self) -> None:
        for item in self._dir_list.selectedItems():
            row = self._dir_list.row(item)
            self._dir_list.takeItem(row)
        self._refresh_status()

    def _on_add_import(self) -> None:
        text = self._imp_input.text().strip()
        if not text:
            return
        try:
            parse_import_statement(text)
        except Exception as exc:
            QMessageBox.warning(self, "Invalid Import", str(exc))
            return

        existing = [
            self._imp_list.item(i).text() for i in range(self._imp_list.count())
        ]
        if text not in existing:
            self._imp_list.addItem(text)
        self._imp_input.clear()
        self._refresh_status()

    def _on_remove_import(self) -> None:
        for item in self._imp_list.selectedItems():
            row = self._imp_list.row(item)
            self._imp_list.takeItem(row)
        self._refresh_status()

    def _on_accept(self) -> None:
        self.pythonpath, self.imports = self._collect_current()
        self.accept()
