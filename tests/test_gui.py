from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from gui.app import MAX_FORMULA_ROWS, PanelyticsWindow
from gui.data_table_dialog import DataTableDialog
from gui.gui_settings import SETTINGS_FILENAME, load_settings, save_settings
from gui.panelyzer_bridge import (
    build_autocomplete_tokens,
    create_source_from_clipboard,
    evaluate_formula_rows,
    load_user_eval_globals,
)


def _ensure_qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app  # type: ignore[return-value]


def test_settings_persistence_json(tmp_path: Path) -> None:
    """Verify panelytics.json persistence round-trip."""
    json_path = tmp_path / SETTINGS_FILENAME
    custom = {
        "ui_theme": "Nord",
        "plot_settings": {"geoms": "L", "theme_str": "dark"},
        "download_settings": {"format": "parquet", "headers": False},
        "data_sources": [{"name": "s1", "source_type": "csv", "path": "a.csv"}],
        "formula_rows": [{"is_regime": True, "formula": "{USA:rate}"}],
        "user_functions": {"pythonpath": ["/tmp"], "imports": ["import math"]},
    }
    save_settings(custom, path=json_path)
    assert json_path.is_file()

    loaded = load_settings(path=json_path)
    assert loaded["ui_theme"] == "Nord"
    assert loaded["plot_settings"]["geoms"] == "L"
    assert loaded["plot_settings"]["theme_str"] == "dark"
    assert loaded["download_settings"]["format"] == "parquet"
    assert loaded["formula_rows"][0]["is_regime"] is True
    assert loaded["user_functions"]["imports"] == ["import math"]


def test_clipboard_source_and_formula_evaluation_with_regime() -> None:
    """Verify clipboard dataset parsing, EntityConfiguration estimation, user functions, and regime evaluation."""
    tsv = (
        "date\tsource\trate\tvol\n"
        "2025-01-01\tUSA\t4.2\t0.1\n"
        "2025-01-02\tUSA\t4.5\t0.2\n"
        "2025-01-03\tUSA\t4.8\t0.3\n"
        "2025-01-01\tDEU\t2.1\t0.1\n"
        "2025-01-02\tDEU\t2.3\t0.2\n"
        "2025-01-03\tDEU\t2.2\t0.3\n"
    )
    src = create_source_from_clipboard(tsv)
    assert src.date_col == "date"
    assert src.entity_col == "source"
    assert src.default_feature == "rate"
    assert set(src.entities) == {"USA", "DEU"}
    assert "rate" in src.features and "vol" in src.features

    eval_globals, errs = load_user_eval_globals([], ["import math as mymath"])
    assert not errs
    assert "mymath" in eval_globals

    tokens = build_autocomplete_tokens([src], eval_globals)
    assert "{USA}" in tokens
    assert "{USA:rate}" in tokens
    assert "{DEU:vol}" in tokens
    assert any(t.startswith("mymath.") for t in tokens)

    rows = [
        {"is_regime": False, "formula": "{USA:rate} - {DEU:rate}"},
        {"is_regime": True, "formula": "({USA:rate} > 4.4).astype(float)"},
    ]
    df, regime_col, warnings_list = evaluate_formula_rows([src], rows, eval_globals)
    assert not warnings_list
    assert df is not None
    assert df.height == 3
    assert regime_col is not None
    assert regime_col in df.columns


def test_gui_window_sections_and_constraints() -> None:
    """Verify the 4 GUI sections, square gears button, single-checkbox regime rule, and 10-row limit."""
    _ensure_qapp()
    win = PanelyticsWindow()

    # Section (1): Square gears button on far right + Add/Remove Functions to its left
    assert win._settings_btn.text() == "⚙"
    assert win._settings_btn.width() == win._settings_btn.height()
    assert win._functions_btn.text() == "Add/Remove Functions"

    # Section (2): Collapsible Data Sources pane
    assert win._sources_pane.is_expanded is True
    win._sources_pane.toggle()
    assert win._sources_pane.is_expanded is False
    win._sources_pane.toggle()
    assert win._sources_pane.is_expanded is True

    # Section (3): Fixed 2 columns, max 10 rows, at most 1 checkbox ticked
    assert win._formula_table.columnCount() == 2
    tsv = (
        "date\tsource\trate\n"
        "2025-01-01\tUSA\t4.2\n"
        "2025-01-02\tUSA\t4.5\n"
        "2025-01-03\tUSA\t4.8\n"
        "2025-01-01\tDEU\t2.1\n"
        "2025-01-02\tDEU\t2.3\n"
        "2025-01-03\tDEU\t2.2\n"
    )
    win._sources = [create_source_from_clipboard(tsv)]
    win._load_formula_rows_into_table(
        [
            {"is_regime": False, "formula": "{USA:rate}"},
            {"is_regime": False, "formula": "{DEU:rate}"},
            {"is_regime": False, "formula": "({USA:rate} > 4.3).astype(float)"},
        ]
    )
    assert win._formula_table.rowCount() == 3

    cb0 = win._get_row_checkbox(0)
    cb1 = win._get_row_checkbox(1)
    cb2 = win._get_row_checkbox(2)
    assert cb0 is not None and cb1 is not None and cb2 is not None

    cb0.setChecked(True)
    assert cb0.isChecked() is True
    assert cb1.isChecked() is False

    cb2.setChecked(True)
    assert cb0.isChecked() is False
    assert cb2.isChecked() is True
    assert cb2.objectName() == "regimeCheckBox"
    # Section (4): Plot rendered, regime shading spans applied, and eye button enabled
    assert win._plot.has_plot is True
    assert len(win._plot._regime_artists) > 0
    assert win._view_data_btn.isEnabled() is True
    assert win._date_range_slider.isEnabled() is True
    assert win._date_range_slider.selected_indices == (0, 2)
    win._date_range_slider.set_range(1, 2, emit=True)
    disp = win._displayed_dataframe()
    assert disp is not None and disp.height == 2
    assert win._last_dataframe is not None
    dlg = DataTableDialog(disp, title="Visualization", theme=win._theme)
    assert dlg.proxy.rowCount() == 2
    assert dlg.proxy.columnCount() == win._last_dataframe.width
    dlg.close()

    # Test maximum 10 rows
    for _ in range(15):
        win._add_formula_row()
    assert win._formula_table.rowCount() == MAX_FORMULA_ROWS
    assert win._add_row_btn.isEnabled() is False

    # Deleting a data source removes formula rows referencing that source
    tsv_fx = "date\tpair\tspot\n2025-01-01\tEURUSD\t1.08\n2025-01-02\tEURUSD\t1.09\n"
    win._sources.append(create_source_from_clipboard(tsv_fx, {"clipboard_data"}))
    win._load_formula_rows_into_table(
        [
            {"is_regime": False, "formula": "{USA:rate}"},
            {"is_regime": False, "formula": "{EURUSD:spot}"},
            {"is_regime": False, "formula": "{DEU} + {EURUSD}"},
        ]
    )
    win._on_delete_source(1)  # delete FX source (EURUSD)
    remaining = win._collect_formula_rows()
    assert remaining == [{"is_regime": False, "formula": "{USA:rate}"}]

    # Invalid formula highlights the row and blocks plotting/adding rows until fixed or deleted
    win._add_formula_row()
    win._formula_table.item(1, 1).setText("{UNKNOWN_ENTITY:rate}")
    assert 1 in win._row_errors
    assert win._plot.has_plot is False
    assert win._add_row_btn.isEnabled() is False
    assert "Formula Error" in win._formula_table.item(1, 1).toolTip()

    # Deleting the broken row clears the error and re-renders valid rows
    win._formula_table.selectRow(1)
    win._delete_selected_formula_row()
    assert not win._row_errors
    assert win._plot.has_plot is True
    assert win._add_row_btn.isEnabled() is True

    win.close()
