from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeySequence, QResizeEvent
from PySide6.QtWidgets import (
    QApplication,
    QTableWidget,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from gui.gui_constants import Theme, get_plot_bg, get_theme


class TableWidget(QTableWidget):
    """Custom table widget with tooltip support and row selection."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.requestor = None

        self.setMouseTracking(True)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectItems)
        self.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.setSortingEnabled(True)

        self.verticalHeader().setDefaultSectionSize(32)
        self.horizontalHeader().setFixedHeight(32)
        self.horizontalHeader().setMouseTracking(True)
        self.horizontalHeader().viewport().setMouseTracking(True)
        self.horizontalHeader().viewport().installEventFilter(self)

    def mouseMoveEvent(self, event) -> None:
        """Show tooltip with cell content when hovering over cells."""
        item = self.itemAt(event.position().toPoint())

        if item is not None:
            cell_text = item.text()
            if cell_text:
                QToolTip.showText(event.globalPosition().toPoint(), cell_text, self)
            else:
                QToolTip.hideText()
        else:
            QToolTip.hideText()

        super().mouseMoveEvent(event)

    def keyPressEvent(self, event) -> None:
        """Handle key press events — Ctrl+C to copy selected range to clipboard."""
        if event.matches(QKeySequence.StandardKey.Copy):
            self.copy_selection_to_clipboard()
        else:
            super().keyPressEvent(event)

    def eventFilter(self, obj, event) -> bool:
        """Show header tooltip with the hovered column name."""
        if obj is self.horizontalHeader().viewport():
            if event.type() == QEvent.Type.MouseMove:
                col = self.horizontalHeader().logicalIndexAt(event.pos())
                if col >= 0:
                    header_item = self.horizontalHeaderItem(col)
                    header_text = header_item.text() if header_item else ""
                    if header_text:
                        QToolTip.showText(
                            self.horizontalHeader().viewport().mapToGlobal(event.pos()),
                            header_text,
                            self,
                        )
                    else:
                        QToolTip.hideText()
                else:
                    QToolTip.hideText()
                return False
            if event.type() == QEvent.Type.Leave:
                QToolTip.hideText()
                return False
        return super().eventFilter(obj, event)

    def copy_selection_to_clipboard(self) -> None:
        """Copy selected cells to clipboard in TSV format."""
        selected_ranges = self.selectedRanges()
        if not selected_ranges:
            return

        min_row = min(r.topRow() for r in selected_ranges)
        max_row = max(r.bottomRow() for r in selected_ranges)
        min_col = min(r.leftColumn() for r in selected_ranges)
        max_col = max(r.rightColumn() for r in selected_ranges)

        selected_cells = set()
        for r in selected_ranges:
            for row in range(r.topRow(), r.bottomRow() + 1):
                for col in range(r.leftColumn(), r.rightColumn() + 1):
                    selected_cells.add((row, col))

        rows_text = []
        for row in range(min_row, max_row + 1):
            row_data = []
            for col in range(min_col, max_col + 1):
                if (row, col) in selected_cells:
                    item = self.item(row, col)
                    cell_text = item.text() if item else ""
                else:
                    cell_text = ""
                row_data.append(cell_text)
            rows_text.append("\t".join(row_data))

        clipboard_text = "\n".join(rows_text)
        clipboard = QApplication.clipboard()
        clipboard.setText(clipboard_text)


def _is_dark_plot_theme(plot_theme_str: str, ui_theme: Theme) -> bool:
    key = (plot_theme_str or "538").lower()
    if key in ("dark", "d"):
        return True
    if key in ("light", "1", "bw", "b", "classic", "c", "minimal", "min", "linedraw", "ld", "tufte", "t"):
        return False
    # Check UI theme luminance as fallback
    bg_hex = ui_theme.bg.lstrip("#")
    if len(bg_hex) == 6:
        r, g, b = int(bg_hex[0:2], 16), int(bg_hex[2:4], 16), int(bg_hex[4:6], 16)
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        return lum < 128
    return True


def _regime_palette(plot_theme_str: str, ui_theme: str | Theme | None = None) -> list[tuple[str, float]]:
    """Return (color_hex, alpha) pairs for regime background shading appropriate to the plot theme."""
    theme_obj = get_theme(ui_theme)
    dark = _is_dark_plot_theme(plot_theme_str, theme_obj)
    alpha = 0.22 if dark else 0.18
    accents = [pair[0] for pair in theme_obj.accents]
    fallback = [
        "#0078D4",
        "#107C10",
        "#D83B01",
        "#C239B3",
        "#FFB900",
        "#0E8A8A",
    ]
    colors = accents + [c for c in fallback if c not in accents]
    return [(c, alpha) for c in colors]


class PlotWidget(QWidget):
    """Custom plot widget with line plot, tooltips, regime shading, and responsive axes."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.figure = Figure(figsize=(8, 6))
        self.figure.patch.set_facecolor(get_plot_bg())
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(get_plot_bg())

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self.lines: list = []
        self.annotation = None
        self._annotation_ax = None

        self.highlight_markers: list = []
        self.highlighted_index: int | None = None

        self._x_slider_enabled = False
        self._x_slider_dragging = False
        self._x_slider_values: list[float] = []
        self._x_slider_index: int | None = None
        self._x_slider_line = None
        self._x_slider_handle = None
        self._x_slider_handle_text = None
        self._x_slider_date_callback = None

        self.requestor = None
        self._external_figure = False
        self._flip_coord = False

        self._has_plot = False
        self._axis_tick_cache: dict[int, dict[str, dict]] = {}
        self._last_responsive_size: tuple[int, int] | None = None
        self._x_numeric_to_display: dict[float, str] = {}
        self._axis_x_lookups: dict[int, dict[float, str]] = {}
        self._regime_artists: list = []

        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_motion)
        self.canvas.mpl_connect("button_press_event", self.on_mouse_click)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.canvas.mpl_connect("resize_event", self._on_resize)

        self.canvas.hide()

    @property
    def has_plot(self) -> bool:
        """Return True if a figure is currently displayed."""
        return self._has_plot

    def clear_plot(self) -> None:
        """Hide the plot area until data is rendered."""
        self._has_plot = False
        self._external_figure = False
        self._flip_coord = False
        self._axis_tick_cache = {}
        self._last_responsive_size = None
        self._x_numeric_to_display = {}
        self._axis_x_lookups = {}
        self._regime_artists = []
        self.lines = []
        self.highlight_markers = []
        self.highlighted_index = None
        self.annotation = None
        self._annotation_ax = None
        self.canvas.hide()

    def _main_axes(self) -> list:
        if self.figure is None:
            return []
        return [ax for ax in self.figure.get_axes() if self._is_main_axes(ax)]

    def apply_regime_shading(
        self,
        regime_x: list[Any],
        regime_values: list[Any],
        *,
        plot_theme_str: str = "538",
        ui_theme: str | Theme | None = None,
        flip_coord: bool = False,
    ) -> None:
        """Draw background shading spans across all main axes according to regime values."""
        import matplotlib.dates as mdates

        from gui.plotnine_wrapper import normalize_x_value

        for artist in self._regime_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._regime_artists = []

        if not regime_x or not regime_values or len(regime_x) != len(regime_values):
            return

        main_axes = self._main_axes()
        if not main_axes:
            return

        def _to_x_coord(val: Any, idx: int) -> float | None:
            norm = normalize_x_value(val)
            if isinstance(norm, (datetime, date)):
                return float(mdates.date2num(norm))
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                if math.isnan(float(val)):
                    return None
                return float(val)
            return float(idx + 1)

        def _normalize_regime(v: Any) -> float | str | None:
            if v is None:
                return None
            if isinstance(v, bool):
                return 1.0 if v else 0.0
            if isinstance(v, (int, float)):
                if math.isnan(float(v)) or math.isinf(float(v)):
                    return None
                return float(v)
            s = str(v).strip()
            if not s or s.lower() in ("nan", "none", "null"):
                return None
            try:
                f = float(s)
                return None if math.isnan(f) else f
            except ValueError:
                return s

        points: list[tuple[float, float | str | None]] = []
        for idx, (rx, rv) in enumerate(zip(regime_x, regime_values)):
            coord = _to_x_coord(rx, idx)
            if coord is None:
                continue
            points.append((coord, _normalize_regime(rv)))

        if not points:
            return

        points.sort(key=lambda item: item[0])
        xs = [p[0] for p in points]
        vals = [p[1] for p in points]
        non_null_vals = [v for v in vals if v is not None]
        if not non_null_vals:
            return

        palette = _regime_palette(plot_theme_str, ui_theme)
        unique_non_null = sorted({str(v): v for v in non_null_vals}.values(), key=lambda x: str(x))

        # Check if numeric and continuous (> 8 distinct values) vs discrete/binary
        all_numeric = all(isinstance(v, (int, float)) for v in non_null_vals)
        is_binary_like = (
            all_numeric
            and len(unique_non_null) <= 2
            and set(float(v) for v in unique_non_null).issubset({0.0, 1.0, -1.0})
        )

        # Build contiguous intervals [start_x, end_x, val]
        n = len(xs)
        boundaries: list[float] = []
        if n == 1:
            boundaries = [xs[0] - 0.5, xs[0] + 0.5]
        else:
            first_step = (xs[1] - xs[0]) / 2.0
            boundaries.append(xs[0] - first_step)
            for i in range(n - 1):
                boundaries.append((xs[i] + xs[i + 1]) / 2.0)
            last_step = (xs[-1] - xs[-2]) / 2.0
            boundaries.append(xs[-1] + last_step)

        # Merge adjacent intervals with identical regime value
        merged_spans: list[tuple[float, float, float | str | None]] = []
        for i, val in enumerate(vals):
            x0, x1 = boundaries[i], boundaries[i + 1]
            if merged_spans and merged_spans[-1][2] == val:
                prev_x0, _, prev_val = merged_spans[-1]
                merged_spans[-1] = (prev_x0, x1, prev_val)
            else:
                merged_spans.append((x0, x1, val))

        # Map values to (color, alpha)
        val_min = min(float(v) for v in non_null_vals) if all_numeric else 0.0
        val_max = max(float(v) for v in non_null_vals) if all_numeric else 1.0

        discrete_color_map: dict[Any, tuple[str, float]] = {}
        if not all_numeric or len(unique_non_null) <= 8:
            # Discrete regimes: if 0.0 is present alongside positive/negative regimes,
            # leave 0.0 unshaded (neutral regime) and shade active regimes!
            has_zero_and_nonzero = (
                all_numeric
                and len(unique_non_null) > 1
                and any(float(v) == 0.0 for v in unique_non_null)
            )
            color_idx = 0
            for uv in unique_non_null:
                if has_zero_and_nonzero and float(uv) == 0.0:
                    continue
                discrete_color_map[uv] = palette[color_idx % len(palette)]
                color_idx += 1

        for ax in main_axes:
            span_fn = ax.axhspan if flip_coord else ax.axvspan
            for x0, x1, val in merged_spans:
                if val is None:
                    continue
                if discrete_color_map:
                    if val not in discrete_color_map:
                        continue
                    color, alpha = discrete_color_map[val]
                elif is_binary_like:
                    if float(val) == 0.0:
                        continue
                    color, alpha = palette[0] if float(val) > 0 else palette[2 % len(palette)]
                else:
                    # Continuous numeric regime value: vary alpha ordiverging color
                    fval = float(val)
                    if val_max > val_min:
                        norm_v = (fval - val_min) / (val_max - val_min)
                    else:
                        norm_v = 0.5
                    base_color, base_alpha = palette[0]
                    color = base_color
                    alpha = max(0.05, min(0.38, base_alpha * (0.35 + 1.15 * norm_v)))

                patch = span_fn(
                    x0,
                    x1,
                    facecolor=color,
                    alpha=alpha,
                    edgecolor="none",
                    zorder=0.5,
                )
                self._regime_artists.append(patch)

        self.canvas.draw_idle()

    def _facet_axis_label(self, ax, ax_idx: int, series_names: list[str]) -> str:
        title = (ax.get_title() or "").strip()
        if title:
            for prefix in ("Series:", "series:"):
                if title.lower().startswith(prefix.lower()):
                    return title[len(prefix) :].strip()
            return title
        sorted_names = (
            sorted(series_names) if len(series_names) > 1 else list(series_names)
        )
        if ax_idx < len(sorted_names):
            return sorted_names[ax_idx]
        return f"series_{ax_idx}"

    def _build_color_to_label(self) -> dict:
        from matplotlib.collections import PathCollection
        from matplotlib.colors import to_rgba
        from matplotlib.lines import Line2D

        def _color_key(color):
            return tuple(round(v, 4) for v in to_rgba(color))

        color_to_label = {}
        all_legends = []
        if self.ax is not None:
            ax_legend = self.ax.get_legend()
            if ax_legend is not None:
                all_legends.append(ax_legend)
        if hasattr(self.figure, "legends"):
            all_legends.extend(self.figure.legends)

        for leg in all_legends:
            handles = getattr(leg, "legend_handles", getattr(leg, "legendHandles", []))
            for handle, text in zip(handles, leg.get_texts()):
                try:
                    if isinstance(handle, Line2D):
                        key = _color_key(handle.get_color())
                    elif isinstance(handle, PathCollection):
                        fc = handle.get_facecolors()
                        key = _color_key(fc[0]) if len(fc) > 0 else None
                    else:
                        key = None
                    if key is not None:
                        color_to_label[key] = text.get_text()
                except Exception:
                    pass
        return color_to_label

    def _register_axis_x_values(self, ax, x_data) -> None:
        """Map x coordinates within a facet panel to display labels."""
        import matplotlib.dates as mdates

        from gui.plotnine_wrapper import format_display_date, normalize_x_value

        lookup = self._axis_x_lookups.setdefault(id(ax), {})
        for idx, x in enumerate(x_data):
            normalized = normalize_x_value(x)
            if isinstance(normalized, (datetime, date)):
                label = format_display_date(normalized)
            else:
                label = format_display_date(x)

            if not label:
                label = str(x)

            lookup[float(idx)] = label
            try:
                if isinstance(normalized, (datetime, date)):
                    x_num = float(mdates.date2num(normalized))
                elif isinstance(x, (datetime, date)):
                    x_num = float(mdates.date2num(x))
                elif hasattr(x, "to_pydatetime"):
                    x_num = float(mdates.date2num(x.to_pydatetime()))
                else:
                    x_num = float(x)
                lookup[x_num] = label
                lookup[round(x_num, 6)] = label
            except Exception:
                pass

    def _extract_tooltip_lines(self, series_names: list[str] | None) -> None:
        """Extract line/point artists from every facet panel for tooltips."""
        from matplotlib.collections import PathCollection
        from matplotlib.colors import to_rgba
        from matplotlib.lines import Line2D

        self.lines = []
        self._axis_x_lookups = {}
        main_axes = self._main_axes()
        if not main_axes:
            return

        names = series_names or []
        color_to_label = self._build_color_to_label()
        sorted_names = sorted(names) if len(names) > 1 else list(names)
        n = len(sorted_names)
        use_facet_labels = len(main_axes) > 1

        def _color_key(color):
            return tuple(round(v, 4) for v in to_rgba(color))

        def _append_line(artist, xd, yd, lbl, ax):
            self._register_axis_x_values(ax, xd)
            if use_facet_labels:
                lbl = self._facet_axis_label(ax, main_axes.index(ax), names)
            self.lines.append((artist, xd, yd, lbl, ax))

        for ax_idx, ax in enumerate(main_axes):
            facet_label = self._facet_axis_label(ax, ax_idx, names)
            ax_lines = []

            if color_to_label and not use_facet_labels:
                seen = set()
                for artist in ax.get_children():
                    lbl = xd = yd = None
                    if isinstance(artist, Line2D):
                        xd = list(artist.get_xdata())
                        yd = list(artist.get_ydata())
                        if len(xd) <= 1:
                            continue
                        try:
                            lbl = color_to_label.get(_color_key(artist.get_color()))
                        except Exception:
                            pass
                    elif isinstance(artist, PathCollection):
                        offsets = artist.get_offsets()
                        if len(offsets) == 0:
                            continue
                        xd = list(offsets[:, 0])
                        yd = list(offsets[:, 1])
                        try:
                            fc = artist.get_facecolors()
                            if len(fc) > 0:
                                lbl = color_to_label.get(_color_key(fc[0]))
                        except Exception:
                            pass
                    if lbl and lbl not in seen:
                        seen.add(lbl)
                        ax_lines.append((artist, xd, yd, lbl, ax))

            if not ax_lines:
                line_artists = []
                point_artists = []
                for artist in ax.get_children():
                    if isinstance(artist, Line2D):
                        xd = list(artist.get_xdata())
                        yd = list(artist.get_ydata())
                        if len(xd) > 2:
                            line_artists.append((artist, xd, yd))
                    elif isinstance(artist, PathCollection):
                        offsets = artist.get_offsets()
                        if len(offsets) > 0:
                            xd = list(offsets[:, 0])
                            yd = list(offsets[:, 1])
                            point_artists.append((artist, xd, yd))

                for i, (artist, xd, yd) in enumerate(
                    line_artists[:n] if n else line_artists
                ):
                    lbl = (
                        facet_label
                        if use_facet_labels
                        else (
                            sorted_names[i]
                            if i < n
                            else (artist.get_label() or f"series_{i}")
                        )
                    )
                    ax_lines.append((artist, xd, yd, lbl, ax))
                for i, (artist, xd, yd) in enumerate(
                    point_artists[:n] if n else point_artists
                ):
                    lbl = (
                        facet_label
                        if use_facet_labels
                        else (
                            sorted_names[i]
                            if i < n
                            else (artist.get_label() or f"series_{i}")
                        )
                    )
                    ax_lines.append((artist, xd, yd, lbl, ax))

            for artist, xd, yd, lbl, line_ax in ax_lines:
                _append_line(artist, xd, yd, lbl, line_ax)

    def _hide_tooltip(self) -> None:
        if self.annotation is not None:
            self.annotation.set_visible(False)
            self.canvas.draw_idle()

    def _show_tooltip(
        self, ax, x_num: float, y_val: float, x_orig: Any, label: str
    ) -> None:
        point_px = ax.transData.transform([x_num, y_val])
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        cx, cy = (xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2
        center_px = ax.transData.transform([cx, cy])
        dx = center_px[0] - point_px[0]
        dy = center_px[1] - point_px[1]
        length = np.sqrt(dx**2 + dy**2)
        arrow_len = 40.0
        if length > 0:
            offset = [dx / length * arrow_len, dy / length * arrow_len]
        else:
            offset = [arrow_len, arrow_len]

        ha = "left" if offset[0] >= 0 else "right"
        va = "bottom" if offset[1] >= 0 else "top"

        if self.annotation is not None and self._annotation_ax is not ax:
            self.annotation.remove()
            self.annotation = None
            self._annotation_ax = None

        if self.annotation is None:
            self.annotation = ax.annotate(
                "",
                xy=(x_num, y_val),
                xytext=offset,
                textcoords="offset points",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    fc="#ffffcc",
                    ec="#999999",
                    alpha=0.95,
                ),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                fontsize=7,
                ha=ha,
                va=va,
            )
            self._annotation_ax = ax
        else:
            self.annotation.xyann = offset
            self.annotation.set_ha(ha)
            self.annotation.set_va(va)
            self.annotation.xy = (x_num, y_val)

        x_str = self._format_x_as_date(x_orig if x_orig is not None else x_num, ax=ax)
        y_str = f"{y_val:.4f}"
        self.annotation.set_text(f"{label}\n{x_str} | {y_str}")
        self.annotation.set_visible(True)
        self.canvas.draw_idle()

    def _layout_params_for_span(self, span_px: int) -> dict:
        """Map available axis span (px) to tick density, font sizes, and label angle."""
        span_px = max(int(span_px), 80)
        max_ticks = max(3, min(int(span_px / 65), 24))

        if span_px < 220:
            rotation, ha = 90, "center"
            tick_fs, label_fs = 6, 8
        elif span_px < 320:
            rotation, ha = 60, "right"
            tick_fs, label_fs = 7, 9
        elif span_px < 450:
            rotation, ha = 45, "right"
            tick_fs, label_fs = 8, 10
        elif span_px < 600:
            rotation, ha = 30, "right"
            tick_fs, label_fs = 9, 11
        else:
            rotation, ha = 0, "center"
            tick_fs, label_fs = 10, 12

        return {
            "max_ticks": max_ticks,
            "rotation": rotation,
            "ha": ha,
            "tick_fs": tick_fs,
            "label_fs": label_fs,
        }

    def _is_main_axes(self, ax) -> bool:
        pos = ax.get_position()
        return pos.width > 0.12 and pos.height > 0.12

    def _capture_axis_tick_cache(self) -> None:
        """Snapshot plotnine tick positions/formatters for responsive re-sampling."""
        self._axis_tick_cache = {}
        if self.figure is None:
            return

        self.canvas.draw()
        for ax in self.figure.get_axes():
            if not self._is_main_axes(ax):
                continue
            self._axis_tick_cache[id(ax)] = {
                "x": {
                    "ticks": list(ax.get_xticks()),
                    "labels": [label.get_text() for label in ax.get_xticklabels()],
                    "formatter": ax.xaxis.get_major_formatter(),
                },
                "y": {
                    "ticks": list(ax.get_yticks()),
                    "labels": [label.get_text() for label in ax.get_yticklabels()],
                    "formatter": ax.yaxis.get_major_formatter(),
                },
            }

    def _apply_axis_ticks_from_cache(
        self, ax, axis_name: str, max_ticks: int
    ) -> None:
        """Subsample or restore ticks from the original plotnine snapshot."""
        if max_ticks < 1:
            return

        cached = self._axis_tick_cache.get(id(ax), {}).get(axis_name)
        if not cached:
            return

        ticks = cached["ticks"]
        labels = cached["labels"]
        formatter = cached.get("formatter")
        if not ticks:
            return

        if len(labels) < len(ticks):
            labels = labels + [""] * (len(ticks) - len(labels))

        if len(ticks) <= max_ticks:
            selected_ticks = ticks
            selected_labels = labels
        else:
            indices = np.unique(
                np.round(np.linspace(0, len(ticks) - 1, max_ticks)).astype(int)
            )
            selected_ticks = [ticks[i] for i in indices]
            selected_labels = [labels[i] for i in indices]

        axis = ax.xaxis if axis_name == "x" else ax.yaxis
        set_ticks = ax.set_xticks if axis_name == "x" else ax.set_yticks
        set_ticklabels = ax.set_xticklabels if axis_name == "x" else ax.set_yticklabels

        set_ticks(selected_ticks)
        if formatter is not None:
            axis.set_major_formatter(formatter)
        elif any(selected_labels):
            set_ticklabels(selected_labels)

    def _tick_pad_for_rotation(self, rotation: float) -> float:
        if rotation >= 90:
            return 8.0
        if rotation >= 60:
            return 6.0
        if rotation >= 45:
            return 5.0
        if rotation >= 30:
            return 4.0
        return 2.0

    def _style_axis_tick_labels(
        self, ax, axis_name: str, params: dict, *, vertical: bool = False
    ) -> None:
        tick_labels = (
            ax.get_xticklabels() if axis_name == "x" else ax.get_yticklabels()
        )
        if not tick_labels:
            return
        rotation = 0 if vertical else params["rotation"]
        ha = "right" if vertical else params["ha"]
        for label in tick_labels:
            label.set_visible(True)
            label.set_rotation(rotation)
            label.set_ha(ha)
            label.set_fontsize(params["tick_fs"])

    def _apply_responsive_axes(
        self, width_px: int | None = None, height_px: int | None = None
    ) -> None:
        """Re-tune axis ticks and label sizes for the current canvas dimensions."""
        if self.figure is None or not self._has_plot:
            return

        width_px = width_px if width_px is not None else self.canvas.width()
        height_px = height_px if height_px is not None else self.canvas.height()
        if width_px <= 0 or height_px <= 0:
            return

        if not self._axis_tick_cache:
            self._capture_axis_tick_cache()

        main_axes = [ax for ax in self.figure.get_axes() if self._is_main_axes(ax)]
        if not main_axes:
            return

        x_params = self._layout_params_for_span(width_px)
        y_params = self._layout_params_for_span(height_px)

        for ax in main_axes:
            self._apply_axis_ticks_from_cache(ax, "x", x_params["max_ticks"])
            if self._flip_coord:
                self._apply_axis_ticks_from_cache(ax, "y", y_params["max_ticks"])
            ax.xaxis.set_visible(True)
            ax.spines["bottom"].set_visible(True)

        self.canvas.draw()

        for ax in main_axes:
            self._style_axis_tick_labels(ax, "x", x_params)
            if self._flip_coord:
                self._style_axis_tick_labels(ax, "y", y_params, vertical=True)
            else:
                for label in ax.get_yticklabels():
                    label.set_fontsize(x_params["tick_fs"])

            ax.xaxis.label.set_fontsize(x_params["label_fs"])
            ax.yaxis.label.set_fontsize(x_params["label_fs"])
            ax.title.set_fontsize(x_params["label_fs"] + 1)
            ax.tick_params(
                axis="x", pad=self._tick_pad_for_rotation(x_params["rotation"])
            )

        self._last_responsive_size = (width_px, height_px)
        self.canvas.draw_idle()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if not self._has_plot:
            return
        size = (self.canvas.width(), self.canvas.height())
        if size == self._last_responsive_size:
            return
        self._apply_responsive_axes(size[0], size[1])

    def _on_resize(self, event) -> None:
        if not self._has_plot:
            return
        width_px = int(event.width) if event is not None else self.canvas.width()
        height_px = int(event.height) if event is not None else self.canvas.height()
        if (width_px, height_px) == self._last_responsive_size:
            return
        self._apply_responsive_axes(width_px, height_px)

    def display_figure(
        self,
        fig: Figure,
        series_names: list[str] | None = None,
        flip_coord: bool = False,
        x_values: list | None = None,
    ) -> None:
        """Replace the current canvas with an external matplotlib Figure."""
        self._external_figure = True
        self._flip_coord = bool(flip_coord)
        self._regime_artists = []
        if x_values:
            from gui.plotnine_wrapper import build_x_numeric_lookup

            self._x_numeric_to_display = build_x_numeric_lookup(x_values)
        else:
            self._x_numeric_to_display = {}

        self.layout().removeWidget(self.canvas)
        self.canvas.setParent(None)
        self.canvas.close()

        self.figure = fig
        self.canvas = FigureCanvas(self.figure)
        self.layout().addWidget(self.canvas)

        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_motion)
        self.canvas.mpl_connect("button_press_event", self.on_mouse_click)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.canvas.mpl_connect("resize_event", self._on_resize)

        axes = self.figure.get_axes()
        self.ax = axes[0] if axes else None

        self.highlight_markers = []
        self.highlighted_index = None
        self.annotation = None
        self._annotation_ax = None

        self._extract_tooltip_lines(series_names if series_names else [])

        self._has_plot = True
        self._last_responsive_size = None
        self.canvas.show()
        QApplication.processEvents()
        self._capture_axis_tick_cache()
        self._apply_responsive_axes()

    def _format_x_as_date(self, x_value: Any, ax=None) -> str:
        """Format numeric/datetime x value as YYYY-mm-dd."""
        from gui.plotnine_wrapper import (
            _matplotlib_num_to_date_str,
            format_display_date,
            lookup_x_display,
            normalize_x_value,
        )

        try:
            normalized = normalize_x_value(x_value)
            if isinstance(normalized, (datetime, date)):
                return format_display_date(normalized)
            if hasattr(x_value, "to_pydatetime"):
                return format_display_date(x_value.to_pydatetime())

            x_num = float(x_value)

            mapped = lookup_x_display(self._x_numeric_to_display, x_num)
            if mapped is not None:
                return mapped

            if ax is not None:
                axis_lookup = self._axis_x_lookups.get(id(ax), {})
                mapped = lookup_x_display(axis_lookup, x_num)
                if mapped is not None:
                    return mapped

            as_date = _matplotlib_num_to_date_str(x_num)
            if as_date is not None:
                return as_date

            return format_display_date(x_value)
        except Exception:
            return str(x_value)

    def on_mouse_motion(self, event) -> None:
        """Show tooltip when mouse hovers over data points."""
        if event.inaxes is None or not self.lines:
            self._hide_tooltip()
            return

        hover_ax = event.inaxes
        import matplotlib.dates as mdates

        min_dist = float("inf")
        closest_point = None
        closest_ax = None

        def _to_x_num(x):
            if isinstance(x, (date, datetime)):
                return float(mdates.date2num(x))
            if hasattr(x, "to_pydatetime"):
                return float(mdates.date2num(x.to_pydatetime()))
            return float(x)

        for _artist, x_data, y_data, label, line_ax in self.lines:
            if line_ax is not hover_ax or len(x_data) == 0:
                continue
            try:
                for x, y in zip(x_data, y_data):
                    if y is None or (isinstance(y, float) and np.isnan(y)):
                        continue
                    x_num = _to_x_num(x)
                    point = line_ax.transData.transform([x_num, float(y)])
                    dist = np.sqrt(
                        (point[0] - event.x) ** 2 + (point[1] - event.y) ** 2
                    )
                    if dist < min_dist and dist < 15:
                        min_dist = dist
                        closest_point = (x_num, float(y), x, label)
                        closest_ax = line_ax
            except Exception:
                pass

        if closest_point and closest_ax is not None:
            x_num, y_val, x_orig, label = closest_point
            self._show_tooltip(closest_ax, x_num, y_val, x_orig, label)
        else:
            self._hide_tooltip()

    def on_mouse_click(self, event) -> None:
        """Handle mouse click on plot."""
        return

    def on_mouse_release(self, event) -> None:
        """Handle mouse release on plot."""
        if event.button == 1 and self._x_slider_dragging:
            self._x_slider_dragging = False
