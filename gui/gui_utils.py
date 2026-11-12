from PySide6.QtCore import (
    QEasingCurve,
    QEvent,
    QObject,
    QPoint,
    QPointF,
    QPropertyAnimation,
    QRect,
    QRectF,
    Qt,
    Signal,
)
from PySide6.QtGui import QBrush, QColor, QCursor, QPainter, QPainterPath, QPen, QRegion
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class FramelessResizeHelper:
    """Edge/corner resize support for frameless top-level windows."""

    MARGIN = 8
    MIN_WIDTH = 720
    MIN_HEIGHT = 500

    def __init__(self, window: QWidget) -> None:
        self._window = window
        self._resize_edge: str | None = None
        self._resize_start_global: QPoint | None = None
        self._resize_start_geom: QRect | None = None
        window.setMouseTracking(True)
        window.setMinimumSize(self.MIN_WIDTH, self.MIN_HEIGHT)

    def install(self) -> None:
        """Install the application-wide mouse event filter."""
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self._window)

    def uninstall(self) -> None:
        """Detach the app-wide filter so a closed window stops seeing every event."""
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self._window)

    def hit_test(self, pos: QPoint) -> str | None:
        """Determine which resize edge or corner the cursor is over."""
        x, y = pos.x(), pos.y()
        w, h = self._window.width(), self._window.height()
        m = self.MARGIN
        on_left = x <= m
        on_right = x >= w - m
        on_top = y <= m
        on_bottom = y >= h - m

        if on_bottom and on_right:
            return "bottom-right"
        if on_bottom and on_left:
            return "bottom-left"
        if on_top and on_right:
            return "top-right"
        if on_top and on_left:
            return "top-left"
        if on_left:
            return "left"
        if on_right:
            return "right"
        if on_top:
            return "top"
        if on_bottom:
            return "bottom"
        return None

    def update_cursor(self, edge: str | None) -> None:
        """Update the window cursor shape based on the hovered edge."""
        cursors = {
            "left": Qt.CursorShape.SizeHorCursor,
            "right": Qt.CursorShape.SizeHorCursor,
            "top": Qt.CursorShape.SizeVerCursor,
            "bottom": Qt.CursorShape.SizeVerCursor,
            "top-left": Qt.CursorShape.SizeFDiagCursor,
            "bottom-right": Qt.CursorShape.SizeFDiagCursor,
            "top-right": Qt.CursorShape.SizeBDiagCursor,
            "bottom-left": Qt.CursorShape.SizeBDiagCursor,
        }
        if edge is None:
            self._window.unsetCursor()
        else:
            self._window.setCursor(QCursor(cursors[edge]))

    def begin_resize(self, edge: str, global_pos: QPoint) -> None:
        """Begin interactive resizing from the given edge."""
        self._resize_edge = edge
        self._resize_start_global = global_pos
        self._resize_start_geom = self._window.geometry()

    def end_resize(self) -> None:
        """Finish interactive resizing."""
        self._resize_edge = None
        self._resize_start_global = None
        self._resize_start_geom = None

    def apply_resize(self, global_pos: QPoint) -> None:
        """Resize the window geometry according to current mouse position."""
        if (
            self._resize_edge is None
            or self._resize_start_global is None
            or self._resize_start_geom is None
        ):
            return

        delta = global_pos - self._resize_start_global
        geom = QRect(self._resize_start_geom)
        min_w, min_h = self.MIN_WIDTH, self.MIN_HEIGHT
        edge = self._resize_edge

        if "left" in edge:
            geom.setLeft(min(geom.left() + delta.x(), geom.right() - min_w))
        if "right" in edge:
            geom.setRight(max(geom.right() + delta.x(), geom.left() + min_w))
        if "top" in edge:
            geom.setTop(min(geom.top() + delta.y(), geom.bottom() - min_h))
        if "bottom" in edge:
            geom.setBottom(max(geom.bottom() + delta.y(), geom.top() + min_h))

        self._window.setGeometry(geom)

    def handle_event_filter(self, watched: QObject, event: QEvent) -> bool:
        """Process mouse events for frameless resizing."""
        if event.type() not in (
            QEvent.Type.MouseMove,
            QEvent.Type.MouseButtonPress,
            QEvent.Type.MouseButtonRelease,
        ):
            return False

        if not self._window.isVisible():
            return False

        from PySide6.QtWidgets import QApplication

        modal = QApplication.activeModalWidget()
        if modal is not None and modal is not self._window:
            return False

        global_pos = event.globalPosition().toPoint()
        local_pos = self._window.mapFromGlobal(global_pos)
        if not self._window.rect().contains(local_pos):
            if self._resize_edge is None:
                return False

        if event.type() == QEvent.Type.MouseButtonPress:
            if event.button() != Qt.MouseButton.LeftButton:
                return False
            edge = self.hit_test(local_pos)
            if edge is None:
                return False
            self.begin_resize(edge, global_pos)
            return True

        if event.type() == QEvent.Type.MouseMove:
            if self._resize_edge is not None:
                self.apply_resize(global_pos)
                return True
            self.update_cursor(self.hit_test(local_pos))
            return False

        if event.type() == QEvent.Type.MouseButtonRelease:
            if (
                event.button() != Qt.MouseButton.LeftButton
                or self._resize_edge is None
            ):
                return False
            self.end_resize()
            self.update_cursor(self.hit_test(local_pos))
            return True

        return False


class RoundedShell(QWidget):
    """Frameless window shell with clipped rounded corners and drop shadow."""

    def __init__(self, parent: QWidget | None = None, radius: int = 12) -> None:
        super().__init__(parent)
        self.setObjectName("dialogShell")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._radius = radius

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(40)
        shadow.setOffset(0, 10)
        shadow.setColor(QColor(0, 0, 0, 90))
        self.setGraphicsEffect(shadow)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), self._radius, self._radius)
        self.setMask(QRegion(path.toFillPolygon().toPolygon()))


class PanelFrame(QFrame):
    """Elevated content panel with optional section header and header trailing widget."""

    def __init__(
        self,
        title: str = "",
        parent: QWidget | None = None,
        header_trailing: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("panelFrame")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(14, 12, 14, 14)
        self._layout.setSpacing(10)

        self._title_label: QLabel | None = None
        if title or header_trailing is not None:
            header_row = QHBoxLayout()
            header_row.setContentsMargins(0, 0, 0, 0)
            header_row.setSpacing(8)
            if title:
                self._title_label = QLabel(title)
                self._title_label.setObjectName("sectionLabel")
                header_row.addWidget(self._title_label)
            header_row.addStretch()
            if header_trailing is not None:
                header_row.addWidget(header_trailing)
            self._layout.addLayout(header_row)

    def set_content(self, widget: QWidget) -> None:
        """Set the main expanding child widget inside the panel."""
        self._layout.addWidget(widget, stretch=1)


class TitleBar(QWidget):
    """Draggable custom title bar for frameless windows."""

    def __init__(
        self,
        window: QWidget,
        title: str,
        parent: QWidget | None = None,
        trailing: QWidget | None = None,
        subtitle: str = "",
        minimizable: bool = False,
    ) -> None:
        super().__init__(parent)
        self._window = window
        self._drag_pos: QPoint | None = None
        self.setObjectName("titleBar")
        self.setFixedHeight(52)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 12, 0)
        layout.setSpacing(12)

        accent = QFrame()
        accent.setObjectName("titleAccent")
        accent.setFixedSize(4, 22)
        layout.addWidget(accent)

        title_col = QVBoxLayout()
        title_col.setContentsMargins(0, 0, 0, 0)
        title_col.setSpacing(0)

        label = QLabel(title)
        label.setObjectName("titleLabel")
        title_col.addWidget(label)

        if subtitle:
            sub = QLabel(subtitle)
            sub.setObjectName("titleSubtitle")
            title_col.addWidget(sub)

        title_widget = QWidget()
        title_widget.setLayout(title_col)
        layout.addWidget(title_widget)
        layout.addStretch()

        if trailing is not None:
            layout.addWidget(trailing)

        if minimizable:
            min_btn = QPushButton("\u2013")
            min_btn.setObjectName("minBtn")
            min_btn.setFixedSize(34, 34)
            min_btn.setToolTip("Minimize")
            min_btn.clicked.connect(window.showMinimized)
            layout.addWidget(min_btn)

        close_btn = QPushButton("\u2715")
        close_btn.setObjectName("closeBtn")
        close_btn.setFixedSize(34, 34)
        close_btn.setToolTip("Close")
        close_btn.clicked.connect(window.close)
        layout.addWidget(close_btn)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = (
                event.globalPosition().toPoint()
                - self._window.frameGeometry().topLeft()
            )
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if (
            self._drag_pos is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            self._window.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_pos = None
        super().mouseReleaseEvent(event)


class SmoothScroller(QObject):
    """Animate mouse-wheel scrolling on a scroll area's vertical scrollbar."""

    def __init__(
        self,
        area: QAbstractScrollArea,
        step: int = 90,
        duration_ms: int = 220,
    ) -> None:
        super().__init__(area)
        self._bar = area.verticalScrollBar()
        self._step = step
        self._target = self._bar.value()
        self._anim = QPropertyAnimation(self._bar, b"value", self)
        self._anim.setDuration(duration_ms)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        area.viewport().installEventFilter(self)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() != QEvent.Type.Wheel or not event.pixelDelta().isNull():
            return False
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            return False
        notches = event.angleDelta().y() / 120.0
        if not notches:
            return False
        start = (
            self._target
            if self._anim.state() == QPropertyAnimation.State.Running
            else self._bar.value()
        )
        self._target = max(
            self._bar.minimum(),
            min(self._bar.maximum(), round(start - notches * self._step)),
        )
        self._anim.stop()
        self._anim.setStartValue(self._bar.value())
        self._anim.setEndValue(self._target)
        self._anim.start()
        return True


class _DualRangeTrack(QWidget):
    """Custom dual-handle horizontal slider track."""

    range_changed = Signal(int, int)

    HANDLE_RADIUS = 7
    TRACK_HEIGHT = 4

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._max_index = 0
        self._low = 0
        self._high = 0
        self._dragging: str | None = None  # "low" | "high" | None
        self._hovered: str | None = None
        self._groove_color = QColor("#3C3C3C")
        self._fill_color = QColor("#0078D4")
        self._hover_color = QColor("#60BDFF")
        self._handle_color = QColor("#F0F0F0")
        self.setMouseTracking(True)
        self.setFixedHeight(24)
        self.setMinimumWidth(180)

    def set_theme_colors(
        self,
        groove: str,
        fill: str,
        hover: str,
        handle: str,
    ) -> None:
        """Update track and handle colors to match the active theme."""
        self._groove_color = QColor(groove)
        self._fill_color = QColor(fill)
        self._hover_color = QColor(hover)
        self._handle_color = QColor(handle)
        self.update()

    def set_bounds(self, max_index: int) -> None:
        """Reset slider bounds to [0, max_index] with full range selected."""
        self._max_index = max(0, int(max_index))
        self._low = 0
        self._high = self._max_index
        self.update()

    def set_indices(self, low: int, high: int, *, emit: bool = False) -> None:
        """Set low and high indices within [0, max_index]."""
        lo = max(0, min(int(low), self._max_index))
        hi = max(lo, min(int(high), self._max_index))
        changed = (lo != self._low) or (hi != self._high)
        self._low, self._high = lo, hi
        self.update()
        if changed and emit:
            self.range_changed.emit(self._low, self._high)

    @property
    def indices(self) -> tuple[int, int]:
        return self._low, self._high

    def _index_to_x(self, index: int) -> float:
        pad = float(self.HANDLE_RADIUS + 2)
        span = max(1.0, float(self.width()) - 2.0 * pad)
        if self._max_index <= 0:
            return pad
        return pad + (index / self._max_index) * span

    def _x_to_index(self, x: float) -> int:
        if self._max_index <= 0:
            return 0
        pad = float(self.HANDLE_RADIUS + 2)
        span = max(1.0, float(self.width()) - 2.0 * pad)
        ratio = max(0.0, min(1.0, (x - pad) / span))
        return round(ratio * self._max_index)

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pad = float(self.HANDLE_RADIUS + 2)
        cy = self.height() / 2.0
        th = float(self.TRACK_HEIGHT)

        # Full groove
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self._groove_color))
        painter.drawRoundedRect(
            QRectF(pad, cy - th / 2.0, max(1.0, self.width() - 2.0 * pad), th),
            th / 2.0,
            th / 2.0,
        )

        x_low = self._index_to_x(self._low)
        x_high = self._index_to_x(self._high)

        # Selected range bar
        active_fill = (
            self._fill_color
            if self.isEnabled()
            else self._groove_color
        )
        painter.setBrush(QBrush(active_fill))
        painter.drawRoundedRect(
            QRectF(x_low, cy - th / 2.0, max(2.0, x_high - x_low), th),
            th / 2.0,
            th / 2.0,
        )

        # Handles
        r = float(self.HANDLE_RADIUS)
        for which, x_pos in (("low", x_low), ("high", x_high)):
            is_active = self._dragging == which or self._hovered == which
            border_col = (
                self._hover_color
                if (is_active and self.isEnabled())
                else active_fill
            )
            fill_col = (
                border_col
                if (self._dragging == which and self.isEnabled())
                else self._handle_color
            )
            painter.setPen(QPen(border_col, 2.0))
            painter.setBrush(QBrush(fill_col))
            painter.drawEllipse(QPointF(x_pos, cy), r, r)

        painter.end()

    def _pick_handle(self, x: float) -> str:
        x_low = self._index_to_x(self._low)
        x_high = self._index_to_x(self._high)
        d_low = abs(x - x_low)
        d_high = abs(x - x_high)
        if d_low < d_high:
            return "low"
        if d_high < d_low:
            return "high"
        return "low" if x < x_low else "high"

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if not self.isEnabled() or event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        x = event.position().x()
        self._dragging = self._pick_handle(x)
        self._move_dragged_handle(x)
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if not self.isEnabled():
            return super().mouseMoveEvent(event)
        x = event.position().x()
        if self._dragging is not None:
            self._move_dragged_handle(x)
            event.accept()
            return
        x_low = self._index_to_x(self._low)
        x_high = self._index_to_x(self._high)
        near = None
        if abs(x - x_low) <= self.HANDLE_RADIUS + 4:
            near = "low"
        elif abs(x - x_high) <= self.HANDLE_RADIUS + 4:
            near = "high"
        if near != self._hovered:
            self._hovered = near
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._dragging is not None:
            self._dragging = None
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hovered = None
        self.update()
        super().leaveEvent(event)

    def _move_dragged_handle(self, x: float) -> None:
        idx = self._x_to_index(x)
        if self._dragging == "low":
            new_low = min(idx, self._high)
            self.set_indices(new_low, self._high, emit=True)
        elif self._dragging == "high":
            new_high = max(idx, self._low)
            self.set_indices(self._low, new_high, emit=True)


class DateRangeSlider(QWidget):
    """Two-handle start/end date/datetime range slider with live boundary labels."""

    range_changed = Signal(int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._labels: list[str] = []

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 0, 0)
        layout.setSpacing(8)

        self._start_label = QLabel("—")
        self._start_label.setObjectName("statusLabel")
        self._start_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._start_label.setMinimumWidth(78)
        layout.addWidget(self._start_label)

        self._track = _DualRangeTrack(self)
        self._track.range_changed.connect(self._on_track_changed)
        layout.addWidget(self._track, stretch=1)

        self._end_label = QLabel("—")
        self._end_label.setObjectName("statusLabel")
        self._end_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._end_label.setMinimumWidth(78)
        layout.addWidget(self._end_label)

        self.setEnabled(False)

    def update_theme(
        self,
        groove: str,
        fill: str,
        hover: str,
        handle: str,
    ) -> None:
        """Apply theme colors to the slider track and handles."""
        self._track.set_theme_colors(groove, fill, hover, handle)

    def set_dates(self, display_labels: list[str]) -> None:
        """Populate the slider with ordered date/datetime display strings."""
        self._labels = list(display_labels)
        if not self._labels:
            self._track.set_bounds(0)
            self._start_label.setText("—")
            self._end_label.setText("—")
            self.setEnabled(False)
            return

        self.setEnabled(len(self._labels) > 1)
        self._track.set_bounds(len(self._labels) - 1)
        self._refresh_labels()

    def set_range(self, low: int, high: int, *, emit: bool = False) -> None:
        """Programmatically set the selected index range `[low, high]`."""
        self._track.set_indices(low, high, emit=emit)
        self._refresh_labels()

    @property
    def selected_indices(self) -> tuple[int, int]:
        """Return `(start_index, end_index)`."""
        return self._track.indices

    def _on_track_changed(self, low: int, high: int) -> None:
        self._refresh_labels()
        self.range_changed.emit(low, high)

    def _refresh_labels(self) -> None:
        if not self._labels:
            self._start_label.setText("—")
            self._end_label.setText("—")
            return
        low, high = self._track.indices
        low = max(0, min(low, len(self._labels) - 1))
        high = max(0, min(high, len(self._labels) - 1))
        self._start_label.setText(self._labels[low])
        self._end_label.setText(self._labels[high])
        self.setToolTip(f"Displayed range: {self._labels[low]} → {self._labels[high]}")

