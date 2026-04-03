import io
from typing import Optional

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel
from PyQt6.QtCore import Qt, QPoint, QRect, pyqtSignal, QSize
from PyQt6.QtGui import (
    QImage, QPainter, QColor, QPen, QBrush, QPixmap,
    QMouseEvent, QKeyEvent, QDragEnterEvent, QDropEvent,
    QImageWriter
)


class Layer:
    """A single named drawing layer backed by QImage."""

    MAX_UNDO = 50

    def __init__(self, name: str, width: int, height: int):
        self.name = name
        self.image = QImage(width, height, QImage.Format.Format_ARGB32)
        self.image.fill(Qt.GlobalColor.transparent)
        self._undo_stack: list[QImage] = []
        self._redo_stack: list[QImage] = []

    def push_undo(self):
        self._undo_stack.append(self.image.copy())
        if len(self._undo_stack) > self.MAX_UNDO:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self):
        if self._undo_stack:
            self._redo_stack.append(self.image.copy())
            self.image = self._undo_stack.pop()

    def redo(self):
        if self._redo_stack:
            self._undo_stack.append(self.image.copy())
            self.image = self._redo_stack.pop()

    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        return bool(self._redo_stack)


class Canvas(QWidget):
    """Multi-layer drawing canvas with Pen/Eraser/Line/Rect/Fill tools."""

    capsule_dropped = pyqtSignal(str)  # emitted with file path on drop

    TOOLS = ["Pen", "Eraser", "Line", "Rectangle", "Fill"]

    def __init__(self, parent=None, width: int = 800, height: int = 600):
        super().__init__(parent)
        self._canvas_w = width
        self._canvas_h = height

        self._layers: list[Layer] = [Layer("Layer 1", width, height)]
        self._active_layer_idx: int = 0

        self._tool: str = "Pen"
        self._pen_color: QColor = QColor(255, 255, 255)
        self._pen_width: int = 3

        self._drawing = False
        self._last_point: Optional[QPoint] = None
        self._start_point: Optional[QPoint] = None

        # overlay for live preview of line/rect
        self._preview_image: Optional[QImage] = None

        self.setMinimumSize(width, height)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAcceptDrops(True)
        self.setStyleSheet("background-color: #1e1e1e;")

    # ------------------------------------------------------------------
    # Layer management
    # ------------------------------------------------------------------

    @property
    def active_layer(self) -> Layer:
        return self._layers[self._active_layer_idx]

    def add_layer(self, name: Optional[str] = None) -> int:
        idx = len(self._layers)
        layer_name = name or f"Layer {idx + 1}"
        self._layers.append(Layer(layer_name, self._canvas_w, self._canvas_h))
        return idx

    def set_active_layer(self, idx: int):
        if 0 <= idx < len(self._layers):
            self._active_layer_idx = idx
            self.update()

    def layer_names(self) -> list[str]:
        return [l.name for l in self._layers]

    def layer_count(self) -> int:
        return len(self._layers)

    def get_active_layer_index(self) -> int:
        return self._active_layer_idx

    # ------------------------------------------------------------------
    # Tool API
    # ------------------------------------------------------------------

    def set_tool(self, tool: str):
        if tool in self.TOOLS:
            self._tool = tool

    def set_pen_color(self, color: QColor):
        self._pen_color = color

    def set_pen_width(self, width: int):
        self._pen_width = max(1, width)

    # ------------------------------------------------------------------
    # Undo / Redo
    # ------------------------------------------------------------------

    def undo(self):
        if self.active_layer.can_undo():
            self.active_layer.undo()
            self.update()

    def redo(self):
        if self.active_layer.can_redo():
            self.active_layer.redo()
            self.update()

    # ------------------------------------------------------------------
    # Image I/O
    # ------------------------------------------------------------------

    def get_image_bytes(self) -> bytes:
        """Return composite PNG bytes of all layers."""
        composite = QImage(self._canvas_w, self._canvas_h, QImage.Format.Format_ARGB32)
        composite.fill(Qt.GlobalColor.transparent)
        painter = QPainter(composite)
        for layer in self._layers:
            painter.drawImage(0, 0, layer.image)
        painter.end()

        from PyQt6.QtCore import QByteArray, QBuffer, QIODevice
        qba = QByteArray()
        buffer = QBuffer(qba)
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        composite.save(buffer, "PNG")
        buffer.close()
        return bytes(qba.data())

    def load_image(self, data: bytes):
        """Load PNG bytes onto the active layer."""
        self.active_layer.push_undo()
        img = QImage()
        img.loadFromData(data, "PNG")
        if not img.isNull():
            scaled = img.scaled(
                self._canvas_w, self._canvas_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            painter = QPainter(self.active_layer.image)
            painter.drawImage(0, 0, scaled)
            painter.end()
        self.update()

    def clear_active_layer(self):
        self.active_layer.push_undo()
        self.active_layer.image.fill(Qt.GlobalColor.transparent)
        self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        # Draw all layers
        for layer in self._layers:
            painter.drawImage(0, 0, layer.image)

        # Draw preview overlay for line/rect
        if self._preview_image is not None:
            painter.drawImage(0, 0, self._preview_image)

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drawing = True
            pos = event.position().toPoint()
            self._start_point = pos
            self._last_point = pos
            if self._tool in ("Pen", "Eraser"):
                self.active_layer.push_undo()
                self._draw_point(pos)

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self._drawing:
            return
        pos = event.position().toPoint()
        if self._tool == "Pen":
            self._draw_line(self._last_point, pos)
            self._last_point = pos
        elif self._tool == "Eraser":
            self._erase_line(self._last_point, pos)
            self._last_point = pos
        elif self._tool in ("Line", "Rectangle"):
            self._update_preview(pos)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self._drawing:
            pos = event.position().toPoint()
            self._drawing = False

            if self._tool == "Line":
                self.active_layer.push_undo()
                self._commit_line(self._start_point, pos)
            elif self._tool == "Rectangle":
                self.active_layer.push_undo()
                self._commit_rect(self._start_point, pos)
            elif self._tool == "Fill":
                self.active_layer.push_undo()
                self._flood_fill(pos, self._pen_color)

            self._preview_image = None
            self.update()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _make_pen(self, color: Optional[QColor] = None) -> QPen:
        pen = QPen(color or self._pen_color)
        pen.setWidth(self._pen_width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        return pen

    def _draw_point(self, pos: QPoint):
        painter = QPainter(self.active_layer.image)
        painter.setPen(self._make_pen())
        painter.drawPoint(pos)
        painter.end()
        self.update()

    def _draw_line(self, p1: QPoint, p2: QPoint):
        painter = QPainter(self.active_layer.image)
        painter.setPen(self._make_pen())
        painter.drawLine(p1, p2)
        painter.end()
        self.update()

    def _erase_line(self, p1: QPoint, p2: QPoint):
        painter = QPainter(self.active_layer.image)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        pen = QPen(Qt.GlobalColor.transparent)
        pen.setWidth(self._pen_width * 3)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(p1, p2)
        painter.end()
        self.update()

    def _update_preview(self, current: QPoint):
        preview = self.active_layer.image.copy()
        painter = QPainter(preview)
        painter.setPen(self._make_pen())
        if self._tool == "Line":
            painter.drawLine(self._start_point, current)
        elif self._tool == "Rectangle":
            painter.drawRect(QRect(self._start_point, current).normalized())
        painter.end()
        self._preview_image = preview
        self.update()

    def _commit_line(self, p1: QPoint, p2: QPoint):
        painter = QPainter(self.active_layer.image)
        painter.setPen(self._make_pen())
        painter.drawLine(p1, p2)
        painter.end()
        self.update()

    def _commit_rect(self, p1: QPoint, p2: QPoint):
        painter = QPainter(self.active_layer.image)
        painter.setPen(self._make_pen())
        painter.drawRect(QRect(p1, p2).normalized())
        painter.end()
        self.update()

    def _flood_fill(self, pos: QPoint, color: QColor):
        """Simple 4-connected flood fill on the active layer image."""
        img = self.active_layer.image
        x, y = pos.x(), pos.y()
        if not (0 <= x < img.width() and 0 <= y < img.height()):
            return

        target_color = img.pixel(x, y)
        fill_color = color.rgba()
        if target_color == fill_color:
            return

        stack = [(x, y)]
        w, h = img.width(), img.height()
        while stack:
            cx, cy = stack.pop()
            if not (0 <= cx < w and 0 <= cy < h):
                continue
            if img.pixel(cx, cy) != target_color:
                continue
            img.setPixel(cx, cy, fill_color)
            stack.append((cx + 1, cy))
            stack.append((cx - 1, cy))
            stack.append((cx, cy + 1))
            stack.append((cx, cy - 1))

        self.update()

    # ------------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent):
        modifiers = event.modifiers()
        key = event.key()
        if modifiers == Qt.KeyboardModifier.ControlModifier:
            if key == Qt.Key.Key_BracketLeft:
                # Previous layer
                self._active_layer_idx = max(0, self._active_layer_idx - 1)
                self.update()
                return
            elif key == Qt.Key.Key_BracketRight:
                # Next layer
                self._active_layer_idx = min(len(self._layers) - 1, self._active_layer_idx + 1)
                self.update()
                return
            elif key == Qt.Key.Key_Z:
                self.undo()
                return
            elif key == Qt.Key.Key_Y:
                self.redo()
                return
        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Drag and drop (files only — emit signal, do NOT add to timeline)
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path:
                self.capsule_dropped.emit(path)
        event.acceptProposedAction()
