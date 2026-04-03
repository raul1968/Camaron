import math
from typing import Optional

from PyQt6.QtWidgets import QWidget, QToolTip
from PyQt6.QtCore import Qt, QPoint, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QMouseEvent, QCursor

from core.capsule import Capsule


class OrbitalMap(QWidget):
    """Displays capsules arranged on concentric orbits by orbit_score."""

    capsule_dragged_to_timeline = pyqtSignal(str)  # capsule_id

    _RING_COLORS = [
        QColor(60, 60, 100),
        QColor(60, 80, 120),
        QColor(60, 100, 140),
        QColor(70, 120, 160),
        QColor(80, 140, 180),
    ]
    _N_RINGS = 5
    _DOT_RADIUS = 8

    def __init__(self, parent=None):
        super().__init__(parent)
        self._capsules: list[Capsule] = []
        self._positions: dict[str, QPointF] = {}  # cid -> center point
        self._hovered_id: Optional[str] = None
        self._dragged_id: Optional[str] = None
        self._drag_start: Optional[QPoint] = None

        self.setMinimumSize(200, 200)
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #141422;")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self, capsules: list[Capsule]):
        self._capsules = list(capsules)
        self._recalculate_positions()
        self.update()

    # ------------------------------------------------------------------
    # Layout calculation
    # ------------------------------------------------------------------

    def _recalculate_positions(self):
        self._positions.clear()
        if not self._capsules:
            return

        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        max_r = min(w, h) / 2 - 20

        # Group by ring (orbit_score 0..1 maps to rings 0..N-1)
        rings: list[list[Capsule]] = [[] for _ in range(self._N_RINGS)]
        for cap in self._capsules:
            score = max(0.0, min(1.0, cap.orbit_score))
            ring_idx = int(score * (self._N_RINGS - 1))
            rings[ring_idx].append(cap)

        for ring_idx, ring_caps in enumerate(rings):
            if not ring_caps:
                continue
            r = max_r * (ring_idx + 1) / self._N_RINGS
            for i, cap in enumerate(ring_caps):
                angle = (2 * math.pi * i / len(ring_caps)) - math.pi / 2
                x = cx + r * math.cos(angle)
                y = cy + r * math.sin(angle)
                self._positions[cap.id] = QPointF(x, y)

    def resizeEvent(self, event):
        self._recalculate_positions()
        self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        max_r = min(w, h) / 2 - 20

        # Draw rings
        for i in range(self._N_RINGS):
            r = max_r * (i + 1) / self._N_RINGS
            pen = QPen(self._RING_COLORS[i])
            pen.setWidth(1)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(cx, cy), r, r)

        # Draw center dot
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(200, 200, 255, 80)))
        painter.drawEllipse(QPointF(cx, cy), 4, 4)

        # Draw capsule dots
        for cap in self._capsules:
            pos = self._positions.get(cap.id)
            if pos is None:
                continue

            is_hovered = cap.id == self._hovered_id
            radius = self._DOT_RADIUS + (2 if is_hovered else 0)

            color = self._kind_color(cap)
            painter.setPen(QPen(color.lighter(150), 1))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(pos, radius, radius)

            if is_hovered:
                painter.setPen(QColor(255, 255, 200))
                font = QFont()
                font.setPointSize(8)
                painter.setFont(font)
                painter.drawText(
                    QRectF(pos.x() + radius + 3, pos.y() - 10, 120, 20),
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    cap.name,
                )

        painter.end()

    @staticmethod
    def _kind_color(cap: Capsule) -> QColor:
        from core.capsule import CapsuleKind
        mapping = {
            CapsuleKind.POSE: QColor(100, 200, 100),
            CapsuleKind.TRANSITION: QColor(200, 150, 100),
            CapsuleKind.TIMING: QColor(200, 200, 100),
            CapsuleKind.CYCLE: QColor(150, 100, 200),
            CapsuleKind.CHARACTER: QColor(100, 180, 220),
            CapsuleKind.MEMORY: QColor(220, 120, 120),
            CapsuleKind.UNASSIGNED: QColor(150, 150, 150),
        }
        return mapping.get(cap.kind, QColor(180, 180, 180))

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def _capsule_at(self, pos: QPoint) -> Optional[str]:
        for cid, pt in self._positions.items():
            dx = pos.x() - pt.x()
            dy = pos.y() - pt.y()
            if math.hypot(dx, dy) <= self._DOT_RADIUS + 4:
                return cid
        return None

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position().toPoint()
        cid = self._capsule_at(pos)
        if cid != self._hovered_id:
            self._hovered_id = cid
            self.update()

        if self._dragged_id and self._drag_start:
            # Check if dragged far enough to consider it a real drag
            delta = pos - self._drag_start
            if math.hypot(delta.x(), delta.y()) > 12:
                self.capsule_dragged_to_timeline.emit(self._dragged_id)
                self._dragged_id = None
                self._drag_start = None

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            cid = self._capsule_at(pos)
            if cid:
                self._dragged_id = cid
                self._drag_start = pos

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._dragged_id = None
        self._drag_start = None
