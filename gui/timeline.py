from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, QRect, QRectF, pyqtSignal, QPoint
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QImage, QPixmap, QFont,
    QMouseEvent, QDragEnterEvent, QDropEvent
)

try:
    from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
    _MEDIA_AVAILABLE = True
except ImportError:
    _MEDIA_AVAILABLE = False
    QMediaPlayer = None
    QAudioOutput = None


_CELL_W = 48
_CELL_H = 48
_THUMB_SIZE = 32
_HEADER_H = 20
_WAVEFORM_H = 30


class FrameSlot:
    def __init__(self):
        self.capsule_id: Optional[str] = None
        self.thumbnail: Optional[QImage] = None  # 32×32 QImage


class Timeline(QWidget):
    """Custom-painted animation timeline with playback controls."""

    frame_selected = pyqtSignal(int)           # index of clicked/scrubbed frame
    frame_dropped = pyqtSignal(int, str)       # (index, file_path) on file drop

    def __init__(self, parent=None, fps: int = 12):
        super().__init__(parent)

        self._fps = fps
        self._slots: list[FrameSlot] = []
        self._playhead: int = 0
        self._playing: bool = False
        self._scrubbing: bool = False

        # Playback timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_frame)

        # Audio support
        self.media_player: Optional[object] = None
        self.audio_output: Optional[object] = None
        if _MEDIA_AVAILABLE:
            self.media_player = QMediaPlayer()
            self.audio_output = QAudioOutput()
            self.media_player.setAudioOutput(self.audio_output)

        self._build_ui()
        self.setAcceptDrops(True)
        self.setMinimumHeight(140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Control bar
        ctrl = QHBoxLayout()
        ctrl.setSpacing(4)

        self._btn_play = QPushButton("▶")
        self._btn_play.setFixedWidth(32)
        self._btn_play.clicked.connect(self.play)

        self._btn_pause = QPushButton("⏸")
        self._btn_pause.setFixedWidth(32)
        self._btn_pause.clicked.connect(self.pause)

        self._btn_stop = QPushButton("⏹")
        self._btn_stop.setFixedWidth(32)
        self._btn_stop.clicked.connect(self.stop)

        self._lbl_frame = QLabel("Frame: 0 / 0")
        self._lbl_frame.setStyleSheet("color: #aaa; font-size: 11px;")

        ctrl.addWidget(self._btn_play)
        ctrl.addWidget(self._btn_pause)
        ctrl.addWidget(self._btn_stop)
        ctrl.addWidget(self._lbl_frame)
        ctrl.addStretch()

        layout.addLayout(ctrl)

        # Painted track area (drawn in paintEvent)
        self._track = _TrackWidget(self)
        self._track.frame_selected.connect(self._on_frame_selected)
        self._track.frame_dropped.connect(self.frame_dropped)
        layout.addWidget(self._track)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def slot_count(self) -> int:
        return len(self._slots)

    def add_slot(self, capsule_id: Optional[str] = None) -> int:
        slot = FrameSlot()
        slot.capsule_id = capsule_id
        self._slots.append(slot)
        self._track.set_slots(self._slots, self._playhead)
        self._update_label()
        return len(self._slots) - 1

    def insert_slot(self, index: int, capsule_id: Optional[str] = None):
        slot = FrameSlot()
        slot.capsule_id = capsule_id
        self._slots.insert(index, slot)
        self._track.set_slots(self._slots, self._playhead)
        self._update_label()

    def set_capsule(self, index: int, capsule_id: Optional[str]):
        if 0 <= index < len(self._slots):
            self._slots[index].capsule_id = capsule_id
            self._track.set_slots(self._slots, self._playhead)

    def load_thumbnail(self, index: int, data: bytes):
        if 0 <= index < len(self._slots):
            img = QImage()
            img.loadFromData(data, "PNG")
            if not img.isNull():
                self._slots[index].thumbnail = img.scaled(
                    _THUMB_SIZE, _THUMB_SIZE,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            self._track.set_slots(self._slots, self._playhead)

    def current_frame(self) -> int:
        return self._playhead

    def set_frame(self, index: int):
        if self._slots:
            self._playhead = max(0, min(index, len(self._slots) - 1))
            self._track.set_playhead(self._playhead)
            self._update_label()
            self.frame_selected.emit(self._playhead)

    def get_slot(self, index: int) -> Optional[FrameSlot]:
        if 0 <= index < len(self._slots):
            return self._slots[index]
        return None

    def get_slots(self) -> list[FrameSlot]:
        return self._slots

    def clear(self):
        self._slots.clear()
        self._playhead = 0
        self._track.set_slots(self._slots, self._playhead)
        self._update_label()

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def play(self):
        if not self._slots:
            return
        self._playing = True
        self._timer.start(int(1000 / self._fps))
        if _MEDIA_AVAILABLE and self.media_player:
            self.media_player.play()

    def pause(self):
        self._playing = False
        self._timer.stop()
        if _MEDIA_AVAILABLE and self.media_player:
            self.media_player.pause()

    def stop(self):
        self._playing = False
        self._timer.stop()
        self.set_frame(0)
        if _MEDIA_AVAILABLE and self.media_player:
            self.media_player.stop()

    def _advance_frame(self):
        if not self._slots:
            return
        next_idx = (self._playhead + 1) % len(self._slots)
        self.set_frame(next_idx)

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    def set_audio(self, path: str):
        if _MEDIA_AVAILABLE and self.media_player:
            from PyQt6.QtCore import QUrl
            self.media_player.setSource(QUrl.fromLocalFile(path))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_frame_selected(self, index: int):
        self._playhead = index
        self._update_label()
        self.frame_selected.emit(index)

    def _update_label(self):
        self._lbl_frame.setText(f"Frame: {self._playhead} / {max(0, len(self._slots) - 1)}")


# ---------------------------------------------------------------------------
# Internal track painting widget
# ---------------------------------------------------------------------------

class _TrackWidget(QWidget):
    """Painted track area — handles click/drag scrubbing and file drops."""

    frame_selected = pyqtSignal(int)
    frame_dropped = pyqtSignal(int, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._slots: list[FrameSlot] = []
        self._playhead: int = 0
        self._scroll_offset: int = 0

        self.setMinimumHeight(_HEADER_H + _CELL_H + _WAVEFORM_H + 4)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)

    def set_slots(self, slots: list[FrameSlot], playhead: int):
        self._slots = slots
        self._playhead = playhead
        total_w = max(len(slots) * _CELL_W, self.width())
        self.setMinimumWidth(total_w)
        self.update()

    def set_playhead(self, index: int):
        self._playhead = index
        self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(self.rect(), QColor(28, 28, 35))

        font = QFont()
        font.setPointSize(7)
        painter.setFont(font)

        # Header row (frame numbers)
        for i, slot in enumerate(self._slots):
            x = i * _CELL_W - self._scroll_offset
            if x + _CELL_W < 0 or x > self.width():
                continue
            # Frame number
            num_rect = QRect(x, 0, _CELL_W, _HEADER_H)
            painter.setPen(QColor(130, 130, 140))
            painter.drawText(num_rect, Qt.AlignmentFlag.AlignCenter, str(i))

        # Cell row
        for i, slot in enumerate(self._slots):
            x = i * _CELL_W - self._scroll_offset
            if x + _CELL_W < 0 or x > self.width():
                continue
            cell_rect = QRect(x, _HEADER_H, _CELL_W, _CELL_H)
            self._draw_cell(painter, cell_rect, slot, i == self._playhead)

        # Waveform placeholder
        wave_y = _HEADER_H + _CELL_H
        wave_rect = QRect(0, wave_y, self.width(), _WAVEFORM_H)
        painter.fillRect(wave_rect, QColor(20, 20, 30))
        painter.setPen(QColor(60, 100, 80))
        mid = wave_y + _WAVEFORM_H // 2
        painter.drawLine(0, mid, self.width(), mid)

        # Playhead
        ph_x = self._playhead * _CELL_W - self._scroll_offset + _CELL_W // 2
        painter.setPen(QPen(QColor(255, 80, 80), 2))
        painter.drawLine(ph_x, 0, ph_x, _HEADER_H + _CELL_H)

        painter.end()

    def _draw_cell(self, painter: QPainter, rect: QRect, slot: FrameSlot, is_playhead: bool):
        bg = QColor(45, 45, 55) if not is_playhead else QColor(70, 55, 80)
        painter.fillRect(rect, bg)
        painter.setPen(QPen(QColor(70, 70, 80), 1))
        painter.drawRect(rect.adjusted(0, 0, -1, -1))

        if slot.thumbnail:
            # Center thumbnail
            tw = slot.thumbnail.width()
            th = slot.thumbnail.height()
            tx = rect.left() + (rect.width() - tw) // 2
            ty = rect.top() + (rect.height() - th) // 2
            painter.drawImage(tx, ty, slot.thumbnail)
        elif slot.capsule_id:
            # Show small indicator
            painter.setPen(QColor(160, 200, 160))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "◆")

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------

    def _index_at(self, pos: QPoint) -> int:
        x = pos.x() + self._scroll_offset
        return x // _CELL_W

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            idx = self._index_at(event.position().toPoint())
            if 0 <= idx < len(self._slots):
                self.frame_selected.emit(idx)

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() == Qt.MouseButton.LeftButton:
            idx = self._index_at(event.position().toPoint())
            if 0 <= idx < len(self._slots):
                self.frame_selected.emit(idx)

    # ------------------------------------------------------------------
    # Drag and drop (image files)
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        pos = event.position().toPoint()
        idx = self._index_at(pos)
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path:
                self.frame_dropped.emit(max(0, idx), path)
                break
        event.acceptProposedAction()
