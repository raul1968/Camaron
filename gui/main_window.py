import io
import json
import os
import time
import zipfile
from typing import Optional

import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QAction, QColor, QIcon, QKeySequence, QImage
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QDockWidget, QToolBar,
    QFileDialog, QMessageBox, QStatusBar, QWidget, QComboBox,
    QLabel, QSizePolicy
)

from core.capsule import Capsule, CapsuleKind, make_capsule_id, image_hash
from core.capsule_store import CapsuleStore
from core.device_manager import get_device_info
from core import router

from gui.canvas import Canvas
from gui.timeline import Timeline
from gui.orbital_map import OrbitalMap
from gui.dialogs import TransitionDialog, CycleDialog, AgreementWarningDialog

_DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
}
QMenuBar {
    background-color: #181825;
    color: #cdd6f4;
}
QMenuBar::item:selected {
    background-color: #313244;
}
QMenu {
    background-color: #181825;
    color: #cdd6f4;
    border: 1px solid #45475a;
}
QMenu::item:selected {
    background-color: #313244;
}
QToolBar {
    background-color: #181825;
    border: none;
    spacing: 4px;
}
QToolButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px 8px;
}
QToolButton:hover {
    background-color: #45475a;
}
QToolButton:checked {
    background-color: #585b70;
}
QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px 10px;
}
QPushButton:hover {
    background-color: #45475a;
}
QDockWidget {
    color: #cdd6f4;
    titlebar-close-icon: none;
}
QDockWidget::title {
    background-color: #181825;
    padding: 4px;
}
QStatusBar {
    background-color: #181825;
    color: #a6adc8;
    font-size: 11px;
}
QComboBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 2px 6px;
}
QComboBox QAbstractItemView {
    background-color: #181825;
    color: #cdd6f4;
    selection-background-color: #313244;
}
QDialog {
    background-color: #1e1e2e;
    color: #cdd6f4;
}
QLabel {
    color: #cdd6f4;
}
QLineEdit, QSpinBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 2px 6px;
}
QDialogButtonBox QPushButton {
    min-width: 70px;
}
"""

# ---------------------------------------------------------------------------
# Worker thread for transition generation
# ---------------------------------------------------------------------------

class TransitionWorker(QThread):
    done = pyqtSignal(list)  # list of np.ndarray frames

    def __init__(self, frame_a: np.ndarray, frame_b: np.ndarray, n_frames: int):
        super().__init__()
        self._frame_a = frame_a
        self._frame_b = frame_b
        self._n_frames = n_frames

    def run(self):
        frames = router.generate_transitions(self._frame_a, self._frame_b, self._n_frames)
        self.done.emit(frames)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _qimage_to_ndarray(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    ptr = img.bits()
    ptr.setsize(h * w * 3)
    return np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 3)).copy()


def _ndarray_to_png_bytes(arr: np.ndarray) -> bytes:
    if _CV2_AVAILABLE:
        ok, buf = cv2.imencode(".png", arr)
        if ok:
            return buf.tobytes()
    # Fallback via QImage
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    h, w = arr.shape[:2]
    channels = arr.shape[2] if arr.ndim == 3 else 1
    if channels == 3:
        fmt = QImage.Format.Format_RGB888
        img = QImage(arr.tobytes(), w, h, w * 3, fmt)
    else:
        img = QImage(arr.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
    from PyQt6.QtCore import QByteArray, QBuffer, QIODevice
    qba = QByteArray()
    buf = QBuffer(qba)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    buf.close()
    return bytes(qba.data())


def _png_bytes_from_file(path: str) -> Optional[bytes]:
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return None


def _orbit_score_from_use(use_count: int) -> float:
    """Map use_count to [0, 1] orbit score using saturation curve."""
    return min(1.0, use_count / 20.0)


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROCA Animator")
        self.resize(1280, 800)

        self._store = CapsuleStore()
        self._current_project_path: Optional[str] = None
        self._transition_worker: Optional[TransitionWorker] = None

        self._init_canvas()
        self._init_timeline_dock()
        self._init_orbital_dock()
        self._init_menu()
        self._init_toolbar()
        self._init_statusbar()
        self._connect_signals()

    # ------------------------------------------------------------------
    # Widget setup
    # ------------------------------------------------------------------

    def _init_canvas(self):
        self._canvas = Canvas(self)
        self.setCentralWidget(self._canvas)

    def _init_timeline_dock(self):
        self._timeline = Timeline(self)
        dock = QDockWidget("Timeline", self)
        dock.setWidget(self._timeline)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

    def _init_orbital_dock(self):
        self._orbital = OrbitalMap(self)
        dock = QDockWidget("Orbital Map", self)
        dock.setWidget(self._orbital)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        dock.setMinimumWidth(220)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _init_menu(self):
        mb = self.menuBar()

        # File menu
        file_menu = mb.addMenu("&File")
        act_new = QAction("&New Project", self)
        act_new.setShortcut(QKeySequence.StandardKey.New)
        act_new.triggered.connect(self._on_new_project)
        file_menu.addAction(act_new)

        act_open = QAction("&Open (.roca)…", self)
        act_open.setShortcut(QKeySequence.StandardKey.Open)
        act_open.triggered.connect(self._on_open)
        file_menu.addAction(act_open)

        act_save = QAction("&Save (.roca)", self)
        act_save.setShortcut(QKeySequence.StandardKey.Save)
        act_save.triggered.connect(self._on_save)
        file_menu.addAction(act_save)

        act_save_as = QAction("Save &As…", self)
        act_save_as.setShortcut(QKeySequence.StandardKey.SaveAs)
        act_save_as.triggered.connect(self._on_save_as)
        file_menu.addAction(act_save_as)

        file_menu.addSeparator()

        act_quit = QAction("&Quit", self)
        act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # Edit menu
        edit_menu = mb.addMenu("&Edit")
        act_undo = QAction("&Undo", self)
        act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        act_undo.triggered.connect(self._canvas.undo)
        edit_menu.addAction(act_undo)

        act_redo = QAction("&Redo", self)
        act_redo.setShortcut(QKeySequence.StandardKey.Redo)
        act_redo.triggered.connect(self._canvas.redo)
        edit_menu.addAction(act_redo)

    def _init_toolbar(self):
        tb = QToolBar("Tools", self)
        tb.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)

        # --- Tool selection ---
        tool_label = QLabel("Tool:")
        tb.addWidget(tool_label)

        self._tool_combo = QComboBox()
        self._tool_combo.addItems(Canvas.TOOLS)
        self._tool_combo.currentTextChanged.connect(self._canvas.set_tool)
        tb.addWidget(self._tool_combo)

        tb.addSeparator()

        # --- Layer selector ---
        layer_label = QLabel("Layer:")
        tb.addWidget(layer_label)

        self._layer_combo = QComboBox()
        self._layer_combo.addItems(self._canvas.layer_names())
        self._layer_combo.currentIndexChanged.connect(self._canvas.set_active_layer)
        tb.addWidget(self._layer_combo)

        act_add_layer = QAction("+ Layer", self)
        act_add_layer.triggered.connect(self._on_add_layer)
        tb.addAction(act_add_layer)

        tb.addSeparator()

        # --- Capsule operations ---
        act_capture = QAction("Capture Pose", self)
        act_capture.setToolTip("Capture current canvas as a Pose capsule")
        act_capture.triggered.connect(self._on_capture_pose)
        tb.addAction(act_capture)

        act_transition = QAction("Generate Transition", self)
        act_transition.setToolTip("Generate transition frames between adjacent poses")
        act_transition.triggered.connect(self._on_generate_transition)
        tb.addAction(act_transition)

        act_cycle = QAction("Create Cycle", self)
        act_cycle.setToolTip("Create a Cycle capsule from the selected frame range")
        act_cycle.triggered.connect(self._on_create_cycle)
        tb.addAction(act_cycle)

        act_apply_cycle = QAction("Apply Cycle", self)
        act_apply_cycle.setToolTip("Apply the selected Cycle capsule at the playhead")
        act_apply_cycle.triggered.connect(self._on_apply_cycle)
        tb.addAction(act_apply_cycle)

        tb.addSeparator()

        # --- Import / Export ---
        act_load_images = QAction("Load Images", self)
        act_load_images.triggered.connect(self._on_load_images)
        tb.addAction(act_load_images)

        act_load_video = QAction("Load Video", self)
        act_load_video.triggered.connect(self._on_load_video)
        tb.addAction(act_load_video)

        act_export_video = QAction("Export Video", self)
        act_export_video.triggered.connect(self._on_export_video)
        tb.addAction(act_export_video)

        act_export_svg = QAction("Export SVG", self)
        act_export_svg.triggered.connect(self._on_export_svg)
        tb.addAction(act_export_svg)

        act_load_audio = QAction("Load Audio", self)
        act_load_audio.triggered.connect(self._on_load_audio)
        tb.addAction(act_load_audio)

    def _init_statusbar(self):
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        info = get_device_info()
        flag = "✓" if info["opencl_available"] else "○"
        self._status.showMessage(f"Device: {flag} {info['device_name']}")

    def _connect_signals(self):
        self._orbital.capsule_dragged_to_timeline.connect(self._on_capsule_dragged)
        self._timeline.frame_selected.connect(self._on_frame_selected)
        self._timeline.frame_dropped.connect(self._on_frame_file_dropped)
        self._canvas.capsule_dropped.connect(self._on_canvas_file_dropped)

    # ------------------------------------------------------------------
    # Layer management
    # ------------------------------------------------------------------

    def _on_add_layer(self):
        idx = self._canvas.add_layer()
        self._layer_combo.addItem(self._canvas.layer_names()[idx])
        self._layer_combo.setCurrentIndex(idx)

    def _refresh_layer_combo(self):
        self._layer_combo.blockSignals(True)
        self._layer_combo.clear()
        self._layer_combo.addItems(self._canvas.layer_names())
        self._layer_combo.setCurrentIndex(self._canvas._active_layer_idx)
        self._layer_combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Capsule operations
    # ------------------------------------------------------------------

    def _on_capture_pose(self):
        data = self._canvas.get_image_bytes()
        if not data:
            return
        h = image_hash(data)
        name = f"pose_{h[:8]}"
        cid = make_capsule_id(CapsuleKind.POSE, name)

        cap = self._store.get(cid)
        if cap is None:
            cap = Capsule(
                id=cid,
                kind=CapsuleKind.POSE,
                name=name,
                image_data=data,
                asset_hash=h,
            )
            self._store.add(cap)

        self._use_capsule(cap)
        # Assign to current timeline frame
        idx = self._timeline.current_frame()
        if self._timeline.slot_count() == 0:
            self._timeline.add_slot(cid)
            self._timeline.load_thumbnail(0, data)
        elif idx < self._timeline.slot_count():
            self._timeline.set_capsule(idx, cid)
            self._timeline.load_thumbnail(idx, data)
        else:
            new_idx = self._timeline.add_slot(cid)
            self._timeline.load_thumbnail(new_idx, data)

        self._orbital.refresh(self._store.all())
        self._status.showMessage(f"Captured pose: {name}")

    def _on_generate_transition(self):
        tl = self._timeline
        idx = tl.current_frame()
        if idx < 1 or idx >= tl.slot_count():
            QMessageBox.information(self, "Generate Transition",
                                    "Place the playhead between two frames (not at position 0).")
            return

        slot_a = tl.get_slot(idx - 1)
        slot_b = tl.get_slot(idx)
        if not slot_a or not slot_b:
            return
        if not slot_a.capsule_id or not slot_b.capsule_id:
            QMessageBox.information(self, "Generate Transition",
                                    "Both adjacent frames must have capsules assigned.")
            return

        cap_a = self._store.get(slot_a.capsule_id)
        cap_b = self._store.get(slot_b.capsule_id)
        if not cap_a or not cap_b:
            return
        if not cap_a.image_data or not cap_b.image_data:
            QMessageBox.warning(self, "Generate Transition",
                                "Both capsules must have image data for transitions.")
            return

        # Agreement check
        if self._store.agreement_score(cap_a.id, cap_b.id) < 1.0:
            dlg = AgreementWarningDialog(self, cap_a.name, cap_b.name)
            if dlg.exec() != AgreementWarningDialog.DialogCode.Accepted:
                return

        dlg = TransitionDialog(self)
        if dlg.exec() != TransitionDialog.DialogCode.Accepted:
            return
        n_frames = dlg.get_n_frames()

        img_a = self._png_bytes_to_ndarray(cap_a.image_data)
        img_b = self._png_bytes_to_ndarray(cap_b.image_data)
        if img_a is None or img_b is None:
            QMessageBox.warning(self, "Error", "Could not decode image data.")
            return

        self._status.showMessage("Generating transitions…")
        self._transition_worker = TransitionWorker(img_a, img_b, n_frames)
        self._transition_worker.done.connect(
            lambda frames: self._on_transition_done(frames, idx, cap_a.id, cap_b.id)
        )
        self._transition_worker.start()

    def _on_transition_done(self, frames: list, insert_at: int, id_a: str, id_b: str):
        for i, frame_arr in enumerate(frames):
            png_data = _ndarray_to_png_bytes(frame_arr)
            h = image_hash(png_data)
            name = f"pose_{h[:8]}"
            cid = make_capsule_id(CapsuleKind.TRANSITION, name)
            cap = Capsule(
                id=cid,
                kind=CapsuleKind.TRANSITION,
                name=name,
                image_data=png_data,
                asset_hash=h,
            )
            self._store.add(cap)
            self._timeline.insert_slot(insert_at + i, cid)
            self._timeline.load_thumbnail(insert_at + i, png_data)

        self._store.record_agreement(id_a, id_b, 1.0)
        self._orbital.refresh(self._store.all())
        self._status.showMessage(f"Inserted {len(frames)} transition frames.")

    def _on_create_cycle(self):
        tl = self._timeline
        if tl.slot_count() < 2:
            QMessageBox.information(self, "Create Cycle", "Need at least 2 frames.")
            return
        dlg = CycleDialog(self)
        if dlg.exec() != CycleDialog.DialogCode.Accepted:
            return
        name = dlg.get_name()
        loop_count = dlg.get_loop_count()

        cid = make_capsule_id(CapsuleKind.CYCLE, name)
        cap = Capsule(id=cid, kind=CapsuleKind.CYCLE, name=name)
        self._store.add(cap)

        # Duplicate current frame range loop_count times
        slots = list(tl.get_slots())
        for _ in range(loop_count - 1):
            for slot in slots:
                new_idx = tl.add_slot(slot.capsule_id)
                if slot.thumbnail:
                    from PyQt6.QtCore import QByteArray, QBuffer, QIODevice
                    qba = QByteArray()
                    buf = QBuffer(qba)
                    buf.open(QIODevice.OpenModeFlag.WriteOnly)
                    slot.thumbnail.save(buf, "PNG")
                    buf.close()
                    tl.load_thumbnail(new_idx, bytes(qba.data()))

        self._orbital.refresh(self._store.all())
        self._status.showMessage(f"Cycle '{name}' created ({loop_count} loops).")

    def _on_apply_cycle(self):
        # Find most recently used cycle capsule
        cycles = [c for c in self._store.all() if c.kind == CapsuleKind.CYCLE]
        if not cycles:
            QMessageBox.information(self, "Apply Cycle", "No Cycle capsules in library.")
            return
        cycles.sort(key=lambda c: c.last_used_at or 0, reverse=True)
        cap = cycles[0]
        self._use_capsule(cap)
        self._status.showMessage(f"Applied cycle: {cap.name}")

    # ------------------------------------------------------------------
    # Import / Export
    # ------------------------------------------------------------------

    def _on_load_images(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
        paths = sorted(
            p for p in (
                os.path.join(folder, f) for f in os.listdir(folder)
            )
            if os.path.splitext(p)[1].lower() in exts
        )
        if not paths:
            QMessageBox.information(self, "Load Images", "No image files found in folder.")
            return
        self._import_images_as_capsules(paths)

    def _import_images_as_capsules(self, paths: list[str]):
        count = 0
        for path in paths:
            data = _png_bytes_from_file(path)
            if not data:
                continue
            h = image_hash(data)
            name = f"pose_{h[:8]}"
            cid = make_capsule_id(CapsuleKind.POSE, name)
            if self._store.get(cid) is None:
                cap = Capsule(
                    id=cid,
                    kind=CapsuleKind.POSE,
                    name=name,
                    image_data=data,
                    asset_hash=h,
                    asset_path=path,
                )
                self._store.add(cap)
            else:
                cap = self._store.get(cid)
            self._use_capsule(cap)
            idx = self._timeline.add_slot(cid)
            self._timeline.load_thumbnail(idx, data)
            count += 1

        self._orbital.refresh(self._store.all())
        self._status.showMessage(f"Loaded {count} images.")

    def _on_load_video(self):
        if not _CV2_AVAILABLE:
            QMessageBox.warning(self, "Load Video", "OpenCV (cv2) is not installed.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video",
            filter="Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)"
        )
        if not path:
            return

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.warning(self, "Load Video", f"Could not open video: {path}")
            return

        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            png_data = _ndarray_to_png_bytes(frame)
            h = image_hash(png_data)
            name = f"pose_{h[:8]}"
            cid = make_capsule_id(CapsuleKind.POSE, name)
            if self._store.get(cid) is None:
                capsule = Capsule(
                    id=cid,
                    kind=CapsuleKind.POSE,
                    name=name,
                    image_data=png_data,
                    asset_hash=h,
                )
                self._store.add(capsule)
            else:
                capsule = self._store.get(cid)
            self._use_capsule(capsule)
            idx = self._timeline.add_slot(cid)
            self._timeline.load_thumbnail(idx, png_data)
            count += 1

        cap.release()
        self._orbital.refresh(self._store.all())
        self._status.showMessage(f"Loaded {count} frames from video.")

    def _on_export_video(self):
        if not _CV2_AVAILABLE:
            QMessageBox.warning(self, "Export Video", "OpenCV (cv2) is not installed.")
            return
        if self._timeline.slot_count() == 0:
            QMessageBox.information(self, "Export Video", "Timeline is empty.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Video",
            filter="MP4 (*.mp4);;AVI (*.avi)"
        )
        if not path:
            return

        slots = self._timeline.get_slots()
        # Determine resolution from first capsule with image
        width, height = 800, 600
        for slot in slots:
            if slot.capsule_id:
                c = self._store.get(slot.capsule_id)
                if c and c.image_data:
                    arr = self._png_bytes_to_ndarray(c.image_data)
                    if arr is not None:
                        height, width = arr.shape[:2]
                    break

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = self._timeline._fps
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

        written = 0
        for slot in slots:
            frame = None
            if slot.capsule_id:
                c = self._store.get(slot.capsule_id)
                if c and c.image_data:
                    frame = self._png_bytes_to_ndarray(c.image_data)
                    if frame is not None:
                        if frame.shape[:2] != (height, width):
                            frame = cv2.resize(frame, (width, height))
            if frame is None:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            writer.write(frame)
            written += 1

        writer.release()
        self._status.showMessage(f"Exported {written} frames to {path}")

    def _on_export_svg(self):
        QMessageBox.information(self, "Export SVG", "SVG export is not yet implemented.")

    def _on_load_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Audio",
            filter="Audio Files (*.mp3 *.wav *.ogg *.flac *.aac);;All Files (*)"
        )
        if path:
            self._timeline.set_audio(path)
            self._status.showMessage(f"Audio loaded: {os.path.basename(path)}")

    # ------------------------------------------------------------------
    # File menu handlers
    # ------------------------------------------------------------------

    def _on_new_project(self):
        reply = QMessageBox.question(
            self, "New Project",
            "Start a new project? Unsaved changes will be lost.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._store = CapsuleStore()
        self._timeline.clear()
        self._orbital.refresh([])
        self._current_project_path = None
        self.setWindowTitle("ROCA Animator")
        self._status.showMessage("New project created.")

    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project",
            filter="ROCA Project (*.roca);;All Files (*)"
        )
        if path:
            self._load_roca(path)

    def _on_save(self):
        if self._current_project_path:
            self._save_roca(self._current_project_path)
        else:
            self._on_save_as()

    def _on_save_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project",
            filter="ROCA Project (*.roca);;All Files (*)"
        )
        if path:
            if not path.endswith(".roca"):
                path += ".roca"
            self._save_roca(path)

    # ------------------------------------------------------------------
    # .roca save / load (ZIP format)
    # ------------------------------------------------------------------

    def _save_roca(self, path: str):
        store_data = self._store.to_dict()

        timeline_data = []
        for slot in self._timeline.get_slots():
            timeline_data.append({"capsule_id": slot.capsule_id})

        manifest = {
            "fps": self._timeline._fps,
            "timeline": timeline_data,
            "capsules": store_data["capsules"],
            "edges": store_data["edges"],
        }

        try:
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("manifest.json", json.dumps(manifest, indent=2))
                for cap in self._store.all():
                    if cap.image_data:
                        zf.writestr(f"assets/{cap.id}.png", cap.image_data)

            self._current_project_path = path
            self.setWindowTitle(f"ROCA Animator — {os.path.basename(path)}")
            self._status.showMessage(f"Saved: {path}")
        except OSError as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    def _load_roca(self, path: str):
        try:
            with zipfile.ZipFile(path, "r") as zf:
                manifest = json.loads(zf.read("manifest.json").decode("utf-8"))

                self._store = CapsuleStore()
                self._store.from_dict({
                    "capsules": manifest.get("capsules", []),
                    "edges": manifest.get("edges", {}),
                })

                # Load image data from assets/
                for cap in self._store.all():
                    asset_name = f"assets/{cap.id}.png"
                    if asset_name in zf.namelist():
                        cap.image_data = zf.read(asset_name)

                self._timeline.clear()
                for entry in manifest.get("timeline", []):
                    cid = entry.get("capsule_id")
                    idx = self._timeline.add_slot(cid)
                    if cid:
                        c = self._store.get(cid)
                        if c and c.image_data:
                            self._timeline.load_thumbnail(idx, c.image_data)

            self._current_project_path = path
            self.setWindowTitle(f"ROCA Animator — {os.path.basename(path)}")
            self._orbital.refresh(self._store.all())
            self._status.showMessage(f"Opened: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Open Error", str(exc))

    # ------------------------------------------------------------------
    # Orbital map → timeline
    # ------------------------------------------------------------------

    def _on_capsule_dragged(self, cid: str):
        cap = self._store.get(cid)
        if cap is None:
            return
        if cap.kind == CapsuleKind.UNASSIGNED:
            self._status.showMessage(f"Capsule '{cap.name}' is UNASSIGNED — not added to timeline.")
            return

        idx = self._timeline.current_frame()
        # Agreement check with previous frame
        if idx > 0:
            prev_slot = self._timeline.get_slot(idx - 1)
            if prev_slot and prev_slot.capsule_id:
                score = self._store.agreement_score(prev_slot.capsule_id, cid)
                if score < 1.0:
                    prev_cap = self._store.get(prev_slot.capsule_id)
                    dlg = AgreementWarningDialog(
                        self,
                        prev_cap.name if prev_cap else "?",
                        cap.name,
                    )
                    if dlg.exec() != AgreementWarningDialog.DialogCode.Accepted:
                        return

        self._use_capsule(cap)
        if idx < self._timeline.slot_count():
            self._timeline.set_capsule(idx, cid)
            if cap.image_data:
                self._timeline.load_thumbnail(idx, cap.image_data)
        else:
            new_idx = self._timeline.add_slot(cid)
            if cap.image_data:
                self._timeline.load_thumbnail(new_idx, cap.image_data)

        self._orbital.refresh(self._store.all())

    def _on_frame_selected(self, index: int):
        slot = self._timeline.get_slot(index)
        if slot and slot.capsule_id:
            cap = self._store.get(slot.capsule_id)
            if cap and cap.image_data:
                self._canvas.load_image(cap.image_data)

    def _on_frame_file_dropped(self, index: int, path: str):
        """File dropped onto a timeline frame slot — import as pose capsule."""
        data = _png_bytes_from_file(path)
        if data is None:
            return
        h = image_hash(data)
        name = f"pose_{h[:8]}"
        cid = make_capsule_id(CapsuleKind.POSE, name)
        if self._store.get(cid) is None:
            cap = Capsule(
                id=cid,
                kind=CapsuleKind.POSE,
                name=name,
                image_data=data,
                asset_hash=h,
                asset_path=path,
            )
            self._store.add(cap)
        else:
            cap = self._store.get(cid)

        self._use_capsule(cap)
        # Ensure slot exists
        while self._timeline.slot_count() <= index:
            self._timeline.add_slot()
        self._timeline.set_capsule(index, cid)
        self._timeline.load_thumbnail(index, data)
        self._orbital.refresh(self._store.all())

    def _on_canvas_file_dropped(self, path: str):
        """File dropped onto canvas — emit signal only, do NOT add to timeline."""
        self._status.showMessage(f"File dropped onto canvas: {path}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _use_capsule(self, cap: Capsule):
        cap.use_count += 1
        cap.last_used_at = time.time()
        cap.orbit_score = _orbit_score_from_use(cap.use_count)

    def _png_bytes_to_ndarray(self, data: bytes) -> Optional[np.ndarray]:
        if _CV2_AVAILABLE:
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
        # Fallback: decode via QImage
        qimg = QImage()
        qimg.loadFromData(data, "PNG")
        if qimg.isNull():
            return None
        return _qimage_to_ndarray(qimg)
