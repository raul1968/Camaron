#!/usr/bin/env python3
"""
ROCA Oracle — GUI Launch File with Chat & Orbital Visualizer
=============================================================
Complete GUI application integrating:
  • Chat interface with ROCA kernel routing
  • Real-time orbital capsule visualizer
  • Knowledge base management
  • Timeline replay viewer
  • Autonomous cycle controls

Creates all required directories on startup.

Run: python3 roca_oracle_gui.py
"""

import json
import math
import os
import re
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# ---------------------------------------------------------------------------
# Ensure the roca_unified_kernel is importable
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from roca_kernel import (
    bootstrap_kernel,
    Capsule,
    CapsuleKind,
    LanePosition,
    LANE_BOUNDARIES,
    ORBIT_LEVEL_TO_LANE,
    ROCAKernel,
    DEFAULT_KNOWLEDGE_BASE_PATH,
    DEFAULT_TEMPORAL_STORE_PATH,
    JSON_DIR,
)

# ---------------------------------------------------------------------------
# Directory Setup — create everything needed
# ---------------------------------------------------------------------------

def ensure_directories() -> Dict[str, Path]:
    """Create all required directories and return their paths."""
    dirs = {
        "project_root": SCRIPT_DIR,
        "json_dir": SCRIPT_DIR / "Json",
        "ingest_dir": SCRIPT_DIR / "Json" / "ingest",
        "backup_dir": SCRIPT_DIR / "Json" / "backups",
        "timeline_dir": SCRIPT_DIR / "Json" / "timeline",
        "roca_docs": SCRIPT_DIR / "Json" / "roca_docs",
        "source_registry": SCRIPT_DIR / "Json" / "source_registry",
        "data_dir": SCRIPT_DIR / "data",
        "logs_dir": SCRIPT_DIR / "logs",
        "exports_dir": SCRIPT_DIR / "exports",
        "capsule_snapshots": SCRIPT_DIR / "Json" / "capsule_snapshots",
    }
    
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
    
    # Create initial data files if they don't exist
    initial_capsules_path = dirs["data_dir"] / "initial_capsules.json"
    if not initial_capsules_path.exists():
        _create_initial_capsules_file(initial_capsules_path)
    
    return dirs


def _create_initial_capsules_file(path: Path) -> None:
    """Create a seed capsule file with diverse capsules for merging."""
    seed_data = {
        "capsules": [
            {
                "id": "topic_python_basics",
                "kind": "topic",
                "name": "Python Basics",
                "content": {
                    "keywords": ["python", "variables", "functions", "loops", "syntax", "types"],
                    "description": "Fundamental Python programming concepts including variables, functions, loops, and basic syntax."
                },
                "gravity_score": 0.4,
                "orbit_level": 2
            },
            {
                "id": "topic_python_advanced",
                "kind": "topic",
                "name": "Python Advanced",
                "content": {
                    "keywords": ["python", "decorators", "generators", "metaclasses", "async", "context managers"],
                    "description": "Advanced Python features including decorators, generators, metaclasses, and async programming."
                },
                "gravity_score": 0.35,
                "orbit_level": 2
            },
            {
                "id": "topic_python_functions",
                "kind": "topic",
                "name": "Python Functions",
                "content": {
                    "keywords": ["python", "functions", "lambda", "closures", "arguments", "decorators"],
                    "description": "Function definition and usage in Python including lambda expressions and closures."
                },
                "gravity_score": 0.3,
                "orbit_level": 2
            },
            {
                "id": "topic_capsule_networks",
                "kind": "topic",
                "name": "Capsule Networks",
                "content": {
                    "keywords": ["capsule", "routing", "agreement", "hinton", "pose", "transformation"],
                    "description": "Capsule network theory: dynamic routing, pose matrices, and part-whole hierarchies."
                },
                "gravity_score": 0.45,
                "orbit_level": 1
            },
            {
                "id": "topic_roca_architecture",
                "kind": "topic",
                "name": "ROCA Architecture",
                "content": {
                    "keywords": ["roca", "capsule", "orbital", "gravity", "routing", "deterministic"],
                    "description": "Routed Orbital Capsule Architecture: deterministic capsule-based cognitive engine."
                },
                "gravity_score": 0.5,
                "orbit_level": 1
            },
            {
                "id": "topic_deterministic_algorithms",
                "kind": "topic",
                "name": "Deterministic Algorithms",
                "content": {
                    "keywords": ["deterministic", "algorithm", "routing", "consensus", "hash", "embedding"],
                    "description": "Deterministic algorithms for routing, consensus, and embedding generation."
                },
                "gravity_score": 0.38,
                "orbit_level": 2
            },
            {
                "id": "skill_code_generation",
                "kind": "skill",
                "name": "Code Generation",
                "content": {
                    "keywords": ["code", "generate", "python", "template", "function", "class"],
                    "description": "Skill: Generate Python code from natural language specifications."
                },
                "gravity_score": 0.3,
                "orbit_level": 2
            },
            {
                "id": "skill_debugging",
                "kind": "skill",
                "name": "Debugging",
                "content": {
                    "keywords": ["debug", "error", "traceback", "fix", "analyze", "exception"],
                    "description": "Skill: Analyze and fix Python code errors using traceback analysis."
                },
                "gravity_score": 0.3,
                "orbit_level": 2
            },
            {
                "id": "memory_session_log",
                "kind": "memory",
                "name": "Session Log",
                "content": {
                    "keywords": ["session", "log", "history", "conversation", "tracking"],
                    "description": "Memory: Tracks conversation history and session metadata."
                },
                "gravity_score": 0.4,
                "orbit_level": 2
            },
            {
                "id": "workflow_code_review",
                "kind": "workflow",
                "name": "Code Review Workflow",
                "content": {
                    "keywords": ["code", "review", "workflow", "quality", "standards", "lint"],
                    "description": "Workflow: Systematic code review process with quality checks."
                },
                "gravity_score": 0.3,
                "orbit_level": 2
            },
        ]
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(seed_data, f, indent=2)


# ---------------------------------------------------------------------------
# GUI Application
# ---------------------------------------------------------------------------

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTextEdit, QPushButton, QLabel, QSplitter, QListWidget,
        QListWidgetItem, QTabWidget, QScrollArea, QFrame, QSlider,
        QSpinBox, QCheckBox, QComboBox, QGroupBox, QGridLayout,
        QProgressBar, QStatusBar, QToolBar, QMenuBar, QMenu,
    )
    from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
    from PyQt6.QtGui import (
        QFont, QTextCursor, QColor, QPainter, QBrush, QPen,
        QRadialGradient, QPainterPath, QAction, QIcon, QFontMetrics,
    )
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    print("PyQt6 not found. Install with: pip install PyQt6")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Orbital Visualizer Widget
# ---------------------------------------------------------------------------

class OrbitalVisualizer(QWidget):
    """Real-time orbital capsule visualization widget.
    
    Renders capsules as orbiting nodes around a central nucleus,
    with orbit lanes, gravity-based sizing, and interactive tooltips.
    """
    
    # Lane colors (Build Guide orbit levels)
    LANE_COLORS = {
        0: QColor(255, 215, 0, 200),     # Gold — Core cognition
        1: QColor(100, 200, 255, 200),    # Blue — Active reasoning
        2: QColor(150, 255, 150, 200),    # Green — Working memory
        3: QColor(255, 150, 150, 200),    # Red — Raw ingestion
    }
    
    LANE_LABELS = {
        0: "Core",
        1: "Active",
        2: "Working",
        3: "Ingest",
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.capsules: Dict[str, Capsule] = {}
        self.selected_capsule_id: Optional[str] = None
        self.highlight_ids: List[str] = []
        self.animation_phase: float = 0.0
        self.show_labels: bool = True
        self.show_trails: bool = False
        
        # Mouse tracking for tooltips
        self.setMouseTracking(True)
        self._hovered_capsule_id: Optional[str] = None
        self._capsule_positions: Dict[str, Tuple[float, float]] = {}
        
        # Animation timer
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._animate)
        self.anim_timer.start(50)  # 20 fps
        
        self.setMinimumSize(400, 400)
        self.setToolTip("Orbital Capsule Network — hover for details")
    
    def update_capsules(self, capsules: Dict[str, Capsule]) -> None:
        """Update the capsule data for rendering."""
        self.capsules = capsules
        self.update()
    
    def highlight_capsules(self, ids: List[str]) -> None:
        """Highlight specific capsules (e.g., from query results)."""
        self.highlight_ids = ids
        self.update()
    
    def select_capsule(self, capsule_id: Optional[str]) -> None:
        """Select a capsule for detailed view."""
        self.selected_capsule_id = capsule_id
        self.update()
    
    def _animate(self) -> None:
        """Advance animation phase for orbital motion."""
        self.animation_phase += 0.01
        if self.animation_phase > 2 * math.pi:
            self.animation_phase -= 2 * math.pi
        self.update()
    
    def paintEvent(self, event) -> None:
        """Render the orbital visualization."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(15, 15, 25))
        
        center_x = self.width() / 2
        center_y = self.height() / 2
        max_radius = min(center_x, center_y) * 0.85
        
        # Draw orbit lanes
        self._draw_lanes(painter, center_x, center_y, max_radius)
        
        # Draw nucleus glow
        self._draw_nucleus(painter, center_x, center_y)
        
        # Calculate and draw capsules
        self._capsule_positions.clear()
        if self.capsules:
            self._draw_capsules(painter, center_x, center_y, max_radius)
        
        # Draw legend
        self._draw_legend(painter)
        
        painter.end()
    
    def _draw_lanes(self, painter: QPainter, cx: float, cy: float, max_r: float) -> None:
        """Draw orbital lane rings."""
        for level in range(4):
            bounds = LANE_BOUNDARIES[ORBIT_LEVEL_TO_LANE[level]]
            inner_r = max_r * bounds[0]
            outer_r = max_r * bounds[1]
            mid_r = (inner_r + outer_r) / 2
            
            color = self.LANE_COLORS[level]
            color.setAlpha(40)
            
            pen = QPen(color, 1, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            
            # Draw lane circle
            painter.drawEllipse(int(cx - mid_r), int(cy - mid_r), 
                               int(mid_r * 2), int(mid_r * 2))
            
            # Lane label
            label_color = self.LANE_COLORS[level]
            label_color.setAlpha(120)
            painter.setPen(label_color)
            font = QFont("Monospace", 8)
            painter.setFont(font)
            painter.drawText(int(cx + mid_r + 5), int(cy), self.LANE_LABELS[level])
    
    def _draw_nucleus(self, painter: QPainter, cx: float, cy: float) -> None:
        """Draw the central nucleus."""
        # Gradient glow
        gradient = QRadialGradient(cx, cy, 30)
        gradient.setColorAt(0, QColor(255, 255, 200, 200))
        gradient.setColorAt(0.3, QColor(255, 200, 50, 150))
        gradient.setColorAt(0.7, QColor(255, 150, 0, 50))
        gradient.setColorAt(1.0, QColor(255, 100, 0, 0))
        
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(int(cx - 20), int(cy - 20), 40, 40)
        
        # Label
        painter.setPen(QColor(255, 255, 255, 200))
        font = QFont("Monospace", 9, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(int(cx - 25), int(cy + 5), "ROCA")
    
    def _draw_capsules(self, painter: QPainter, cx: float, cy: float, max_r: float) -> None:
        """Draw all capsules in their orbital positions."""
        active_capsules = {
            cid: cap for cid, cap in self.capsules.items()
            if not cap.merged_into and not cap.hibernating
        }
        
        for capsule_id, capsule in active_capsules.items():
            # Get lane bounds
            bounds = LANE_BOUNDARIES[capsule.lane]
            orbit_r = max_r * (bounds[0] + bounds[1]) / 2
            
            # Add some perturbation based on gravity (higher gravity = slightly inward)
            gravity_offset = (1.0 - capsule.gravity_score) * max_r * 0.05
            orbit_r -= gravity_offset
            
            # Calculate position
            angle = capsule.orbit_angle + self.animation_phase * (0.3 + capsule.gravity_score * 0.7)
            x = cx + orbit_r * math.cos(angle)
            y = cy + orbit_r * math.sin(angle)
            
            self._capsule_positions[capsule_id] = (x, y)
            
            # Size based on gravity
            base_size = 6
            size = base_size + capsule.gravity_score * 12
            
            # Color based on lane
            color = self.LANE_COLORS[capsule.orbit_level]
            
            # Highlight if selected or in highlight list
            is_selected = capsule_id == self.selected_capsule_id
            is_highlighted = capsule_id in self.highlight_ids
            is_hovered = capsule_id == self._hovered_capsule_id
            
            if is_selected:
                color = QColor(255, 255, 255, 255)
                size += 4
            elif is_highlighted:
                color = QColor(255, 255, 100, 255)
                size += 2
            elif is_hovered:
                color = color.lighter(150)
                size += 3
            
            # Draw capsule node
            painter.setBrush(QBrush(color))
            if is_selected or is_hovered:
                painter.setPen(QPen(QColor(255, 255, 255, 200), 2))
            else:
                painter.setPen(QPen(color.darker(150), 1))
            
            painter.drawEllipse(int(x - size/2), int(y - size/2), int(size), int(size))
            
            # Label
            if self.show_labels and (is_hovered or is_selected or size > 12):
                label_color = QColor(255, 255, 255, 180)
                painter.setPen(label_color)
                font = QFont("Monospace", 7)
                painter.setFont(font)
                short_name = capsule.name[:15] + ".." if len(capsule.name) > 15 else capsule.name
                painter.drawText(int(x + size/2 + 3), int(y + 3), short_name)
    
    def _draw_legend(self, painter: QPainter) -> None:
        """Draw the orbit legend."""
        painter.setPen(QColor(200, 200, 200, 150))
        font = QFont("Monospace", 7)
        painter.setFont(font)
        
        y_start = self.height() - 80
        x_start = 10
        
        for level in range(4):
            color = self.LANE_COLORS[level]
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(color.darker(150), 1))
            painter.drawEllipse(x_start, y_start + level * 18, 10, 10)
            
            painter.setPen(QColor(200, 200, 200, 150))
            painter.drawText(x_start + 15, y_start + level * 18 + 10, 
                           f"Orbit {level}: {self.LANE_LABELS[level]}")
        
        # Capsule count
        active = sum(1 for c in self.capsules.values() if not c.merged_into and not c.hibernating)
        hibernating = sum(1 for c in self.capsules.values() if c.hibernating)
        painter.drawText(x_start, y_start + 85, 
                        f"Capsules: {active} active | {hibernating} hibernating")
    
    def mouseMoveEvent(self, event) -> None:
        """Handle mouse movement for hover detection."""
        mouse_x = event.position().x()
        mouse_y = event.position().y()
        
        self._hovered_capsule_id = None
        for capsule_id, (cx, cy) in self._capsule_positions.items():
            dist = math.sqrt((mouse_x - cx)**2 + (mouse_y - cy)**2)
            if dist < 15:
                self._hovered_capsule_id = capsule_id
                break
        
        if self._hovered_capsule_id:
            cap = self.capsules.get(self._hovered_capsule_id)
            if cap:
                self.setToolTip(
                    f"{cap.name}\n"
                    f"Kind: {cap.kind.value}\n"
                    f"Orbit: {cap.orbit_level} ({cap.lane.value})\n"
                    f"Gravity: {cap.gravity_score:.3f}\n"
                    f"Agreement: {cap.agreement_score:.3f}\n"
                    f"Usage: {cap.usage_count}"
                )
        
        self.update()
    
    def mousePressEvent(self, event) -> None:
        """Handle mouse click for selection."""
        if self._hovered_capsule_id:
            self.selected_capsule_id = self._hovered_capsule_id
            self.update()


# ---------------------------------------------------------------------------
# Chat Widget
# ---------------------------------------------------------------------------

class ChatWidget(QWidget):
    """Chat interface with message history and input."""
    
    message_sent = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self.message_history: List[Dict[str, str]] = []
    
    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Chat history
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a2e;
                color: #e0e0e0;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Monospace', 'Courier New';
                font-size: 12px;
            }
        """)
        layout.addWidget(self.chat_display, 1)
        
        # Input area
        input_frame = QFrame()
        input_frame.setStyleSheet("""
            QFrame {
                background-color: #16213e;
                border: 1px solid #333;
                border-radius: 4px;
            }
        """)
        input_layout = QVBoxLayout(input_frame)
        input_layout.setContentsMargins(5, 5, 5, 5)
        
        self.input_field = QTextEdit()
        self.input_field.setMaximumHeight(80)
        self.input_field.setPlaceholderText(
            "Ask about ROCA, run 'autonomy status', or say 'ingest codebase'..."
        )
        self.input_field.setStyleSheet("""
            QTextEdit {
                background-color: #0f0f23;
                color: #e0e0e0;
                border: none;
                font-family: 'Monospace', 'Courier New';
                font-size: 12px;
            }
        """)
        input_layout.addWidget(self.input_field)
        
        # Send button row
        btn_layout = QHBoxLayout()
        
        self.send_button = QPushButton("▶ Send")
        self.send_button.clicked.connect(self._send_message)
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #0f3460;
                color: #e0e0e0;
                border: 1px solid #1a5276;
                border-radius: 4px;
                padding: 6px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1a5276;
            }
        """)
        btn_layout.addWidget(self.send_button)
        
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self._clear_chat)
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: #e0e0e0;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #555;
            }
        """)
        btn_layout.addWidget(self.clear_button)
        
        btn_layout.addStretch()
        
        input_layout.addLayout(btn_layout)
        layout.addWidget(input_frame)
        
        # Connect Enter key
        self.input_field.installEventFilter(self)
    
    def eventFilter(self, obj, event) -> bool:
        if obj == self.input_field and event.type() == event.Type.KeyPress:
            if event.key() == Qt.Key.Key_Return and event.modifiers() == Qt.KeyboardModifier.NoModifier:
                self._send_message()
                return True
        return super().eventFilter(obj, event)
    
    def _send_message(self) -> None:
        """Send the current message."""
        text = self.input_field.toPlainText().strip()
        if not text:
            return
        
        self.add_message("You", text, "#4fc3f7")
        self.input_field.clear()
        self.message_sent.emit(text)
    
    def add_message(self, sender: str, message: str, color: str = "#e0e0e0") -> None:
        """Add a message to the chat display."""
        self.message_history.append({"sender": sender, "content": message})
        
        # Format message
        formatted = f'<p><b style="color:{color}">{sender}:</b><br>'
        # Pre-format code blocks
        msg = message
        msg = msg.replace("```python", '<pre style="background:#0a0a1a; padding:8px; border-radius:4px;">')
        msg = msg.replace("```", '</pre>')
        msg = msg.replace("\n", "<br>")
        formatted += f'{msg}</p><hr style="border-color:#333">'
        
        self.chat_display.append(formatted)
        self.chat_display.moveCursor(QTextCursor.MoveOperation.End)
    
    def _clear_chat(self) -> None:
        """Clear chat history."""
        self.chat_display.clear()
        self.message_history.clear()
        self.add_message("System", "Chat cleared.", "#888")


# ---------------------------------------------------------------------------
# Capsule Inspector Widget
# ---------------------------------------------------------------------------

class CapsuleInspector(QWidget):
    """Detailed capsule information panel."""
    
    capsule_selected = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self.capsules: Dict[str, Capsule] = {}
    
    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Search bar
        self.search_input = QTextEdit()
        self.search_input.setMaximumHeight(30)
        self.search_input.setPlaceholderText("Search capsules...")
        self.search_input.setStyleSheet("""
            QTextEdit {
                background-color: #0f0f23;
                color: #e0e0e0;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 4px;
            }
        """)
        layout.addWidget(self.search_input)
        
        # Capsule list
        self.capsule_list = QListWidget()
        self.capsule_list.setStyleSheet("""
            QListWidget {
                background-color: #1a1a2e;
                color: #e0e0e0;
                border: 1px solid #333;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 4px;
                border-bottom: 1px solid #2a2a4a;
            }
            QListWidget::item:selected {
                background-color: #0f3460;
            }
            QListWidget::item:hover {
                background-color: #16213e;
            }
        """)
        self.capsule_list.itemClicked.connect(self._on_capsule_clicked)
        layout.addWidget(self.capsule_list)
        
        # Detail panel
        self.detail_display = QTextEdit()
        self.detail_display.setReadOnly(True)
        self.detail_display.setStyleSheet("""
            QTextEdit {
                background-color: #0f0f23;
                color: #e0e0e0;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Monospace';
                font-size: 11px;
            }
        """)
        layout.addWidget(self.detail_display)
    
    def update_capsules(self, capsules: Dict[str, Capsule]) -> None:
        """Update the capsule list."""
        self.capsules = capsules
        self.capsule_list.clear()
        
        # Sort by gravity (most salient first)
        sorted_caps = sorted(
            [c for c in capsules.values() if not c.merged_into],
            key=lambda c: c.gravity_score,
            reverse=True
        )
        
        for capsule in sorted_caps:
            status = "💤" if capsule.hibernating else "●"
            item_text = (
                f"{status} [{capsule.orbit_level}] {capsule.name} "
                f"(grav: {capsule.gravity_score:.2f})"
            )
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, capsule.id)
            
            # Color based on orbit level
            colors = {
                0: QColor(255, 215, 0),     # Gold
                1: QColor(100, 200, 255),    # Blue
                2: QColor(150, 255, 150),    # Green
                3: QColor(255, 150, 150),    # Red
            }
            color = colors.get(capsule.orbit_level, QColor(200, 200, 200))
            item.setForeground(color)
            
            self.capsule_list.addItem(item)
    
    def _on_capsule_clicked(self, item: QListWidgetItem) -> None:
        """Show capsule details when clicked."""
        capsule_id = item.data(Qt.ItemDataRole.UserRole)
        capsule = self.capsules.get(capsule_id)
        if not capsule:
            return
        
        self.capsule_selected.emit(capsule_id)
        
        # Build detail text
        details = f"""
<b style="color:gold">Name:</b> {capsule.name}
<b style="color:gold">ID:</b> {capsule.id}
<b style="color:gold">Kind:</b> {capsule.kind.value}
<b style="color:gold">Orbit Level:</b> {capsule.orbit_level} ({capsule.lane.value})
<b style="color:gold">Gravity Score:</b> {capsule.gravity_score:.4f}
<b style="color:gold">Agreement Score:</b> {capsule.agreement_score:.4f}
<b style="color:gold">Confidence:</b> {capsule.confidence:.4f}
<b style="color:gold">Usage Count:</b> {capsule.usage_count}
<b style="color:gold">Hibernating:</b> {capsule.hibernating}
<b style="color:gold">Merged Into:</b> {capsule.merged_into or 'None'}
<b style="color:gold">Shadows:</b> {', '.join(capsule.shadows) if capsule.shadows else 'None'}
<b style="color:gold">Created:</b> {capsule.created_at.strftime('%Y-%m-%d %H:%M:%S')}
<b style="color:gold">Last Used:</b> {capsule.last_used_at.strftime('%Y-%m-%d %H:%M:%S')}

<b style="color:#4fc3f7">Content Keywords:</b>
{', '.join(capsule.content.get('keywords', [])[:20])}

<b style="color:#4fc3f7">Description:</b>
{capsule.content.get('description', 'No description')}
"""
        self.detail_display.setHtml(details)


# ---------------------------------------------------------------------------
# Main Oracle GUI
# ---------------------------------------------------------------------------

class OracleGUI(QMainWindow):
    """Main ROCA Oracle GUI application."""
    
    def __init__(self):
        super().__init__()
        
        # Initialize directories
        self.dirs = ensure_directories()
        
        # Initialize kernel
        self.kernel = self._init_kernel()
        
        # Auto-cycle timer
        self.auto_cycle_timer = QTimer(self)
        self.auto_cycle_timer.timeout.connect(self._auto_cycle)
        self.auto_cycle_enabled = False
        
        # GUI state
        self.query_results_ids: List[str] = []
        
        # Build UI
        self._init_ui()
        
        # Initial status update
        self._update_all_displays()
        
        # Welcome message
        self._show_welcome()
    
    def _init_kernel(self) -> ROCAKernel:
        """Initialize the ROCA kernel with seed capsules."""
        kernel = bootstrap_kernel(load_existing=True)
        
        # Load additional seed capsules from data file
        seed_path = self.dirs["data_dir"] / "initial_capsules.json"
        if seed_path.exists():
            try:
                with open(seed_path, "r", encoding="utf-8") as f:
                    seed_data = json.load(f)
                
                from roca_kernel import deterministic_embed
                for cap_data in seed_data.get("capsules", []):
                    if cap_data["id"] not in kernel.capsules:
                        emb = deterministic_embed(
                            cap_data["name"] + " " + 
                            " ".join(cap_data.get("content", {}).get("keywords", []))
                        )
                        capsule = Capsule(
                            id=cap_data["id"],
                            kind=CapsuleKind(cap_data.get("kind", "topic")),
                            name=cap_data["name"],
                            content=cap_data.get("content", {}),
                            embedding=emb,
                            gravity_score=cap_data.get("gravity_score", 0.3),
                            orbit_level=cap_data.get("orbit_level", 2),
                            lane=ORBIT_LEVEL_TO_LANE.get(
                                cap_data.get("orbit_level", 2), LanePosition.OUTER
                            ),
                        )
                        kernel.capsules[capsule.id] = capsule
                
                kernel.store.save(kernel.capsules)
            except Exception as e:
                print(f"Error loading seed capsules: {e}")
        
        return kernel
    
    def _init_ui(self) -> None:
        """Build the complete GUI."""
        self.setWindowTitle("ROCA Oracle — Capsule Network Interface")
        self.setGeometry(80, 80, 1400, 850)
        
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0a0a1a;
            }
            QWidget {
                color: #e0e0e0;
                font-family: 'Monospace', 'Courier New';
            }
            QSplitter::handle {
                background-color: #333;
                width: 2px;
            }
            QTabWidget::pane {
                background-color: #1a1a2e;
                border: 1px solid #333;
            }
            QTabBar::tab {
                background-color: #16213e;
                color: #e0e0e0;
                padding: 8px 16px;
                border: 1px solid #333;
                border-radius: 4px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #0f3460;
            }
        """)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        
        # Toolbar
        self._create_toolbar()
        
        # Main splitter (chat + visualizer)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: Chat + Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Chat widget
        self.chat_widget = ChatWidget()
        self.chat_widget.message_sent.connect(self._handle_chat_message)
        left_layout.addWidget(self.chat_widget, 2)
        
        # Control tabs below chat
        control_tabs = QTabWidget()
        control_tabs.setMaximumHeight(250)
        
        # Capsule inspector tab
        self.capsule_inspector = CapsuleInspector()
        self.capsule_inspector.capsule_selected.connect(
            lambda cid: self.orbital_visualizer.select_capsule(cid)
        )
        control_tabs.addTab(self.capsule_inspector, "Capsules")
        
        # Query results tab
        self.query_results_display = QTextEdit()
        self.query_results_display.setReadOnly(True)
        self.query_results_display.setStyleSheet("""
            QTextEdit {
                background-color: #0f0f23;
                color: #e0e0e0;
                border: 1px solid #333;
                font-family: 'Monospace';
                font-size: 11px;
            }
        """)
        control_tabs.addTab(self.query_results_display, "Query Results")
        
        # Timeline tab
        self.timeline_display = QTextEdit()
        self.timeline_display.setReadOnly(True)
        self.timeline_display.setStyleSheet("""
            QTextEdit {
                background-color: #0f0f23;
                color: #e0e0e0;
                border: 1px solid #333;
                font-family: 'Monospace';
                font-size: 11px;
            }
        """)
        control_tabs.addTab(self.timeline_display, "Timeline")
        
        left_layout.addWidget(control_tabs)
        main_splitter.addWidget(left_panel)
        
        # Right panel: Orbital Visualizer + Status
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Orbital visualizer
        self.orbital_visualizer = OrbitalVisualizer()
        right_layout.addWidget(self.orbital_visualizer, 3)
        
        # Status panel
        status_frame = QFrame()
        status_frame.setStyleSheet("""
            QFrame {
                background-color: #16213e;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        status_layout = QGridLayout(status_frame)
        status_layout.setContentsMargins(8, 8, 8, 8)
        
        self.status_labels = {}
        status_items = [
            (0, 0, "Total Capsules:", "total_capsules"),
            (0, 1, "Active:", "active_capsules"),
            (0, 2, "Hibernating:", "hibernating"),
            (1, 0, "Cycles Run:", "cycles"),
            (1, 1, "Merges:", "merges"),
            (1, 2, "Pressure:", "pressure"),
            (2, 0, "Orbit 0:", "orbit_0"),
            (2, 1, "Orbit 1:", "orbit_1"),
            (2, 2, "Orbit 2-3:", "orbit_23"),
        ]
        
        for row, col, label, key in status_items:
            lbl = QLabel(f"{label} --")
            lbl.setStyleSheet("color: #888; font-size: 11px;")
            val = QLabel("--")
            val.setStyleSheet("color: #4fc3f7; font-size: 11px; font-weight: bold;")
            self.status_labels[key] = val
            status_layout.addWidget(lbl, row, col * 2)
            status_layout.addWidget(val, row, col * 2 + 1)
        
        right_layout.addWidget(status_frame)
        main_splitter.addWidget(right_panel)
        
        main_splitter.setSizes([700, 700])
        main_layout.addWidget(main_splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #0a0a1a;
                color: #888;
                border-top: 1px solid #333;
            }
        """)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("ROCA Oracle ready. Type 'help' for commands.")
        
        # Refresh timer
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self._update_all_displays)
        self.refresh_timer.start(3000)  # Update every 3 seconds
    
    def _create_toolbar(self) -> None:
        """Create the top toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setStyleSheet("""
            QToolBar {
                background-color: #16213e;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 4px;
                spacing: 4px;
            }
            QToolButton {
                background-color: #0f3460;
                color: #e0e0e0;
                border: 1px solid #1a5276;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QToolButton:hover {
                background-color: #1a5276;
            }
        """)
        self.addToolBar(toolbar)
        
        # Run cycle button
        cycle_btn = toolbar.addAction("🔄 Run Cycle")
        cycle_btn.triggered.connect(self._run_cycle)
        cycle_btn.setToolTip("Execute one cognition cycle")
        
        # Auto cycle toggle
        self.auto_cycle_btn = toolbar.addAction("⏯ Auto Cycle")
        self.auto_cycle_btn.setCheckable(True)
        self.auto_cycle_btn.triggered.connect(self._toggle_auto_cycle)
        self.auto_cycle_btn.setToolTip("Toggle autonomous cycling every 10s")
        
        toolbar.addSeparator()
        
        # Ingest buttons
        ingest_code_btn = toolbar.addAction("📥 Ingest Codebase")
        ingest_code_btn.triggered.connect(lambda: self._ingest_command("ingest codebase"))
        ingest_code_btn.setToolTip("Scan and ingest code files as capsules")
        
        ingest_papers_btn = toolbar.addAction("📄 Ingest Papers")
        ingest_papers_btn.triggered.connect(lambda: self._ingest_command("ingest papers"))
        ingest_papers_btn.setToolTip("Ingest research papers from JSON")
        
        toolbar.addSeparator()
        
        # View buttons
        status_btn = toolbar.addAction("📊 Status")
        status_btn.triggered.connect(lambda: self._handle_chat_message("autonomy status"))
        status_btn.setToolTip("Show full system status")
        
        timeline_btn = toolbar.addAction("📅 Timeline")
        timeline_btn.triggered.connect(self._show_timeline)
        timeline_btn.setToolTip("View timeline frames")
        
        toolbar.addSeparator()
        
        # Save button
        save_btn = toolbar.addAction("💾 Save")
        save_btn.triggered.connect(self._save_state)
        save_btn.setToolTip("Persist all capsules to disk")
        
        # Help button
        help_btn = toolbar.addAction("❓ Help")
        help_btn.triggered.connect(self._show_help)
        help_btn.setToolTip("Show available commands")
    
    def _handle_chat_message(self, message: str) -> None:
        """Process a chat message."""
        lower = message.lower().strip()
        
        # Command routing
        if lower in ("help", "?"):
            response = self._get_help_text()
        
        elif lower in ("status", "autonomy status", "system status"):
            response = self._get_status_text()
        
        elif lower.startswith("query ") or lower.startswith("search "):
            term = message.split(" ", 1)[1] if " " in message else message
            response = self._handle_query(term)
        
        elif lower.startswith("ingest "):
            response = self._handle_ingest(message)
        
        elif lower in ("run cycle", "cycle", "cognition cycle"):
            response = self._run_cycle()
        
        elif lower in ("save", "persist"):
            self._save_state()
            response = "✅ State saved to disk."
        
        elif lower in ("timeline", "show timeline", "frames"):
            response = self._show_timeline()
        
        elif lower in ("visualize", "show orbits", "orbit map"):
            response = "The orbital visualization is displayed in the right panel."
        
        else:
            # General query through kernel
            response = self._handle_query(message)
        
        self.chat_widget.add_message("Oracle", response, "#80cbc4")
        self._update_all_displays()
    
    def _get_help_text(self) -> str:
        """Return help text."""
        return """
<b style="color:gold">ROCA Oracle Commands:</b>

<b>Chat Commands:</b>
  • <b>help</b> — Show this help
  • <b>status</b> — Full system status report
  • <b>query &lt;term&gt;</b> — Search capsules by topic
  • <b>search &lt;term&gt;</b> — Same as query
  • <b>ingest codebase</b> — Scan and ingest code files
  • <b>ingest papers</b> — Load research paper data
  • <b>run cycle</b> — Execute one cognition cycle
  • <b>save</b> — Persist capsules to disk
  • <b>timeline</b> — View temporal frame history
  • <b>visualize</b> — View orbital map (right panel)

<b>Any other message</b> will be routed through the capsule 
network for semantic search and response generation.

<b>Toolbar buttons</b> provide quick access to common actions.
"""
    
    def _get_status_text(self) -> str:
        """Return formatted system status."""
        try:
            status = self.kernel.status()
            
            orbit_str = " | ".join(
                f"L{level}: {count}" for level, count in 
                sorted(status["orbit_distribution"].items())
            )
            
            kind_str = " | ".join(
                f"{kind}: {count}" for kind, count in 
                sorted(status["kind_distribution"].items())
            )
            
            timeline = status["timeline_stats"]
            routing = status["routing_stats"]
            
            return f"""
<b style="color:gold">═══ ROCA System Status ═══</b>

<b>Capsules:</b>
  • Total: {status['total_capsules']}
  • Active: {status['active_capsules']}
  • Hibernating: {status['hibernating']}

<b>Orbit Distribution:</b>
  {orbit_str}

<b>Kind Distribution:</b>
  {kind_str}

<b>Routing:</b>
  • Cycles run: {status['cycle_count']}
  • Total merges: {routing['merges_performed']}
  • Agreements computed: {routing['total_agreements_computed']}

<b>Timeline:</b>
  • Total frames: {timeline.get('total_frames', 0)}
  • Avg active/frame: {timeline.get('avg_active_capsules', 0):.1f}
  • Total merges: {timeline.get('total_merges', 0)}

<b>Resources:</b>
  • Max budget: {status['resource_stats'].get('max_budget', 'N/A')}
  • Pressure: {status['pressure']:.2f}
"""
        except Exception as e:
            return f"Error getting status: {e}"
    
    def _handle_query(self, query_text: str) -> str:
        """Handle a semantic query."""
        try:
            # Clean query
            for prefix in ("query ", "search ", "find "):
                if query_text.lower().startswith(prefix):
                    query_text = query_text[len(prefix):]
                    break
            
            # Run query through kernel
            results = self.kernel.query(query_text, top_k=8)
            
            # Store result IDs for highlighting in visualizer
            self.query_results_ids = [r["id"] for r in results]
            self.orbital_visualizer.highlight_capsules(self.query_results_ids)
            
            # Format results
            if not results:
                return f"No capsules found matching '{query_text}'."
            
            lines = [f"<b style='color:gold'>Query: '{query_text}' — {len(results)} results</b>\n"]
            
            for i, r in enumerate(results, 1):
                orbit_label = self.orbital_visualizer.LANE_LABELS.get(r['orbit_level'], '?')
                lines.append(
                    f"{i}. <b>[{orbit_label}]</b> {r['name']} "
                    f"(grav: {r['gravity']:.3f}, score: {r['score']:.3f})"
                )
                lines.append(f"   <span style='color:#888'>{r['content_summary']}</span>")
            
            # Auto-ingest the query as a topic for learning
            self.kernel.ingest_input(query_text, source="user_query")
            
            # Run a cycle to incorporate
            cycle_result = self.kernel.run_cycle()
            
            lines.append(f"\n<span style='color:#888'>Query ingested • Cycle {cycle_result['cycle']} complete</span>")
            
            return "\n".join(lines)
        
        except Exception as e:
            return f"Query error: {e}\n\n{traceback.format_exc()}"
    
    def _handle_ingest(self, message: str) -> str:
        """Handle ingestion requests."""
        try:
            if "codebase" in message.lower() or "code" in message.lower():
                # Scan current directory for code files
                code_extensions = {'.py', '.pyi', '.js', '.ts', '.json', '.yaml', '.yml', 
                                  '.md', '.txt', '.sh', '.sql', '.toml', '.ini'}
                ingested = 0
                
                for root, dirs, files in os.walk(self.dirs["project_root"]):
                    # Skip hidden and special directories
                    dirs[:] = [d for d in dirs if not d.startswith('.') 
                              and d not in {'__pycache__', 'venv', '.venv', 'node_modules',
                                           'Json', 'Backup', 'logs', 'exports'}]
                    
                    for filename in files:
                        ext = os.path.splitext(filename)[1].lower()
                        if ext in code_extensions:
                            filepath = os.path.join(root, filename)
                            try:
                                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                                    content = f.read()
                                if len(content) > 50:
                                    rel_path = os.path.relpath(filepath, self.dirs["project_root"])
                                    self.kernel.ingest_input(
                                        f"File: {rel_path}\n\n{content[:3000]}", 
                                        source=f"codebase:{rel_path}"
                                    )
                                    ingested += 1
                            except Exception:
                                pass
                
                # Run a cycle
                cycle_result = self.kernel.run_cycle()
                
                return f"""
<b style="color:#4fc3f7">✅ Codebase Ingestion Complete</b>

  • Files scanned: Scanned project directory
  • Files ingested: {ingested}
  • Cycle run: {cycle_result['cycle']}
  • Active capsules: {cycle_result['active_capsules']}
  • Merges this cycle: {cycle_result['merges_this_cycle']}

New code capsules are now in orbit 3 (raw ingestion).
They will migrate inward as they gain gravity through usage.
"""
            
            elif "paper" in message.lower() or "research" in message.lower():
                # Check for paper data
                papers_path = self.dirs["json_dir"] / "roca_papers.json"
                if papers_path.exists():
                    with open(papers_path, 'r', encoding='utf-8') as f:
                        papers = json.load(f)
                    
                    paper_list = papers if isinstance(papers, list) else papers.get("papers", [])
                    for paper in paper_list[:10]:  # Limit to 10
                        if isinstance(paper, dict):
                            title = paper.get("title", "Untitled")
                            abstract = paper.get("abstract", "")
                            key_ideas = paper.get("key_ideas", paper.get("key_concepts", []))
                            text = f"Paper: {title}\n\n{abstract}\n\nKey ideas: {'; '.join(key_ideas)}"
                            self.kernel.ingest_input(text, source=f"paper:{title[:40]}")
                    
                    cycle_result = self.kernel.run_cycle()
                    
                    return f"""
<b style="color:#4fc3f7">✅ Paper Ingestion Complete</b>

  • Papers ingested: {min(len(paper_list), 10)}
  • Research capsules created
  • Cycle run: {cycle_result['cycle']}
"""
                else:
                    return """
<b style="color:#ff6b6b">No paper data found.</b>

Place your research paper JSON file at:
  <b>Json/roca_papers.json</b>

Format:
  { "papers": [{"title": "...", "abstract": "...", "key_ideas": [...]}] }
"""
            
            else:
                # General text ingestion
                self.kernel.ingest_input(message, source="user")
                cycle_result = self.kernel.run_cycle()
                return f"✅ Text ingested • Cycle {cycle_result['cycle']} complete • {cycle_result['active_capsules']} active capsules"
        
        except Exception as e:
            return f"Ingestion error: {e}\n\n{traceback.format_exc()}"
    
    def _ingest_command(self, command: str) -> None:
        """Handle ingest from toolbar."""
        self.chat_widget.add_message("You", command, "#4fc3f7")
        response = self._handle_ingest(command)
        self.chat_widget.add_message("Oracle", response, "#80cbc4")
        self._update_all_displays()
    
    def _run_cycle(self) -> str:
        """Run a cognition cycle and return result."""
        try:
            result = self.kernel.run_cycle()
            self._update_all_displays()
            
            orbit_str = ", ".join(
                f"L{level}: {count}" for level, count in 
                sorted(result["orbit_distribution"].items())
            )
            
            return f"""
<b style="color:#4fc3f7">🔄 Cognition Cycle {result['cycle']} Complete</b>

  • Active capsules: {result['active_capsules']}
  • Merges this cycle: {result['merges_this_cycle']}
  • Max gravity: {result['max_gravity']:.3f}
  • Orbit distribution: {orbit_str}
  • Total capsules: {result['total_capsules']}
  • Hibernating: {result['hibernating']}
  • Elapsed: {result['elapsed_seconds']:.4f}s
  • Pressure: {result['pressure']:.2f}
"""
        except Exception as e:
            return f"Cycle error: {e}"
    
    def _show_timeline(self) -> str:
        """Show timeline frames."""
        try:
            stats = self.kernel.timeline.stats()
            frames = self.kernel.timeline.replay(-10)  # Last 10 frames
            
            lines = [f"<b style='color:gold'>═══ Timeline Frames (last {len(frames)}) ═══</b>\n"]
            
            for frame_data in frames:
                orbit_str = ", ".join(
                    f"L{level}: {count}" for level, count in 
                    sorted(frame_data["orbit_distribution"].items())
                )
                lines.append(
                    f"[{frame_data['timestamp'][:19]}] "
                    f"Active: {frame_data['active_count']} | "
                    f"Merges: {frame_data['merges_this_cycle']} | "
                    f"MaxG: {frame_data['max_gravity']:.2f} | "
                    f"Orbits: {orbit_str}"
                )
            
            lines.append(f"\nTotal frames: {stats['total_frames']}")
            lines.append(f"Avg active per frame: {stats.get('avg_active_capsules', 0):.1f}")
            
            return "\n".join(lines)
        except Exception as e:
            return f"Timeline error: {e}"
    
    def _save_state(self) -> None:
        """Save all state to disk."""
        try:
            self.kernel.store.save(self.kernel.capsules)
            self.status_bar.showMessage("✅ State saved to disk")
        except Exception as e:
            self.status_bar.showMessage(f"❌ Save error: {e}")
    
    def _toggle_auto_cycle(self) -> None:
        """Toggle autonomous cycling."""
        self.auto_cycle_enabled = self.auto_cycle_btn.isChecked()
        if self.auto_cycle_enabled:
            self.auto_cycle_timer.start(10000)  # Every 10 seconds
            self.status_bar.showMessage("🔄 Auto-cycle enabled (every 10s)")
        else:
            self.auto_cycle_timer.stop()
            self.status_bar.showMessage("⏸ Auto-cycle disabled")
    
    def _auto_cycle(self) -> None:
        """Autonomous cycle handler."""
        try:
            self.kernel.run_cycle()
            self._update_all_displays()
            self.status_bar.showMessage(
                f"🔄 Auto-cycle — {self.kernel.cycle_count} cycles run"
            )
        except Exception:
            pass
    
    def _update_all_displays(self) -> None:
        """Refresh all UI components."""
        # Update orbital visualizer
        self.orbital_visualizer.update_capsules(self.kernel.capsules)
        
        # Update capsule inspector
        self.capsule_inspector.update_capsules(self.kernel.capsules)
        
        # Update status labels
        try:
            status = self.kernel.status()
            
            self.status_labels["total_capsules"].setText(str(status["total_capsules"]))
            self.status_labels["active_capsules"].setText(str(status["active_capsules"]))
            self.status_labels["hibernating"].setText(str(status["hibernating"]))
            self.status_labels["cycles"].setText(str(status["cycle_count"]))
            self.status_labels["merges"].setText(str(status["routing_stats"]["merges_performed"]))
            self.status_labels["pressure"].setText(f"{status['pressure']:.2f}")
            
            orbit_dist = status["orbit_distribution"]
            self.status_labels["orbit_0"].setText(str(orbit_dist.get(0, 0)))
            self.status_labels["orbit_1"].setText(str(orbit_dist.get(1, 0)))
            self.status_labels["orbit_23"].setText(
                str(orbit_dist.get(2, 0) + orbit_dist.get(3, 0))
            )
        except Exception:
            pass
    
    def _show_welcome(self) -> None:
        """Display welcome message."""
        status = self.kernel.status()
        self.chat_widget.add_message(
            "Oracle",
            f"""
<b style="color:gold">🪐 ROCA Oracle — Online</b>

<b>System Ready:</b>
  • Capsules: {status['total_capsules']}
  • Orbit distribution: {status['orbit_distribution']}
  • Cycles available

<b>Type 'help' for commands or just ask a question.</b>
The orbital visualizer (right panel) shows all capsules
in real-time with their orbit levels and gravity scores.
""",
            "#80cbc4"
        )
    
    def _show_help(self) -> None:
        """Show help from toolbar."""
        self.chat_widget.add_message("You", "help", "#4fc3f7")
        self.chat_widget.add_message("Oracle", self._get_help_text(), "#80cbc4")
    
    def closeEvent(self, event) -> None:
        """Save state on close."""
        self._save_state()
        self.auto_cycle_timer.stop()
        self.refresh_timer.stop()
        event.accept()


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """Launch the ROCA Oracle GUI."""
    print("=" * 60)
    print("ROCA Oracle — GUI Launch")
    print("=" * 60)
    
    # Ensure directories
    dirs = ensure_directories()
    print(f"\nDirectories created/verified:")
    for name, path in dirs.items():
        print(f"  {name}: {path}")
    
    print(f"\nKnowledge base: {DEFAULT_KNOWLEDGE_BASE_PATH}")
    print(f"Timeline store: {DEFAULT_TEMPORAL_STORE_PATH}")
    
    # Launch GUI
    print("\nStarting GUI...")
    app = QApplication(sys.argv)
    app.setApplicationName("ROCA Oracle")
    
    window = OracleGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()