from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QLineEdit, QPushButton, QDialogButtonBox
from PyQt6.QtCore import Qt


class TransitionDialog(QDialog):
    """Dialog to configure the number of transition frames."""

    def __init__(self, parent=None, n_frames_default: int = 5):
        super().__init__(parent)
        self.setWindowTitle("Generate Transition")
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Number of intermediate frames:"))

        self._spin = QSpinBox()
        self._spin.setRange(1, 120)
        self._spin.setValue(n_frames_default)
        layout.addWidget(self._spin)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_n_frames(self) -> int:
        return self._spin.value()


class CycleDialog(QDialog):
    """Dialog to name a cycle and set loop count."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create Cycle")
        self.setModal(True)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Cycle name:"))
        self._name_edit = QLineEdit("cycle_1")
        layout.addWidget(self._name_edit)

        layout.addWidget(QLabel("Loop count:"))
        self._loop_spin = QSpinBox()
        self._loop_spin.setRange(1, 999)
        self._loop_spin.setValue(2)
        layout.addWidget(self._loop_spin)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_name(self) -> str:
        return self._name_edit.text().strip() or "cycle_1"

    def get_loop_count(self) -> int:
        return self._loop_spin.value()


class AgreementWarningDialog(QDialog):
    """Warns that two capsules have not been tested together."""

    def __init__(self, parent=None, capsule_a_name: str = "A", capsule_b_name: str = "B"):
        super().__init__(parent)
        self.setWindowTitle("Agreement Warning")
        self.setModal(True)

        layout = QVBoxLayout(self)
        msg = (
            f"Capsules <b>{capsule_a_name}</b> and <b>{capsule_b_name}</b> "
            "have a low agreement score.<br>They have not been used together "
            "before. Continue?"
        )
        label = QLabel(msg)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setWordWrap(True)
        layout.addWidget(label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Yes | QDialogButtonBox.StandardButton.No
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
