import sys

from PyQt6.QtWidgets import QApplication

from gui.main_window import MainWindow, _DARK_STYLE


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("ROCA Animator")
    app.setOrganizationName("ROCA")
    app.setStyleSheet(_DARK_STYLE)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
