"""camaron.py — Camaron chatbot entry point.

CLI usage:
    python camaron.py

GUI usage (requires PyQt6):
    python camaron.py --gui

Camaron is a local-first knowledge chatbot.  It ingests text files from the
data/ directory, keeps a lightweight capsule memory, and answers questions
about those documents.  It also understands workspace-file queries and a
basic coding lane.

Under the hood Camaron reuses the DokuEntity engine from doku_main.py.  The
public interface (window title, prompt, branding) is updated to "Camaron".
"""

from __future__ import annotations

import argparse
import sys
from typing import List

# ---------------------------------------------------------------------------
# Try to import the underlying engine from doku_main.
# ---------------------------------------------------------------------------
try:
    from doku_main import DokuEntity, _HAS_QT, _safe_console_text
    if _HAS_QT:
        from doku_main import DokuChatWindow
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtGui import QFont
except Exception as _import_err:  # pragma: no cover
    print(f"[camaron] Could not import doku_main: {_import_err}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# CLI chatbot
# ---------------------------------------------------------------------------

_BANNER = r"""
  ______ ___  __  __  ___  ____  ___  _   _
 / ____// _ \ | \/ / / _ \|  _ \/ _ \| \ | |
| |    | |_| || /\ || |_| | |_) | | | |  \| |
| |____|  _  || |  || |_| |  _ <| |_| | |\  |
 \_____/_/ \_||_|  |_|\___/|_| \_\\___/|_| \_|

 Local-first knowledge chatbot  |  type /help for commands
"""


def run_cli() -> int:
    """Start the Camaron chatbot in terminal (REPL) mode."""
    entity = DokuEntity()
    boot_message = entity.bootstrap()

    print(_BANNER)
    print(_safe_console_text(boot_message))
    print()

    while True:
        try:
            user_text = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0

        if not user_text:
            continue

        if user_text.lower() in {"/quit", "quit", "exit"}:
            print("Goodbye.")
            return 0

        entity.log_turn("user", user_text)
        reply = entity.answer(user_text)
        entity.log_turn("camaron", reply)
        print(_safe_console_text(f"Camaron> {reply}\n"))


# ---------------------------------------------------------------------------
# GUI chatbot (PyQt6 with Camaron branding)
# ---------------------------------------------------------------------------

def run_gui() -> int:
    """Start the Camaron chatbot in GUI mode."""
    if not _HAS_QT or QApplication is None:
        print("PyQt6 is unavailable — falling back to CLI mode.")
        return run_cli()

    app = QApplication(sys.argv)

    entity = DokuEntity()
    window = DokuChatWindow(entity)

    # Rebrand the window as Camaron
    window.setWindowTitle("Camaron")
    # Update the "Doku" title label inside the window if accessible
    try:
        from PyQt6.QtWidgets import QLabel as _QLabel
        for label in window.findChildren(_QLabel):
            if label.text() == "Doku":
                font = QFont()
                font.setPointSize(18)
                font.setBold(True)
                label.setFont(font)
                label.setText("Camaron")
                break
    except Exception:
        pass

    window.show()
    return app.exec()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="camaron",
        description="Camaron — local-first knowledge chatbot",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="open the graphical chat window (requires PyQt6)",
    )
    args = parser.parse_args(argv)

    if args.gui:
        return run_gui()
    return run_cli()


if __name__ == "__main__":
    raise SystemExit(main())
