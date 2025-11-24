#!/usr/bin/env python3
"""
Root entrypoint for the Execution Trace Viewer GUI.

Usage (from repo root, inside the existing .venv):

    python run_trace_viewer.py

This launches the Qt application defined in app/main_app.py.
"""

import sys
from pathlib import Path

from PyQt5 import QtWidgets  # type: ignore

# Ensure the repo root is on sys.path so that "app" and "main_execution_tracer"
# can be imported reliably, even when running this script directly.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main_app import MainWindow  # noqa: E402


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()