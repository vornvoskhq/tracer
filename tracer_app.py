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

# Ensure the repo root is on sys.path so that the "app" package (and its tracer
# engine module) can be imported reliably, even when running this script
# directly.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main_app import MainWindow  # noqa: E402


def main() -> None:
    # Optional first CLI argument can be a path to an initial codebase folder.
    initial_codebase = None
    if len(sys.argv) > 1:
        candidate = Path(sys.argv[1]).expanduser().resolve()
        if candidate.exists() and candidate.is_dir():
            initial_codebase = candidate

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(initial_codebase=initial_codebase)
    win.show()

    def _on_about_to_quit():
        """
        Slot invoked when QApplication is about to quit (tracer_app entrypoint).

        This is a last-chance hook to ensure worker threads are cleaned up.
        """
        if hasattr(win, "viewer") and hasattr(win.viewer, "cleanup_threads"):
            win.viewer.cleanup_threads()

    # Ensure background worker threads are cleaned up before the app exits,
    # even if it is quit without explicitly closing the main window.
    app.aboutToQuit.connect(_on_about_to_quit)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
