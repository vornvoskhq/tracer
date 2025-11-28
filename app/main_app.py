import sys
from pathlib import Path

from PyQt5 import QtWidgets

from .trace_viewer import TraceViewerWidget


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, initial_codebase: Path | None = None):
        super().__init__()

        self.setWindowTitle("Execution Trace Viewer")
        self.resize(1200, 800)

        self.viewer = TraceViewerWidget(self)
        self.setCentralWidget(self.viewer)

        # If an initial codebase was provided (e.g. from run_trcr.sh / global
        # "trcr" symlink), configure the viewer with that folder immediately.
        if initial_codebase is not None:
            self.viewer.set_codebase(initial_codebase)

        self._create_actions()
        self._create_menu()

    def _create_actions(self):
        self.action_open_codebase = QtWidgets.QAction("&Open Codebase...", self)
        self.action_open_codebase.triggered.connect(self._on_open_codebase)

        self.action_run_trace = QtWidgets.QAction("&Run Trace", self)
        self.action_run_trace.triggered.connect(self._on_run_trace)

        self.action_quit = QtWidgets.QAction("&Quit", self)
        # Route Quit through the main window's close() so our closeEvent handler
        # can perform thread cleanup before the application exits.
        self.action_quit.triggered.connect(self.close)

        # Config menu actions
        self.action_toggle_caller_column = QtWidgets.QAction(
            "Show &Caller Column", self
        )
        self.action_toggle_caller_column.setCheckable(True)
        self.action_toggle_caller_column.setChecked(True)
        # Toggle visibility of the Caller column in the trace viewer
        self.action_toggle_caller_column.toggled.connect(
            self._on_toggle_caller_column
        )

        self.action_toggle_phase_column = QtWidgets.QAction(
            "Show &Phase Column", self
        )
        self.action_toggle_phase_column.setCheckable(True)
        # Default: Phase column off until explicitly enabled
        self.action_toggle_phase_column.setChecked(False)
        self.action_toggle_phase_column.toggled.connect(
            self._on_toggle_phase_column
        )

        self.action_toggle_import_rows = QtWidgets.QAction(
            "Hide &Import-time Calls", self
        )
        self.action_toggle_import_rows.setCheckable(True)
        self.action_toggle_import_rows.setChecked(False)
        self.action_toggle_import_rows.toggled.connect(
            self._on_toggle_import_rows
        )

    def _create_menu(self):
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.action_open_codebase)
        file_menu.addSeparator()
        file_menu.addAction(self.action_run_trace)
        file_menu.addSeparator()
        file_menu.addAction(self.action_quit)

        config_menu = self.menuBar().addMenu("&Config")
        config_menu.addAction(self.action_toggle_caller_column)
        config_menu.addAction(self.action_toggle_phase_column)
        config_menu.addAction(self.action_toggle_import_rows)

    # Slots ---------------------------------------------------------------

    def _on_open_codebase(self):
        default_dir = Path("target")
        if default_dir.exists():
            start_dir = str(default_dir.resolve())
        else:
            start_dir = str(Path.cwd())
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Codebase Folder",
            start_dir,
        )
        if directory:
            self.viewer.set_codebase(Path(directory))

    def _on_run_trace(self):
        self.viewer.run_trace()

    def _on_toggle_caller_column(self, checked: bool):
        """
        Toggle visibility of the Caller column in the left-hand trace tree.
        """
        if hasattr(self.viewer, "set_caller_column_visible"):
            self.viewer.set_caller_column_visible(checked)

    def _on_toggle_phase_column(self, checked: bool):
        """
        Toggle visibility of the Phase column ("Import" vs "Runtime") in the left-hand trace tree.
        """
        if hasattr(self.viewer, "set_phase_column_visible"):
            self.viewer.set_phase_column_visible(checked)

    def _on_toggle_import_rows(self, checked: bool):
        """
        Hide or show import-time calls in the execution tree.
        """
        if hasattr(self.viewer, "set_import_rows_hidden"):
            self.viewer.set_import_rows_hidden(checked)

    def closeEvent(self, event):
        """
        Ensure background threads are stopped cleanly before the app exits.
        """
        if hasattr(self.viewer, "cleanup_threads"):
            self.viewer.cleanup_threads()
        event.accept()


def main():
    # Basic global exception hook to log uncaught errors before exiting.
    def _excepthook(exc_type, exc_value, exc_traceback):
        import traceback

        print("\n[Execution Trace Viewer] Unhandled exception in main thread:")
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print("[Execution Trace Viewer] Application will terminate due to the above unhandled exception.")
        # Delegate to the default handler as well so Qt / Python can do their normal shutdown.
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = _excepthook

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()

    def _on_about_to_quit():
        """
        Slot invoked when QApplication is about to quit.

        This is a last-chance hook to ensure worker threads are cleaned up.
        """
        if hasattr(win, "viewer") and hasattr(win.viewer, "cleanup_threads"):
            win.viewer.cleanup_threads()

    # Ensure background worker threads are cleaned up even if the application
    # is quit via mechanisms other than closing the main window directly.
    app.aboutToQuit.connect(_on_about_to_quit)

    exit_code = app.exec_()
    print(f"[Execution Trace Viewer] Application exiting with code {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()