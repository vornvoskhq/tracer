import sys
from pathlib import Path

from PyQt5 import QtWidgets

from .trace_viewer import TraceViewerWidget


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Execution Trace Viewer")
        self.resize(1200, 800)

        self.viewer = TraceViewerWidget(self)
        self.setCentralWidget(self.viewer)

        self._create_actions()
        self._create_menu()

    def _create_actions(self):
        self.action_open_codebase = QtWidgets.QAction("&Open Codebase...", self)
        self.action_open_codebase.triggered.connect(self._on_open_codebase)

        self.action_run_trace = QtWidgets.QAction("&Run Trace", self)
        self.action_run_trace.triggered.connect(self._on_run_trace)

        self.action_quit = QtWidgets.QAction("&Quit", self)
        self.action_quit.triggered.connect(QtWidgets.qApp.quit)  # type: ignore[attr-defined]

    def _create_menu(self):
        menu = self.menuBar().addMenu("&File")
        menu.addAction(self.action_open_codebase)
        menu.addSeparator()
        menu.addAction(self.action_run_trace)
        menu.addSeparator()
        menu.addAction(self.action_quit)

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

    def closeEvent(self, event):
        """
        Ensure background threads are stopped cleanly before the app exits.
        """
        if hasattr(self.viewer, "cleanup_threads"):
            self.viewer.cleanup_threads()
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()