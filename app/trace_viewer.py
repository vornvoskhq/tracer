import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

from PyQt5 import QtCore, QtWidgets
from PyQt5.Qsci import QsciScintilla, QsciLexerPython

from main_execution_tracer import MainExecutionTracer
from .code_utils import find_enclosing_function, extract_source_segment
from .llm_client import OpenRouterClient


@dataclass
class FunctionCallView:
    index: int
    file: str
    function: str
    line: int
    depth: int


@dataclass
class FileAccessView:
    index: int
    mode: str
    src_file: str
    src_func: str
    src_line: int
    file_path: str


class CodeEditor(QsciScintilla):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setUtf8(True)

        # Lexer for Python syntax highlighting
        lexer = QsciLexerPython(self)
        self.setLexer(lexer)

        # Line numbers
        self.setMarginType(0, QsciScintilla.NumberMargin)
        self.setMarginWidth(0, "0000")
        # Improve contrast of line numbers vs background
        self.setMarginsForegroundColor(QtCore.Qt.black)
        self.setMarginsBackgroundColor(QtCore.Qt.white)

        # Basic editor settings
        self.setTabWidth(4)
        self.setIndentationGuides(True)
        self.setIndentationsUseTabs(False)
        self.setAutoIndent(True)
        self.setFolding(QsciScintilla.BoxedTreeFoldStyle)

        # Always enable word wrap
        self.setWrapMode(QsciScintilla.WrapWord)

    def set_code(self, code: str):
        self.setText(code)
        # Reset cursor to top
        self.setCursorPosition(0, 0)


class TraceViewerWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._current_codebase: Optional[Path] = None
        self._current_command: str = ""
        self._function_calls: List[FunctionCallView] = []
        self._file_accesses: List[FileAccessView] = []

        self._llm_client = OpenRouterClient()

        # Background workers
        self._trace_worker: Optional[_TraceWorker] = None
        self._llm_worker: Optional[_LLMSummaryWorker] = None

        self._build_ui()
        self._connect_signals()

    # UI construction -----------------------------------------------------

    def _build_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        main_layout.addWidget(main_splitter)

        # Left side: vertical split (top: controls + execution + I/O, bottom: summary)
        left_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, main_splitter)

        # Top-left container: codebase label, command row, combined function execution and file I/O
        top_left = QtWidgets.QWidget(left_splitter)
        top_left_layout = QtWidgets.QVBoxLayout(top_left)
        top_left_layout.setContentsMargins(4, 4, 4, 4)
        top_left_layout.setSpacing(4)

        # Codebase label that reflects current selection
        self.codebase_label = QtWidgets.QLabel("Codebase: (none selected)", top_left)
        self.codebase_label.setStyleSheet("font-weight: bold;")
        top_left_layout.addWidget(self.codebase_label)

        # Command row: entry command textbox + Run Trace button
        command_row = QtWidgets.QHBoxLayout()
        self.command_edit = QtWidgets.QLineEdit(top_left)
        self.command_edit.setPlaceholderText("Entry command to trace (e.g., ./vg list)")
        self.run_button = QtWidgets.QPushButton("Run Trace", top_left)
        command_row.addWidget(self.command_edit, stretch=1)
        command_row.addWidget(self.run_button, stretch=0)
        top_left_layout.addLayout(command_row)

        # Combined function execution and file I/O tree
        self.left_tree = QtWidgets.QTreeWidget(top_left)
        # Columns: Order, Depth, Kind, File, Function, Line/Mode
        self.left_tree.setHeaderLabels(
            ["Order", "Depth", "Kind", "File", "Function", "Line/Mode"]
        )
        self.left_tree.setColumnWidth(0, 60)
        self.left_tree.setColumnWidth(1, 60)
        self.left_tree.setColumnWidth(2, 70)
        self.left_tree.setColumnWidth(3, 220)
        self.left_tree.setColumnWidth(4, 140)
        top_left_layout.addWidget(self.left_tree, stretch=1)

        # Bottom-left: summary area
        summary_container = QtWidgets.QWidget(left_splitter)
        summary_layout = QtWidgets.QVBoxLayout(summary_container)
        summary_layout.setContentsMargins(4, 4, 4, 4)
        summary_layout.setSpacing(4)

        self.summary_label = QtWidgets.QLabel("LLM Summary", summary_container)
        self.summary_label.setStyleSheet("font-weight: bold;")
        self.summary_text = QtWidgets.QPlainTextEdit(summary_container)
        self.summary_text.setReadOnly(True)
        self.summary_button = QtWidgets.QPushButton(
            "Summarize Highlighted Function", summary_container
        )

        summary_layout.addWidget(self.summary_label)
        summary_layout.addWidget(self.summary_text, stretch=1)
        summary_layout.addWidget(self.summary_button, stretch=0)

        # Right side: code editor
        self.editor = CodeEditor(main_splitter)

        # Adjust splitter sizes: make summary vertically smaller
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        left_splitter.setStretchFactor(0, 5)
        left_splitter.setStretchFactor(1, 1)

    def _connect_signals(self):
        self.left_tree.itemClicked.connect(self._on_left_item_clicked)
        self.summary_button.clicked.connect(self._on_summarize_clicked)
        self.run_button.clicked.connect(self._on_run_button_clicked)

    # Public API ----------------------------------------------------------

    def set_codebase(self, path: Path):
        self._current_codebase = path
        # Update label to reflect current codebase (just top-level directory)
        self.codebase_label.setText(f"Codebase: {path.name}")

    def set_command(self, command: str):
        self._current_command = command
        self.command_edit.setText(command)

    def has_configuration(self) -> bool:
        return self._current_codebase is not None and bool(self._current_command)

    def run_trace(self):
        """
        Run the tracer for the current configuration.

        This is potentially slow; we run it in a background thread to keep
        the UI responsive.
        """
        # Pull command from the textbox
        self._current_command = self.command_edit.text().strip()

        if not self.has_configuration():
            QtWidgets.QMessageBox.warning(
                self,
                "Trace Not Configured",
                "Please select a codebase and enter a command first.",
            )
            return

        if self._trace_worker is not None and self._trace_worker.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "Trace Running",
                "A trace is already running. Please wait for it to complete.",
            )
            return

        codebase = self._current_codebase
        assert codebase is not None
        command = self._current_command

        self.left_tree.clear()
        self.editor.set_code("")
        self.summary_text.clear()

        self.left_tree.setDisabled(True)
        self.summary_button.setDisabled(True)
        self.run_button.setDisabled(True)

        worker = _TraceWorker(codebase, command)
        worker.finished_with_result.connect(self._on_trace_finished)
        worker.error_occurred.connect(self._on_trace_error)
        self._trace_worker = worker

        worker.start()

    def _on_run_button_clicked(self):
        self.run_trace()

    # Slots ---------------------------------------------------------------

    def _on_trace_finished(
        self,
        function_calls: List[FunctionCallView],
        file_accesses: List[FileAccessView],
    ):
        self._function_calls = function_calls
        self._file_accesses = file_accesses

        self.left_tree.setDisabled(False)
        self.summary_button.setDisabled(False)
        self.run_button.setDisabled(False)
        self._trace_worker = None

        # Populate tree with collapsible groups
        root_execution = QtWidgets.QTreeWidgetItem(
            self.left_tree, ["", "", "Exec", "Function Execution Order", "", ""]
        )
        root_execution.setExpanded(True)

        for call in function_calls:
            item = QtWidgets.QTreeWidgetItem(
                root_execution,
                [
                    str(call.index),          # Order
                    str(call.depth),          # Depth
                    "Func",                   # Kind
                    call.file,                # File
                    call.function,            # Function
                    str(call.line),           # Line
                ],
            )
            # Store metadata for click handling
            item.setData(0, QtCore.Qt.UserRole, ("func", call))

        root_io = QtWidgets.QTreeWidgetItem(
            self.left_tree, ["", "", "I/O", "External File I/O", "", ""]
        )
        root_io.setExpanded(True)

        for fa in file_accesses:
            item = QtWidgets.QTreeWidgetItem(
                root_io,
                [
                    str(fa.index),           # Order
                    "",                      # Depth (not applicable)
                    fa.mode,                 # Kind / mode
                    fa.file_path,            # File
                    f"{fa.src_file}:{fa.src_func}",  # Function context
                    str(fa.src_line) if fa.src_line else "",  # Line
                ],
            )
            item.setData(0, QtCore.Qt.UserRole, ("io", fa))

        self.left_tree.expandAll()

    def _on_trace_error(self, message: str):
        # Re-enable controls
        self.left_tree.setDisabled(False)
        self.summary_button.setDisabled(False)
        self.run_button.setDisabled(False)
        self._trace_worker = None

        # Prefer console logging over GUI popups for trace errors
        print(f"[TraceViewerWidget] Trace failed: {message}")

    def _on_left_item_clicked(self, item: QtWidgets.QTreeWidgetItem, column: int):
        payload = item.data(0, QtCore.Qt.UserRole)
        if not payload:
            return

        kind, obj = payload
        if kind != "func":
            return

        call: FunctionCallView = obj
        if self._current_codebase is None:
            return

        file_path = (self._current_codebase / call.file).resolve()
        if not file_path.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "File Not Found",
                f"Could not locate file:\n{file_path}",
            )
            return

        func_loc = find_enclosing_function(
            file_path=file_path, line_number=call.line, function_name=call.function
        )
        if not func_loc:
            # Fallback: show from the clicked line forward
            code = extract_source_segment(file_path, call.line, call.line + 80)
            self.editor.set_code(code)
            return

        code = extract_source_segment(
            func_loc.file_path, func_loc.start_line, func_loc.end_line
        )
        self.editor.set_code(code)

    def _on_summarize_clicked(self):
        code = self.editor.text()
        if not code.strip():
            QtWidgets.QMessageBox.information(
                self,
                "No Code",
                "There is no function code to summarize. Select a function first.",
            )
            return

        self.summary_button.setDisabled(True)
        self.summary_text.setPlainText("Requesting summary from OpenRouter...")

        if self._llm_worker is not None and self._llm_worker.isRunning():
            # Avoid starting multiple concurrent LLM requests
            return

        worker = _LLMSummaryWorker(code, self._llm_client)
        worker.finished_with_result.connect(self._on_summary_finished)
        worker.error_occurred.connect(self._on_summary_error)
        self._llm_worker = worker
        worker.start()

    def _on_summary_finished(self, text: str):
        self.summary_button.setDisabled(False)
        self.summary_text.setPlainText(text)
        self._llm_worker = None

    def _on_summary_error(self, message: str):
        self.summary_button.setDisabled(False)
        self.summary_text.setPlainText(message)
        self._llm_worker = None

    def cleanup_threads(self):
        """
        Stop background threads cleanly. Intended to be called on application exit.

        As a precaution, this also prints basic debug information to the console
        about any worker threads that were still running at shutdown time.
        """
        # Trace worker
        if self._trace_worker is not None:
            if self._trace_worker.isRunning():
                print(
                    "[TraceViewerWidget] cleanup_threads: trace worker still running, "
                    "requesting quit() and waiting for it to exit..."
                )
                self._trace_worker.quit()
                self._trace_worker.wait()
                print(
                    "[TraceViewerWidget] cleanup_threads: trace worker exited cleanly."
                )
            else:
                print(
                    "[TraceViewerWidget] cleanup_threads: trace worker exists but is not running."
                )
            self._trace_worker = None

        # LLM worker
        if self._llm_worker is not None:
            if self._llm_worker.isRunning():
                print(
                    "[TraceViewerWidget] cleanup_threads: LLM worker still running, "
                    "requesting quit() and waiting for it to exit..."
                )
                self._llm_worker.quit()
                self._llm_worker.wait()
                print(
                    "[TraceViewerWidget] cleanup_threads: LLM worker exited cleanly."
                )
            else:
                print(
                    "[TraceViewerWidget] cleanup_threads: LLM worker exists but is not running."
                )
            self._llm_worker = None


class _TraceWorker(QtCore.QThread):
    finished_with_result = QtCore.pyqtSignal(list, list)
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(self, codebase: Path, command: str):
        super().__init__()
        self._codebase = codebase
        self._command = command

    def run(self):
        try:
            tracer = MainExecutionTracer(target_dir=str(self._codebase), auto_detect=False)
            trace = tracer.trace_command(self._command)

            function_calls: List[FunctionCallView] = []
            file_accesses: List[FileAccessView] = []

            # Build function execution order from full call records if available
            calls_raw: List[Dict[str, Any]] = getattr(trace, "calls", []) or []
            if calls_raw:
                for idx, c in enumerate(calls_raw, start=1):
                    func_name = c.get("function", "")
                    # Filter out lambda functions
                    if func_name == "<lambda>":
                        continue
                    function_calls.append(
                        FunctionCallView(
                            index=idx,
                            file=c.get("file", ""),
                            function=func_name,
                            line=int(c.get("line", 0) or 0),
                            depth=int(c.get("depth", 0) or 0),
                        )
                    )
            else:
                # Fallback to call_sequence with or without line numbers
                sequence_with_lines = getattr(trace, "call_sequence_with_lines", None)
                if sequence_with_lines:
                    for idx, entry in enumerate(sequence_with_lines, start=1):
                        # entry: "file.py::func:line"
                        file_part, rest = entry.split("::", 1)
                        if ":" in rest:
                            func_name, line_str = rest.split(":", 1)
                            try:
                                line_no = int(line_str)
                            except ValueError:
                                line_no = 0
                        else:
                            func_name = rest
                            line_no = 0

                        if func_name == "<lambda>":
                            continue

                        function_calls.append(
                            FunctionCallView(
                                index=idx,
                                file=file_part,
                                function=func_name,
                                line=line_no,
                                depth=0,
                            )
                        )
                else:
                    # Fallback to call_sequence without line numbers
                    for idx, entry in enumerate(trace.call_sequence, start=1):
                        if "::" in entry:
                            file_part, func_name = entry.split("::", 1)
                        else:
                            file_part, func_name = "", entry

                        if func_name == "<lambda>":
                            continue

                        function_calls.append(
                            FunctionCallView(
                                index=idx,
                                file=file_part,
                                function=func_name,
                                line=0,
                                depth=0,
                            )
                        )

            # External file I/O with filtering
            file_accesses_raw: List[Dict[str, Any]] = getattr(trace, "file_accesses", []) or []
            for idx, fa in enumerate(file_accesses_raw, start=1):
                file_path = fa.get("file", "") or ""
                # Filter out virtualenv and standard-library style paths
                skip = False
                for pattern in ("/.venv/", "\\\\.venv\\\\", "site-packages", "/lib/python", "\\\\Lib\\\\"):
                    if pattern in file_path:
                        skip = True
                        break
                if skip:
                    continue

                file_accesses.append(
                    FileAccessView(
                        index=idx,
                        mode=fa.get("mode", ""),
                        src_file=fa.get("src_file", ""),
                        src_func=fa.get("src_func", ""),
                        src_line=int(fa.get("src_line", 0) or 0),
                        file_path=file_path,
                    )
                )

            self.finished_with_result.emit(function_calls, file_accesses)
        except Exception as exc:
            self.error_occurred.emit(str(exc))


class _LLMSummaryWorker(QtCore.QThread):
    finished_with_result = QtCore.pyqtSignal(str)
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(self, code: str, client: OpenRouterClient):
        super().__init__()
        self._code = code
        self._client = client

    def run(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                text = loop.run_until_complete(self._client.summarize_function(self._code))
            finally:
                loop.close()
            self.finished_with_result.emit(text)
        except Exception as exc:
            self.error_occurred.emit(str(exc))