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
        self.setMarginsForegroundColor(QtCore.Qt.gray)
        self.setMarginsBackgroundColor(QtCore.Qt.lightGray)

        # Basic editor settings
        self.setTabWidth(4)
        self.setIndentationGuides(True)
        self.setIndentationsUseTabs(False)
        self.setAutoIndent(True)
        self.setFolding(QsciScintilla.BoxedTreeFoldStyle)

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

        self._build_ui()
        self._connect_signals()

    # UI construction -----------------------------------------------------

    def _build_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        main_layout.addWidget(main_splitter)

        # Left side: vertical split (top: execution + I/O, bottom: summary)
        left_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, main_splitter)

        # Top-left: combined function execution and file I/O
        self.left_tree = QtWidgets.QTreeWidget(left_splitter)
        self.left_tree.setHeaderLabels(
            ["Order", "Kind", "File/Func", "Line/Mode", "Extra"]
        )
        self.left_tree.setColumnWidth(0, 60)
        self.left_tree.setColumnWidth(1, 80)
        self.left_tree.setColumnWidth(2, 260)

        # Bottom-left: summary area
        summary_container = QtWidgets.QWidget(left_splitter)
        summary_layout = QtWidgets.QVBoxLayout(summary_container)

        self.summary_label = QtWidgets.QLabel(
            "LLM Summary of Selected Function", summary_container
        )
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

        # Adjust splitter sizes
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        left_splitter.setStretchFactor(0, 3)
        left_splitter.setStretchFactor(1, 2)

    def _connect_signals(self):
        self.left_tree.itemClicked.connect(self._on_left_item_clicked)
        self.summary_button.clicked.connect(self._on_summarize_clicked)

    # Public API ----------------------------------------------------------

    def set_codebase(self, path: Path):
        self._current_codebase = path

    def set_command(self, command: str):
        self._current_command = command

    def has_configuration(self) -> bool:
        return self._current_codebase is not None and bool(self._current_command)

    def run_trace(self):
        """
        Run the tracer for the current configuration.

        This is potentially slow; we run it in a background thread to keep
        the UI responsive.
        """
        if not self.has_configuration():
            QtWidgets.QMessageBox.warning(
                self,
                "Trace Not Configured",
                "Please select a codebase and command first.",
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

        worker = _TraceWorker(codebase, command)
        worker.finished_with_result.connect(self._on_trace_finished)
        worker.error_occurred.connect(self._on_trace_error)

        worker.start()

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

        # Populate tree with collapsible groups
        root_execution = QtWidgets.QTreeWidgetItem(
            self.left_tree, ["", "Exec", "Function Execution Order", "", ""]
        )
        root_execution.setExpanded(True)

        for call in function_calls:
            label = f"{call.file}::{call.function}"
            extra = ""
            item = QtWidgets.QTreeWidgetItem(
                root_execution,
                [
                    str(call.index),
                    "Func",
                    label,
                    str(call.line),
                    extra,
                ],
            )
            # Store metadata for click handling
            item.setData(0, QtCore.Qt.UserRole, ("func", call))

        root_io = QtWidgets.QTreeWidgetItem(
            self.left_tree, ["", "I/O", "External File I/O", "", ""]
        )
        root_io.setExpanded(True)

        for fa in file_accesses:
            label = fa.file_path
            extra = f"{fa.src_file}::{fa.src_func}:{fa.src_line}"
            item = QtWidgets.QTreeWidgetItem(
                root_io,
                [
                    str(fa.index),
                    fa.mode,
                    label,
                    "",
                    extra,
                ],
            )
            item.setData(0, QtCore.Qt.UserRole, ("io", fa))

        self.left_tree.expandAll()

    def _on_trace_error(self, message: str):
        self.left_tree.setDisabled(False)
        self.summary_button.setDisabled(False)
        QtWidgets.QMessageBox.critical(self, "Trace Failed", message)

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

        worker = _LLMSummaryWorker(code, self._llm_client)
        worker.finished_with_result.connect(self._on_summary_finished)
        worker.error_occurred.connect(self._on_summary_error)
        worker.start()

    def _on_summary_finished(self, text: str):
        self.summary_button.setDisabled(False)
        self.summary_text.setPlainText(text)

    def _on_summary_error(self, message: str):
        self.summary_button.setDisabled(False)
        self.summary_text.setPlainText(message)


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

            # Build function execution order with line numbers where possible
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

                    function_calls.append(
                        FunctionCallView(
                            index=idx,
                            file=file_part,
                            function=func_name,
                            line=line_no,
                        )
                    )
            else:
                # Fallback to call_sequence without line numbers
                for idx, entry in enumerate(trace.call_sequence, start=1):
                    if "::" in entry:
                        file_part, func_name = entry.split("::", 1)
                    else:
                        file_part, func_name = "", entry
                    function_calls.append(
                        FunctionCallView(
                            index=idx,
                            file=file_part,
                            function=func_name,
                            line=0,
                        )
                    )

            # External file I/O
            file_accesses_raw: List[Dict[str, Any]] = getattr(trace, "file_accesses", []) or []
            for idx, fa in enumerate(file_accesses_raw, start=1):
                file_accesses.append(
                    FileAccessView(
                        index=idx,
                        mode=fa.get("mode", ""),
                        src_file=fa.get("src_file", ""),
                        src_func=fa.get("src_func", ""),
                        src_line=fa.get("src_line", 0),
                        file_path=fa.get("file", ""),
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