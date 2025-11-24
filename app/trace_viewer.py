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

        # Track the base (absolute) line number for the first line of the
        # currently displayed code segment so that margin numbers can match
        # the original file's line numbering.
        self._base_line: int = 1

        # Line numbers: use a text margin so we can control the starting
        # line number (offset) instead of always starting at 1.
        self.setMarginType(0, QsciScintilla.TextMargin)
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

    def _update_margin_line_numbers(self) -> None:
        """
        Update the margin text to reflect the current base line number.
        """
        # Clear and repopulate the margin text for each line.
        lines = self.lines()
        for i in range(lines):
            # Absolute line number in the original file.
            line_no = self._base_line + i
            # Use style 0 for all margin text; this uses the default margin
            # foreground/background colors configured on the editor.
            self.setMarginText(i, str(line_no), 0)

    def set_code(self, code: str, base_line: int = 1):
        """
        Set the editor contents and adjust the visible line numbers so that
        they match the original file's line numbering, starting at base_line.
        """
        self._base_line = max(int(base_line), 1)
        self.setText(code)
        self._update_margin_line_numbers()
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
        # Columns: (indent), Order, Depth, Kind, File, Function, Line/Mode
        # Column 0 is a narrow, mostly empty column that holds the tree
        # indentation and expand/collapse icons so that the visible "Order"
        # numbers in column 1 are not pushed to the right by tree padding.
        self.left_tree.setHeaderLabels(
            ["", "Order", "Depth", "Kind", "File", "Function", "Line/Mode"]
        )
        # Make the indent column very narrow and the Order column small
        self.left_tree.setColumnWidth(0, 18)
        self.left_tree.setColumnWidth(1, 50)
        self.left_tree.setColumnWidth(2, 60)
        self.left_tree.setColumnWidth(3, 70)
        self.left_tree.setColumnWidth(4, 220)
        self.left_tree.setColumnWidth(5, 140)
        # Enable a custom context menu so we can offer "Copy tree to clipboard"
        self.left_tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
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

        # Right side: container with label + code editor
        right_container = QtWidgets.QWidget(main_splitter)
        right_layout = QtWidgets.QVBoxLayout(right_container)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(4)

        self.editor_label = QtWidgets.QLabel("File: (none)", right_container)
        self.editor_label.setStyleSheet("font-weight: bold;")
        right_layout.addWidget(self.editor_label, stretch=0)

        self.editor = CodeEditor(right_container)
        right_layout.addWidget(self.editor, stretch=1)

        # Adjust splitter sizes: make summary vertically smaller
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        left_splitter.setStretchFactor(0, 5)
        left_splitter.setStretchFactor(1, 1)

    def _connect_signals(self):
        self.left_tree.itemClicked.connect(self._on_left_item_clicked)
        # Right-click context menu on the left tree
        self.left_tree.customContextMenuRequested.connect(
            self._on_left_tree_context_menu
        )
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
        # Reset editor to an empty buffer with line numbers starting at 1.
        self.editor.set_code("", base_line=1)
        if hasattr(self, "editor_label"):
            self.editor_label.setText("File: (none)")
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

        # Normalize depths so the shallowest call starts at 0
        if self._function_calls:
            min_depth = min(fc.depth for fc in self._function_calls)
            if min_depth != 0:
                for fc in self._function_calls:
                    fc.depth = max(fc.depth - min_depth, 0)

        self.left_tree.setDisabled(False)
        self.summary_button.setDisabled(False)
        self.run_button.setDisabled(False)

        # Ensure the worker thread has fully finished before dropping our
        # reference to it, to avoid the QThread being destroyed while it is
        # still running.
        worker = self._trace_worker
        if worker is not None:
            if worker.isRunning():
                worker.wait()
            worker.deleteLater()
        self._trace_worker = None

        # Populate tree with collapsible groups. Column layout:
        #   0: (indent / tree controls)
        #   1: Order
        #   2: Depth
        #   3: Kind
        #   4: File
        #   5: Function
        #   6: Line/Mode
        root_execution = QtWidgets.QTreeWidgetItem(
            self.left_tree, ["", "", "", "Exec", "Function Execution Order", "", ""]
        )
        #   1: Order
        #   2: Depth
        #   3: Kind
        #   4: File
        #   5: Function
        #   6: Line/Mode
        root_execution = QtWidgets.QTreeWidgetItem(
            self.left_tree, ["", "", "", "Exec", "Function Execution Order", "", ""]
        )
        root_execution.setExpanded(True)

        for call in self._function_calls:
            item = QtWidgets.QTreeWidgetItem(
                root_execution,
                [
                    "",                       # (indent only)
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
            self.left_tree, ["", "", "", "I/O", "External File I/O", "", ""]
        )
        root_io.setExpanded(True)

        for fa in file_accesses:
            item = QtWidgets.QTreeWidgetItem(
                root_io,
                [
                    "",                       # (indent only)
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
        # Make the Order column just wide enough for its contents
        self.left_tree.resizeColumnToContents(1)

    def _on_trace_error(self, message: str):
        # Re-enable controls
        self.left_tree.setDisabled(False)
        self.summary_button.setDisabled(False)
        self.run_button.setDisabled(False)

        # Ensure the worker thread has fully finished before dropping our
        # reference to it, to avoid the QThread being destroyed while it is
        # still running.
        worker = self._trace_worker
        if worker is not None:
            if worker.isRunning():
                worker.wait()
            worker.deleteLater()
        self._trace_worker = None

        # Populate tree with collapsible groups. Column layout:
        #   0: (indent / tree controls)
        #   1: Order
        #   2: Depth
        #   3: Kind
        #   4: File
        #   5: Function
        #   6: Line/Mode
        root_execution = QtWidgets.QTreeWidgetItem(
            self.left_tree, ["", "", "", "Exec", "Function Execution Order", "", ""]
        )
        open_file_action = None
        if payload:
            open_file_action = menu.addAction("Open full file in right pane")

        selected = menu.exec_(self.left_tree.viewport().mapToGlobal(pos))
        if selected is copy_action:
            self._copy_tree_to_clipboard()
        elif open_file_action is not None and selected is open_file_action:
            self._open_full_file_for_item(item)

    def _open_full_file_for_item(self, item: QtWidgets.QTreeWidgetItem) -> None:
        """
        Open the entire source file corresponding to the given tree item in the
        right-hand code editor, instead of a snippet.

        For execution entries, this uses the function's file.
        For I/O entries, this uses the caller's source file when available.
        """
        if self._current_codebase is None:
            return

        payload = item.data(0, QtCore.Qt.UserRole)

        # If a top-level group header (e.g., "Function Execution Order" or
        # "External File I/O") is clicked, show the full file/module when
        # possible instead of doing nothing.
        if not payload:
            # For now we treat left-click on group headers as a no-op; the
            # context menu "Open full file in right pane" covers the module-
            # level display use case more explicitly.
            return

        kind, obj = payload
        if self._current_codebase is None:
            return

        # Handle function execution entries
        if kind == "func":
            call: FunctionCallView = obj
            rel_path = call.file
            target_line = call.line or 1
        elif kind == "io":
            fa: FileAccessView = obj
            # Prefer the caller's source file for I/O entries
            rel_path = fa.src_file or ""
            target_line = fa.src_line or 1
            if not rel_path:
                return
        else:
            return

        file_path = (self._current_codebase / rel_path).resolve()
        if not file_path.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "File Not Found",
                f"Could not locate file:\n{file_path}",
            )
            return

        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Error Reading File",
                f"Could not read file:\n{file_path}\n\nError: {exc}",
            )
            return

        # Show the entire file with line numbers starting at 1
        self.editor.set_code(text, base_line=1)

        # Position the cursor near the relevant line if we have one
        if target_line > 0:
            line_index = max(target_line - 1, 0)
            self.editor.setCursorPosition(line_index, 0)

        if hasattr(self, "editor_label"):
            try:
                rel_display = file_path.relative_to(self._current_codebase)
            except ValueError:
                rel_display = file_path
            self.editor_label.setText(f"File: {rel_display}")

    def _copy_tree_to_clipboard(self):
        """
        Copy the full contents of the left tree to the clipboard as
        tab-separated text for easy sharing/analysis.
        """
        column_count = self.left_tree.columnCount()
        if column_count <= 0:
            return

        header_item = self.left_tree.headerItem()
        headers = [header_item.text(col) for col in range(column_count)]
        lines = ["\t".join(headers)]

        def visit(item: QtWidgets.QTreeWidgetItem, depth: int) -> None:
            columns: List[str] = []
            for col in range(column_count):
                text = item.text(col)
                # Indent the Kind column to reflect tree depth (column 3)
                if col == 3 and depth > 0:
                    text = "  " * depth + text
                # Ensure the Order column (column 1) has no leading/trailing whitespace
                if col == 1:
                    text = text.strip()
                columns.append(text)
            lines.append("\t".join(columns))
            for idx in range(item.childCount()):
                visit(item.child(idx), depth + 1)

        for i in range(self.left_tree.topLevelItemCount()):
            root_item = self.left_tree.topLevelItem(i)
            if root_item is not None:
                visit(root_item, 0)

        text = "\n".join(lines)
        if not text.strip():
            return

        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setText(text)
        print("[TraceViewerWidget] Copied left tree to clipboard")

    def _on_left_item_clicked(self, item: QtWidgets.QTreeWidgetItem, column: int):
        payload = item.data(0, QtCore.Qt.UserRole)
        if not payload:
            return

        kind, obj = payload
        if self._current_codebase is None:
            return

        # Handle function execution entries
        if kind == "func":
            call: FunctionCallView = obj

            # Special-case module-level entries: show the full file in the
            # right-hand editor instead of trying to extract a function span.
            if call.function == "<module>":
                file_path = (self._current_codebase / call.file).resolve()
                if not file_path.exists():
                    QtWidgets.QMessageBox.warning(
                        self,
                        "File Not Found",
                        f"Could not locate file:\n{file_path}",
                    )
                    return

                try:
                    text = file_path.read_text(encoding="utf-8")
                except OSError as exc:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Error Reading File",
                        f"Could not read file:\n{file_path}\n\nError: {exc}",
                    )
                    return

                # Show full file starting at line 1
                self.editor.set_code(text, base_line=1)

                # Position the cursor near the call line, if we have one
                if call.line > 0:
                    line_index = max(call.line - 1, 0)
                    self.editor.setCursorPosition(line_index, 0)

                if hasattr(self, "editor_label"):
                    try:
                        rel_display = file_path.relative_to(self._current_codebase)
                    except ValueError:
                        rel_display = file_path
                    self.editor_label.setText(f"File: {rel_display}")
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
                # Fallback: show a reasonable window around the clicked line
                start_line = max(call.line - 5, 1)
                end_line = call.line + 40
                code = extract_source_segment(file_path, start_line, end_line)
                # Align editor line numbers with the original file
                self.editor.set_code(code, base_line=start_line)
            else:
                code = extract_source_segment(
                    func_loc.file_path, func_loc.start_line, func_loc.end_line
                )
                # Start line numbers at the function's first line (including decorators)
                self.editor.set_code(code, base_line=func_loc.start_line)

            if hasattr(self, "editor_label"):
                # Show the source file being displayed
                rel_path = file_path
                try:
                    rel_path = file_path.relative_to(self._current_codebase)
                except ValueError:
                    pass
                self.editor_label.setText(f"File: {rel_path}")
            return

        # Handle external I/O entries
        if kind == "io":
            fa: FileAccessView = obj
            src_file = fa.src_file
            src_line = fa.src_line
            src_func = fa.src_func

            if not src_file:
                return

            file_path = (self._current_codebase / src_file).resolve()
            if not file_path.exists():
                QtWidgets.QMessageBox.warning(
                    self,
                    "File Not Found",
                    f"Could not locate file that performed I/O:\n{file_path}",
                )
                return

            func_loc = None
            if src_line:
                func_loc = find_enclosing_function(
                    file_path=file_path, line_number=src_line, function_name=src_func or None
                )

            if not func_loc and src_line:
                # Fallback: show a window around the call site
                start_line = max(src_line - 5, 1)
                end_line = src_line + 40
                code = extract_source_segment(file_path, start_line, end_line)
                # Align editor line numbers with the original file
                self.editor.set_code(code, base_line=start_line)
            elif func_loc:
                code = extract_source_segment(
                    func_loc.file_path, func_loc.start_line, func_loc.end_line
                )
                # Start line numbers at the enclosing function's first line
                self.editor.set_code(code, base_line=func_loc.start_line)
            else:
                # If we have neither a function location nor a line, just show the top of the file
                code = extract_source_segment(file_path, 1, 80)
                # Top of file: line numbers start at 1
                self.editor.set_code(code, base_line=1)

            if hasattr(self, "editor_label"):
                try:
                    rel_path = file_path.relative_to(self._current_codebase)
                except ValueError:
                    rel_path = file_path
                self.editor_label.setText(f"File: {rel_path}")
            return

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

        This method now runs quietly unless there is actually work to do, so it
        does not spam the console on every shutdown.
        """
        # Trace worker
        if self._trace_worker is not None:
            if self._trace_worker.isRunning():
                self._trace_worker.quit()
                self._trace_worker.wait()
            self._trace_worker = None

        # LLM worker
        if self._llm_worker is not None:
            if self._llm_worker.isRunning():
                self._llm_worker.quit()
                self._llm_worker.wait()
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

                # Re-normalize depths so that they reflect relative nesting
                # within the traced call sequence rather than the raw stack
                # depth (which can include frames we do not show in the UI).
                if function_calls:
                    raw_depths = [fc.depth for fc in function_calls]
                    effective_depths: List[int] = []

                    # Start the first call at depth 0.
                    prev_eff = 0
                    effective_depths.append(prev_eff)

                    for i in range(1, len(raw_depths)):
                        prev_raw = raw_depths[i - 1]
                        curr_raw = raw_depths[i]
                        delta = curr_raw - prev_raw

                        if delta > 0:
                            # Any increase moves one level deeper, regardless
                            # of how many intermediate frames were present.
                            prev_eff = prev_eff + 1
                        elif delta == 0:
                            # Same depth: stay at the current level.
                            prev_eff = prev_eff
                        else:
                            # Decrease: move back up, but never below 0.
                            prev_eff = max(prev_eff + delta, 0)

                        effective_depths.append(prev_eff)

                    for fc, eff in zip(function_calls, effective_depths):
                        fc.depth = eff
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