import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
import ast

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.Qsci import QsciScintilla, QsciLexerPython

from .execution_tracer import MainExecutionTracer
from .code_utils import find_enclosing_function, extract_source_segment
from .llm_client import OpenRouterClient
from .llm_config_store import load_llm_config, save_llm_config


@dataclass
class FunctionCallView:
    index: int
    file: str
    function: str
    line: int
    depth: int
    kind: str  # "Func", "Class", or "Module"
    is_entry: bool = False  # True for the inferred entrypoint call


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

        # LLM + app configuration (loaded from app_config.json; required)
        self._llm_config: Dict[str, Any] = load_llm_config()

        # LLM client and configuration
        price_table = self._llm_config.get("model_prices") or {}
        config_model = str(self._llm_config.get("model") or "").strip()
        if not config_model:
            raise RuntimeError("LLM model must be specified in app_config.json['model']")

        # Determine initial prompt template from the default preset, if any.
        presets = self._llm_config.get("presets") or {}
        default_preset_id = self._llm_config.get("default_prompt_preset")
        initial_template: Optional[str] = None
        if default_preset_id and default_preset_id in presets:
            initial_template = presets[default_preset_id].get("template")

        temperature = float(self._llm_config.get("temperature", 0.10))
        max_tokens = self._llm_config.get("max_tokens")
        if not isinstance(max_tokens, int):
            max_tokens = None

        self._llm_client = OpenRouterClient(
            model=config_model,
            prompt_template=initial_template,
            max_tokens=max_tokens,
            temperature=temperature,
            price_table=price_table,
        )
        self._llm_model_override: Optional[str] = config_model
        self._llm_max_tokens: Optional[int] = max_tokens
        self._llm_temperature: float = temperature
        self._llm_prompt_template: Optional[str] = initial_template
        self._llm_presets: Dict[str, Dict[str, str]] = self._llm_config.get("presets", {}) or {}

        default_preset_id = self._llm_config.get("default_prompt_preset")
        if not default_preset_id or default_preset_id not in self._llm_presets:
            # Fallback to first preset ID if config is missing or invalid
            default_preset_id = next(iter(self._llm_presets.keys())) if self._llm_presets else None
        self._llm_current_preset_id: Optional[str] = default_preset_id

        # UI state: splitter sizes, dialog sizes, and viewer flags are stored
        # under the "ui" key in app_config.json.
        self._ui_state: Dict[str, Any] = dict(self._llm_config.get("ui", {}) or {})

        # Verbose logging flag controls whether prompts/responses are written
        # in detail to the LLM log file. It is stored inside the ui section.
        self._llm_verbose_logging: bool = bool(self._ui_state.get("verbose_logging", False))

        # Initialize overrides from config
        config_model = self._llm_config.get("model")
        if isinstance(config_model, str) and config_model:
            self._llm_model_override = config_model

        config_temp = self._llm_config.get("temperature")
        if isinstance(config_temp, (int, float)):
            self._llm_temperature = float(config_temp)

        config_max_tokens = self._llm_config.get("max_tokens")
        if isinstance(config_max_tokens, int):
            self._llm_max_tokens = config_max_tokens

        # Pick the current prompt template from the selected preset if available
        if self._llm_current_preset_id and self._llm_current_preset_id in self._llm_presets:
            self._llm_prompt_template = self._llm_presets[self._llm_current_preset_id].get("template")

        # Apply initial configuration to the LLM client
        if self._llm_model_override:
            self._llm_client.model = self._llm_model_override
        if self._llm_prompt_template:
            self._llm_client.prompt_template = self._llm_prompt_template
        if self._llm_max_tokens is not None:
            self._llm_client.max_tokens = self._llm_max_tokens
        self._llm_client.temperature = float(self._llm_temperature or 0.1)

        # Track last-used LLM context for building summary headers
        self._last_llm_kind: Optional[str] = None  # "function" or "path"
        self._last_llm_model: Optional[str] = None
        self._last_llm_preset_id: Optional[str] = None

        # Background workers
        self._trace_worker: Optional[_TraceWorker] = None
        self._llm_worker: Optional[_LLMSummaryWorker] = None

        self._build_ui()
        self._connect_signals()

    # UI construction -----------------------------------------------------

    def _build_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)

        # Main horizontal splitter: left (trees + summary) vs right (code editor)
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        main_layout.addWidget(self.main_splitter)

        # Left side: vertical split (top: controls + execution + I/O, bottom: summary)
        self.left_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, self.main_splitter)
        # Allow the left pane to become narrow; the actual minimum width is
        # governed by the child widgets (top_left and summary_container).
        self.left_splitter.setMinimumWidth(0)

        # Top-left container: codebase label, command row, combined function execution and file I/O
        top_left = QtWidgets.QWidget(self.left_splitter)
        # Permit the container to shrink but keep a small visible minimum so it does not vanish.
        top_left.setMinimumWidth(60)
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
        # Allow the tree itself to shrink as much as needed; the user can rely
        # on horizontal scrolling when columns no longer fit.
        self.left_tree.setMinimumWidth(0)
        # Columns:
        #   0: (indent)
        #   1: Order
        #   2: Depth
        #   3: Kind
        #   4: Phase  ("Import" vs "Runtime")
        #   5: Caller (file:line)
        #   6: File
        #   7: Function
        #   8: Line/Mode
        # Column 0 is a narrow, mostly empty column that holds the tree
        # indentation and expand/collapse icons so that the visible "Order"
        # numbers in column 1 are not pushed to the right by tree padding.
        self.left_tree.setHeaderLabels(
            ["", "Order", "Depth", "Kind", "Phase", "Caller", "File", "Function", "Line/Mode"]
        )
        # Approximate default column widths based on expected content:
        # - indent: tiny
        # - Order: up to 3 digits
        # - Depth: small integer
        # - Kind: e.g. "Module"
        # - Phase: "Import"/"Runtime"
        # - Caller: "src/file.py:123"
        # - File: "src/experiment_configs.py"
        # - Function: "VisualizationConfig"
        # - Line/Mode: 3-digit line or short mode
        default_col_widths = [14, 40, 40, 70, 70, 140, 210, 170, 60]
        for idx, w in enumerate(default_col_widths):
            if idx < self.left_tree.columnCount():
                self.left_tree.setColumnWidth(idx, w)
        # Enable interactive column resizing with a small minimum so the user
        # can compress the left pane without hitting an artificial floor.
        header = self.left_tree.header()
        header.setMinimumSectionSize(20)
        header.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        # Prefer smooth horizontal scrolling when columns overflow.
        self.left_tree.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        # If we have persisted column widths in the UI state, apply them now.
        col_widths = self._ui_state.get("left_tree_column_widths")
        if isinstance(col_widths, list):
            for idx, w in enumerate(col_widths):
                if (
                    isinstance(w, int)
                    and w > 0
                    and idx < self.left_tree.columnCount()
                ):
                    self.left_tree.setColumnWidth(idx, w)
        # Enable a custom context menu so we can offer "Copy tree to clipboard"
        self.left_tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        top_left_layout.addWidget(self.left_tree, stretch=1)

        # Bottom-left: summary area
        summary_container = QtWidgets.QWidget(self.left_splitter)
        # Keep a small minimum so the entire left pane does not collapse to zero width.
        summary_container.setMinimumWidth(60)
        summary_layout = QtWidgets.QVBoxLayout(summary_container)
        summary_layout.setContentsMargins(4, 4, 4, 4)
        summary_layout.setSpacing(4)

        self.summary_label = QtWidgets.QLabel("LLM Summary", summary_container)
        self.summary_label.setStyleSheet("font-weight: bold;")
        # Use a rich-text editor so that Markdown coming back from the LLM
        # (e.g. headings, bullet lists, code blocks) is easier to read.
        self.summary_text = QtWidgets.QTextEdit(summary_container)
        self.summary_text.setReadOnly(True)
        # Very compact button labels to fit comfortably on a single header row.
        self.summary_button = QtWidgets.QPushButton("Func", summary_container)
        self.summary_button.setToolTip("Summarize highlighted function")
        self.summary_path_button = QtWidgets.QPushButton("Path", summary_container)
        self.summary_path_button.setToolTip("Summarize execution path")
        self.summary_entrypoints_button = QtWidgets.QPushButton("Entry", summary_container)
        self.summary_entrypoints_button.setToolTip("Suggest likely entry points")
        self.summary_config_button = QtWidgets.QToolButton(summary_container)
        self.summary_config_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView)
        )
        self.summary_config_button.setToolTip("Open LLM Summary Settings")

        # Compact header row: label on the left, all LLM actions on the right.
        header_row = QtWidgets.QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(6)
        header_row.addWidget(self.summary_label)
        header_row.addStretch(1)
        header_row.addWidget(self.summary_button)
        header_row.addWidget(self.summary_path_button)
        header_row.addWidget(self.summary_entrypoints_button)
        header_row.addWidget(self.summary_config_button)
        summary_layout.addLayout(header_row)

        summary_layout.addWidget(self.summary_text, stretch=1)

        # Right side: container with label + code editor
        right_container = QtWidgets.QWidget(self.main_splitter)
        # Allow the right side to shrink as well; minimum is effectively governed
        # by its contents (editor, label), but we do not force a large floor here.
        right_container.setMinimumWidth(0)
        right_layout = QtWidgets.QVBoxLayout(right_container)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(4)

        self.editor_label = QtWidgets.QLabel("File: (none)", right_container)
        self.editor_label.setStyleSheet("font-weight: bold;")
        right_layout.addWidget(self.editor_label, stretch=0)

        self.editor = CodeEditor(right_container)
        right_layout.addWidget(self.editor, stretch=1)

        # Adjust splitter sizes: make summary vertically smaller. If we have
        # persisted sizes in the config, prefer those over the default stretch
        # factors so that the layout is restored across runs.
        if self._ui_state:
            main_sizes = self._ui_state.get("main_splitter_sizes")
            if isinstance(main_sizes, list) and all(isinstance(x, int) for x in main_sizes):
                self.main_splitter.setSizes(main_sizes)
            left_sizes = self._ui_state.get("left_splitter_sizes")
            if isinstance(left_sizes, list) and all(isinstance(x, int) for x in left_sizes):
                self.left_splitter.setSizes(left_sizes)
        else:
            self.main_splitter.setStretchFactor(0, 0)
            self.main_splitter.setStretchFactor(1, 1)
            self.left_splitter.setStretchFactor(0, 5)
            self.left_splitter.setStretchFactor(1, 1)

    def _connect_signals(self):
        self.left_tree.itemClicked.connect(self._on_left_item_clicked)
        # Right-click context menu on the left tree
        self.left_tree.customContextMenuRequested.connect(
            self._on_left_tree_context_menu
        )
        self.summary_button.clicked.connect(self._on_summarize_clicked)
        self.summary_path_button.clicked.connect(self._on_summarize_path_clicked)
        self.summary_entrypoints_button.clicked.connect(self._on_suggest_entrypoints_clicked)
        self.summary_config_button.clicked.connect(self._on_llm_config_button_clicked)
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
        self.summary_path_button.setDisabled(True)
        self.summary_entrypoints_button.setDisabled(True)
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
        self.summary_path_button.setDisabled(False)
        self.summary_entrypoints_button.setDisabled(False)
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
        #   4: Phase
        #   5: Caller
        #   6: File
        #   7: Function
        #   8: Line/Mode
        root_execution = QtWidgets.QTreeWidgetItem(
            self.left_tree, ["", "", "", "Exec", "", "", "Function Execution Order", ""]
        )
        root_execution.setExpanded(True)

        # Pre-compute caller and phase for each function call based on depth and kind.
        caller_strings: List[str] = []
        phase_strings: List[str] = []

        # Infer a single entrypoint: first depth-0 call in the sequence.
        entry_marked = False
        for fc in self._function_calls:
            fc.is_entry = False
        for fc in self._function_calls:
            if fc.depth == 0:
                fc.is_entry = True
                entry_marked = True
                break

        for i, call in enumerate(self._function_calls):
            # Phase: simple heuristic based on kind, but always treat the
            # inferred entrypoint as Runtime so it stands out from imports.
            if call.is_entry:
                phase_strings.append("Runtime")
            elif call.kind in ("Module", "Class"):
                phase_strings.append("Import")
            else:
                phase_strings.append("Runtime")

            caller_str = ""
            # Look backwards for the most recent shallower-depth call.
            for j in range(i - 1, -1, -1):
                prev = self._function_calls[j]
                if prev.depth < call.depth:
                    # Use file:line for the caller.
                    caller_line = prev.line if prev.line else 0
                    # Avoid showing meaningless "0" line numbers.
                    if prev.file and caller_line > 0:
                        caller_str = f"{prev.file}:{caller_line}"
                    elif prev.file:
                        caller_str = prev.file
                    break
            caller_strings.append(caller_str)

        for call, caller_str, phase_str in zip(self._function_calls, caller_strings, phase_strings):
            # Display a custom kind label for the inferred entrypoint.
            kind_label = "Entry" if call.is_entry else call.kind
            item = QtWidgets.QTreeWidgetItem(
                root_execution,
                [
                    "",                       # (indent only)
                    str(call.index),          # Order
                    str(call.depth),          # Depth
                    kind_label,               # Kind
                    phase_str,                # Phase ("Import" / "Runtime")
                    caller_str,               # Caller (inferred)
                    call.file,                # File
                    call.function,            # Function
                    str(call.line),           # Line
                ],
            )
            if call.is_entry:
                # Soft highlight for the entrypoint row.
                entry_color = QtGui.QColor("#e8f4ff")
                for col in range(0, 9):
                    item.setBackground(col, entry_color)
            # Store metadata for click handling
            item.setData(0, QtCore.Qt.UserRole, ("func", call))

        root_io = QtWidgets.QTreeWidgetItem(
            self.left_tree, ["", "", "", "I/O", "", "", "External File I/O", ""]
        )
        root_io.setExpanded(True)

        for fa in file_accesses:
            # For I/O rows, use the recorded src_file:src_line as the caller.
            if fa.src_file and fa.src_line:
                io_caller = f"{fa.src_file}:{fa.src_line}"
            elif fa.src_file:
                io_caller = fa.src_file
            else:
                io_caller = ""
            item = QtWidgets.QTreeWidgetItem(
                root_io,
                [
                    "",                       # (indent only)
                    str(fa.index),           # Order
                    "",                      # Depth (not applicable)
                    fa.mode,                 # Kind / mode
                    "",                      # Phase (not applicable)
                    io_caller,               # Caller (I/O source)
                    fa.file_path,            # File
                    f"{fa.src_file}:{fa.src_func}",  # Function context
                    str(fa.src_line) if fa.src_line else "",  # Line
                ],
            )
            item.setData(0, QtCore.Qt.UserRole, ("io", fa))

        # Apply persisted preference for hiding import-time calls, if any.
        try:
            ui_cfg = getattr(self, "_llm_config", {}) or {}
            ui_state = ui_cfg.get("ui") or {}
            hide_imports = bool(ui_state.get("hide_import_rows", False))
            if hide_imports:
                self.set_import_rows_hidden(True)
        except Exception:
            pass

        self.left_tree.expandAll()
        # Make the Order column just wide enough for its contents
        self.left_tree.resizeColumnToContents(1)

    def _on_trace_error(self, message: str):
        # Re-enable controls
        self.left_tree.setDisabled(False)
        self.summary_button.setDisabled(False)
        self.summary_path_button.setDisabled(False)
        self.summary_entrypoints_button.setDisabled(False)
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

        # Prefer console logging over GUI popups for trace errors
        print(f"[TraceViewerWidget] Trace failed: {message}")

    def _on_left_tree_context_menu(self, pos: QtCore.QPoint):
        """
        Show a context menu on the left tree with options such as copying the
        tree to the clipboard, opening the full source file in the right pane,
        and showing the call stack for a given row.
        """
        item = self.left_tree.itemAt(pos)
        payload = item.data(0, QtCore.Qt.UserRole) if item is not None else None

        menu = QtWidgets.QMenu(self.left_tree)
        copy_action = menu.addAction("Copy tree to clipboard")
        open_file_action = None
        show_stack_action = None
        # Only allow per-item actions when we actually have per-item metadata
        # (i.e., real function or I/O rows), not on the group headers.
        if payload:
            open_file_action = menu.addAction("Open full file in right pane")
            show_stack_action = menu.addAction("Show call stack for this row")

        selected = menu.exec_(self.left_tree.viewport().mapToGlobal(pos))
        if selected is copy_action:
            self._copy_tree_to_clipboard()
        elif open_file_action is not None and selected is open_file_action:
            self._open_full_file_for_item(item)
        elif show_stack_action is not None and selected is show_stack_action:
            self._show_call_stack_for_item(item)

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

    def _show_call_stack_for_item(self, item: QtWidgets.QTreeWidgetItem) -> None:
        """
        Show a simple call stack for the selected execution-row item, walking
        back through the function_calls list using the depth information.
        """
        payload = item.data(0, QtCore.Qt.UserRole)
        if not payload or payload[0] != "func":
            return

        _, call = payload
        # Find the index of this call in our sequence (index is 1-based).
        current_idx = max(int(call.index) - 1, 0)
        if not (0 <= current_idx < len(self._function_calls)):
            return

        stack_indices: List[int] = [current_idx]
        cur = current_idx
        # Walk backwards, adding the most recent shallower-depth ancestor each time.
        while True:
            current_call = self._function_calls[cur]
            found_ancestor = False
            for j in range(cur - 1, -1, -1):
                prev = self._function_calls[j]
                if prev.depth < current_call.depth:
                    stack_indices.append(j)
                    cur = j
                    found_ancestor = True
                    break
            if not found_ancestor:
                break

        # Build a human-readable stack from entry to the selected call.
        lines: List[str] = []
        for idx in reversed(stack_indices):
            fc = self._function_calls[idx]
            kind_label = "Entry" if fc.is_entry else fc.kind
            label = f"{idx+1:4d}: {kind_label}  {fc.file}::{fc.function}"
            if fc.line:
                label += f" (line {fc.line})"
            lines.append(label)

        text = "\n".join(lines) if lines else "No call stack available."

        dlg = QtWidgets.QMessageBox(self)
        dlg.setWindowTitle("Call Stack")
        dlg.setIcon(QtWidgets.QMessageBox.Information)
        # Show the entire call stack directly in the main text so there is no
        # extra click required to reveal the details.
        dlg.setText(text or "No call stack available.")
        dlg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        dlg.exec_()

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

    def _apply_llm_overrides(self) -> None:
        """
        Apply the current UI/config LLM overrides to the client instance.
        """
        if self._llm_model_override:
            self._llm_client.model = self._llm_model_override
        self._llm_client.max_tokens = self._llm_max_tokens
        self._llm_client.temperature = float(self._llm_temperature or 0.1)
        if self._llm_prompt_template:
            self._llm_client.prompt_template = self._llm_prompt_template

    def _build_path_context(self) -> Optional[str]:
        """
        Build a textual representation of the current execution path suitable
        for feeding into the LLM prompt {code} placeholder.

        We include a brief instruction asking the model to be concise so that
        responses are more likely to fit within the configured token budget.
        """
        if not self._function_calls:
            return None

        lines: List[str] = []
        lines.append(
            "Instruction: Provide a concise, high-level summary of this execution "
            "path that fits within the response token budget. Avoid repetition "
            "and unnecessary detail."
        )
        lines.append("")
        lines.append("Execution call path (in order):")
        for fc in self._function_calls:
            label = f"{fc.file}::{fc.function}" if fc.file else fc.function
            if fc.line:
                label += f" (line {fc.line})"
            label += f" depth={fc.depth}"
            lines.append(f"  {fc.index:4d}. {label}")

        return "\n".join(lines)

    def _build_entrypoints_context(self) -> Optional[str]:
        """
        Build a context string listing Python files in the current codebase and
        short snippets from each, suitable for the entrypoints preset.
        """
        if self._current_codebase is None:
            return None

        base = self._current_codebase
        py_files_raw = list(base.rglob("*.py"))
        # Filter out virtualenvs and common third-party / stdlib-style directories
        py_files = []
        skip_fragments = ("/.venv/", "\\\\.venv\\\\", "site-packages", "/lib/python", "\\\\Lib\\\\")
        for path in py_files_raw:
            path_str = str(path)
            if any(fragment in path_str for fragment in skip_fragments):
                continue
            py_files.append(path)

        if not py_files:
            return None

        max_files = 30
        max_lines_per_file = 80
        max_total_chars = 20000

        lines: List[str] = []
        lines.append(f"Codebase: {base.name}")
        lines.append("Collected Python file snippets (limited sample):")

        total_chars = 0
        for idx, path in enumerate(py_files):
            if idx >= max_files:
                lines.append("... (additional files omitted for brevity) ...")
                break
            try:
                rel = path.relative_to(base)
            except ValueError:
                rel = path
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            snippet_lines = text.splitlines()[:max_lines_per_file]
            snippet = "\n".join(snippet_lines)
            block = f"\n### {rel}\n```python\n{snippet}\n```"
            if total_chars + len(block) > max_total_chars:
                lines.append("... (truncated due to size limits) ...")
                break
            lines.append(block)
            total_chars += len(block)

        return "\n".join(lines)

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
        self.summary_path_button.setDisabled(True)
        self.summary_entrypoints_button.setDisabled(True)
        self.summary_text.setMarkdown("Requesting summary from OpenRouter...")

        if self._llm_worker is not None and self._llm_worker.isRunning():
            # Avoid starting multiple concurrent LLM requests
            return

        # Apply any overrides from the settings dialog to the LLM client.
        self._apply_llm_overrides()

        # Remember context for headers and logging
        self._last_llm_kind = "function"
        self._last_llm_model = self._llm_client.model
        self._last_llm_preset_id = self._llm_current_preset_id

        meta: Dict[str, Any] = {}
        if self._current_codebase is not None:
            meta["codebase"] = self._current_codebase.name
        if self._current_command:
            meta["command"] = self._current_command
        meta["kind"] = "function"
        meta["verbose_logging"] = bool(getattr(self, "_llm_verbose_logging", False))

        worker = _LLMSummaryWorker(code, self._llm_client, self._last_llm_preset_id, meta)
        worker.finished_with_result.connect(self._on_summary_finished)
        worker.error_occurred.connect(self._on_summary_error)
        self._llm_worker = worker
        worker.start()

    def _on_summarize_path_clicked(self):
        context = self._build_path_context()
        if not context:
            QtWidgets.QMessageBox.information(
                self,
                "No Trace",
                "There is no execution path to summarize. Run a trace first.",
            )
            return

        self.summary_button.setDisabled(True)
        self.summary_path_button.setDisabled(True)
        self.summary_entrypoints_button.setDisabled(True)
        self.summary_text.setMarkdown("Requesting path summary from OpenRouter...")

        if self._llm_worker is not None and self._llm_worker.isRunning():
            # Avoid starting multiple concurrent LLM requests
            return

        # Apply any overrides from the settings dialog to the LLM client.
        self._apply_llm_overrides()

        # For path summaries, allow a larger completion budget: use twice the
        # configured max_tokens if one is set, otherwise leave as-is.
        if self._llm_max_tokens is not None and self._llm_max_tokens > 0:
            self._llm_client.max_tokens = self._llm_max_tokens * 2

        # Remember context for headers and logging
        self._last_llm_kind = "path"
        self._last_llm_model = self._llm_client.model
        self._last_llm_preset_id = self._llm_current_preset_id

        meta: Dict[str, Any] = {}
        if self._current_codebase is not None:
            meta["codebase"] = self._current_codebase.name
        if self._current_command:
            meta["command"] = self._current_command
        meta["kind"] = "path"
        meta["verbose_logging"] = bool(getattr(self, "_llm_verbose_logging", False))

        worker = _LLMSummaryWorker(context, self._llm_client, self._last_llm_preset_id, meta)
        worker.finished_with_result.connect(self._on_summary_finished)
        worker.error_occurred.connect(self._on_summary_error)
        self._llm_worker = worker
        worker.start()

    def _on_suggest_entrypoints_clicked(self):
        context = self._build_entrypoints_context()
        if not context:
            QtWidgets.QMessageBox.information(
                self,
                "No Python Files",
                "No Python files were found in the current codebase.",
            )
            return

        self.summary_button.setDisabled(True)
        self.summary_path_button.setDisabled(True)
        self.summary_entrypoints_button.setDisabled(True)
        self.summary_text.setMarkdown("Requesting entrypoint suggestions from OpenRouter...")

        if self._llm_worker is not None and self._llm_worker.isRunning():
            # Avoid starting multiple concurrent LLM requests
            return

        # Apply any overrides from the settings dialog to the LLM client.
        self._apply_llm_overrides()

        # For this action, prefer the dedicated 'entrypoints' preset if available,
        # without changing the user's default preset selection.
        preset_id = "entrypoints" if "entrypoints" in self._llm_presets else self._llm_current_preset_id
        if preset_id and preset_id in self._llm_presets:
            tmpl = self._llm_presets[preset_id].get("template") or self._llm_client.prompt_template
            self._llm_client.prompt_template = tmpl

        # Remember context for headers and logging
        self._last_llm_kind = "entrypoints"
        self._last_llm_model = self._llm_client.model
        self._last_llm_preset_id = preset_id

        meta: Dict[str, Any] = {}
        if self._current_codebase is not None:
            meta["codebase"] = self._current_codebase.name
        if self._current_command:
            meta["command"] = self._current_command
        meta["kind"] = "entrypoints"
        meta["verbose_logging"] = bool(getattr(self, "_llm_verbose_logging", False))

        worker = _LLMSummaryWorker(context, self._llm_client, preset_id, meta)
        worker.finished_with_result.connect(self._on_summary_finished)
        worker.error_occurred.connect(self._on_summary_error)
        self._llm_worker = worker
        worker.start()

    def _on_summary_finished(self, text: str):
        self.summary_button.setDisabled(False)
        self.summary_path_button.setDisabled(False)
        self.summary_entrypoints_button.setDisabled(False)

        # Build a small header describing what was summarized and with which settings.
        if self._last_llm_kind == "function":
            kind_label = "Function"
        elif self._last_llm_kind == "path":
            kind_label = "Path"
        elif self._last_llm_kind:
            kind_label = self._last_llm_kind.capitalize()
        else:
            kind_label = "LLM"

        model = self._last_llm_model or self._llm_client.model
        preset = self._last_llm_preset_id or "-"
        header = f"**[{kind_label}]** `model={model}` `preset={preset}`"

        # Treat the body as Markdown so headings, lists, and code blocks are rendered.
        combined = f"{header}\n\n{text}"
        self.summary_text.setMarkdown(combined)
        self._llm_worker = None

    def _on_summary_error(self, message: str):
        self.summary_button.setDisabled(False)
        self.summary_path_button.setDisabled(False)
        self.summary_entrypoints_button.setDisabled(False)
        # Errors are simple text; render them as-is.
        self.summary_text.setPlainText(message)
        self._llm_worker = None

    def set_caller_column_visible(self, visible: bool) -> None:
        """
        Show or hide the Caller column in the left-hand tree.

        Column indices:
          0: (indent)
          1: Order
          2: Depth
          3: Kind
          4: Phase
          5: Caller
          6: File
          7: Function
          8: Line/Mode
        """
        # Column 5 is the Caller column.
        self.left_tree.setColumnHidden(5, not visible)

    def set_phase_column_visible(self, visible: bool) -> None:
        """
        Show or hide the Phase column ("Import" vs "Runtime") in the left-hand tree.
        """
        # Column 4 is the Phase column.
        self.left_tree.setColumnHidden(4, not visible)

    def set_import_rows_hidden(self, hidden: bool) -> None:
        """
        Show or hide rows that belong to the import-time phase in the execution
        tree. This operates on the Phase column value ("Import" / "Runtime").
        """
        # Phase column index
        phase_col = 4
        root = self.left_tree.invisibleRootItem()
        stack = [root]
        while stack:
            parent = stack.pop()
            for i in range(parent.childCount()):
                child = parent.child(i)
                stack.append(child)
                phase_text = child.text(phase_col)
                if phase_text == "Import":
                    child.setHidden(hidden)

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

    def save_ui_state(self) -> None:
        """
        Persist current splitter sizes, column widths, and verbose logging flag
        into app_config.json.
        """
        config = dict(self._llm_config or {})

        # Start from whatever UI state is currently stored in the config so we
        # do not drop keys such as "llm_dialog_size" that may have been written
        # by the main window.
        base_ui = {}
        if isinstance(self._llm_config, dict):
            base_ui = dict(self._llm_config.get("ui") or {})
        ui_state = base_ui

        # Persist verbose logging inside the ui section.
        ui_state["verbose_logging"] = bool(self._llm_verbose_logging)

        if hasattr(self, "main_splitter") and hasattr(self, "left_splitter"):
            try:
                ui_state["main_splitter_sizes"] = self.main_splitter.sizes()
                ui_state["left_splitter_sizes"] = self.left_splitter.sizes()
            except Exception:
                pass
        if hasattr(self, "left_tree"):
            try:
                col_count = self.left_tree.columnCount()
                ui_state["left_tree_column_widths"] = [
                    self.left_tree.columnWidth(i) for i in range(col_count)
                ]
            except Exception:
                pass
        config["ui"] = ui_state

        save_llm_config(config)
        self._llm_config = config
        self._ui_state = ui_state

    def _on_llm_config_button_clicked(self) -> None:
        """
        Open the LLM Summary Settings dialog via the parent main window.
        """
        parent = self.parent()
        if parent is not None and hasattr(parent, "_on_llm_settings"):
            try:
                parent._on_llm_settings()
            except Exception:
                pass

    def persist_llm_settings(
        self,
        model: Optional[str],
        max_tokens: Optional[int],
        temperature: float,
        preset_id: Optional[str],
        prompt_template: str,
    ) -> None:
        """
        Update in-memory LLM settings and persist them to llm_config.json.
        """
        if model:
            self._llm_model_override = model
        else:
            self._llm_model_override = None

        self._llm_max_tokens = max_tokens
        self._llm_temperature = float(temperature)
        self._llm_prompt_template = prompt_template or None

        if preset_id and preset_id in self._llm_presets:
            self._llm_current_preset_id = preset_id
            # Update the preset's template with any edits from the dialog
            self._llm_presets[preset_id]["template"] = prompt_template

        # Build config object for saving
        config = dict(self._llm_config or {})
        if self._llm_model_override:
            config["model"] = self._llm_model_override
        if self._llm_max_tokens is not None:
            config["max_tokens"] = int(self._llm_max_tokens)
        else:
            config["max_tokens"] = None
        config["temperature"] = float(self._llm_temperature)
        if self._llm_current_preset_id:
            config["default_prompt_preset"] = self._llm_current_preset_id
        config["presets"] = self._llm_presets

        # Start from whatever UI state is currently stored in the config so we
        # do not drop keys such as "llm_dialog_size" that may have been written
        # by the main window.
        base_ui = {}
        if isinstance(self._llm_config, dict):
            base_ui = dict(self._llm_config.get("ui") or {})
        ui_state = base_ui

        # Ensure verbose logging is persisted inside the ui section.
        ui_state["verbose_logging"] = bool(self._llm_verbose_logging)

        # Capture current splitter sizes so the layout and columns are restored on next run.
        if hasattr(self, "main_splitter") and hasattr(self, "left_splitter"):
            try:
                ui_state["main_splitter_sizes"] = self.main_splitter.sizes()
                ui_state["left_splitter_sizes"] = self.left_splitter.sizes()
            except Exception:
                pass
        if hasattr(self, "left_tree"):
            try:
                col_count = self.left_tree.columnCount()
                ui_state["left_tree_column_widths"] = [
                    self.left_tree.columnWidth(i) for i in range(col_count)
                ]
            except Exception:
                pass
        config["ui"] = ui_state

        save_llm_config(config)
        self._llm_config = config
        self._ui_state = ui_state


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

            # Simple classification cache to avoid reparsing the same file
            ast_cache: Dict[Path, ast.AST] = {}
            class_names_cache: Dict[Path, set] = {}

            def classify_kind(file_rel: str, func_name: str, line_no: int) -> str:
                """
                Classify the kind of call for UI purposes:

                  - "Module" for <module> entries
                  - "Class"  for class bodies AND constructor calls
                  - "Func"   otherwise

                A call is treated as "Class" if either:
                  * the enclosing AST node is a ClassDef, or
                  * the function name matches a class defined in the file.
                """
                if func_name == "<module>":
                    return "Module"

                if not file_rel:
                    return "Func"

                file_path = (self._codebase / file_rel).resolve()
                if not file_path.exists():
                    return "Func"

                # Load and cache the AST and class names for this file
                tree = ast_cache.get(file_path)
                if tree is None:
                    try:
                        source = file_path.read_text(encoding="utf-8")
                        tree = ast.parse(source, filename=str(file_path))
                    except Exception:
                        return "Func"
                    ast_cache[file_path] = tree

                    # Collect class names defined in this file for constructor classification
                    names = set()
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            names.add(node.name)
                    class_names_cache[file_path] = names

                class_names = class_names_cache.get(file_path, set())

                # If the function name matches a class name in the file, treat it
                # as a Class call (constructor) even if the enclosing node is not
                # directly the class body.
                if func_name in class_names:
                    return "Class"

                # If we have no line information, we cannot refine further.
                if not line_no:
                    return "Func"

                best_node = None
                best_span: Optional[int] = None  # length of span (end - start)

                for node in ast.walk(tree):
                    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        continue

                    start = getattr(node, "lineno", None)
                    if start is None:
                        continue

                    end = getattr(node, "end_lineno", None)
                    if not isinstance(end, int):
                        # Fallback: approximate by taking the max child lineno
                        max_line = start
                        for child in ast.walk(node):
                            child_line = getattr(child, "lineno", None)
                            if isinstance(child_line, int) and child_line > max_line:
                                max_line = child_line
                        end = max_line

                    if not (start <= line_no <= end):
                        continue

                    span = end - start
                    if best_node is None or span < (best_span if best_span is not None else span + 1):
                        best_node = node
                        best_span = span

                if isinstance(best_node, ast.ClassDef):
                    return "Class"
                return "Func"

            def _reorder_by_entrypoint(calls: List[FunctionCallView]) -> List[FunctionCallView]:
                """
                Ensure that the first row in the execution order corresponds to the
                detected entrypoint file (e.g. vgmini.py) when possible.

                We keep all calls, but rotate the list so that the earliest call
                from the entrypoint module appears first. Indices are then
                re-assigned in the new order.
                """
                if not calls:
                    return calls

                entry_point = getattr(tracer, "entry_point", None)
                if not entry_point:
                    return calls

                entry_name = Path(entry_point).name

                # Find the earliest call whose file matches the entrypoint module.
                idx_entry = None
                for i, fc in enumerate(calls):
                    file_name = Path(fc.file).name if fc.file else ""
                    if file_name == entry_name:
                        idx_entry = i
                        break

                if idx_entry is None or idx_entry == 0:
                    return calls

                reordered = calls[idx_entry:] + calls[:idx_entry]
                for new_index, fc in enumerate(reordered, start=1):
                    fc.index = new_index
                return reordered

            def _inject_entrypoint_row(calls: List[FunctionCallView]) -> List[FunctionCallView]:
                """
                Insert a synthetic entrypoint row at the top of the execution order.

                This ensures that the GUI always shows the detected entry script
                (e.g. vgmini.py) as the first row, even when no function calls are
                recorded from that file (for example when it only imports and
                delegates into src/ modules).
                """
                entry_point = getattr(tracer, "entry_point", None)
                if not entry_point or not calls:
                    return calls

                entry_name = Path(entry_point).name

                # Try to reuse an existing call from the entrypoint module if we have one.
                existing_idx = None
                for i, fc in enumerate(calls):
                    if Path(fc.file).name == entry_name:
                        existing_idx = i
                        break

                new_calls: List[FunctionCallView] = []

                if existing_idx is not None:
                    # Use the earliest recorded call from the entrypoint module as the root.
                    entry_call = calls[existing_idx]
                    entry_call.index = 1
                    entry_call.depth = 0
                    new_calls.append(entry_call)

                    remaining = calls[:existing_idx] + calls[existing_idx + 1 :]
                    for new_index, fc in enumerate(remaining, start=2):
                        fc.index = new_index
                        fc.depth = max(fc.depth + 1, 0)
                        new_calls.append(fc)
                    return new_calls

                # Otherwise, synthesize a module-level entry row.
                synthetic = FunctionCallView(
                    index=1,
                    file=entry_point,
                    function="<module>",
                    line=0,
                    depth=0,
                    kind="Module",
                )
                new_calls.append(synthetic)

                for new_index, fc in enumerate(calls, start=2):
                    fc.index = new_index
                    fc.depth = max(fc.depth + 1, 0)
                    new_calls.append(fc)

                return new_calls

            # Build function execution order from full call records if available
            calls_raw: List[Dict[str, Any]] = getattr(trace, "calls", []) or []
            if calls_raw:
                for idx, c in enumerate(calls_raw, start=1):
                    func_name = c.get("function", "")
                    # Filter out lambda functions
                    if func_name == "<lambda>":
                        continue
                    file_rel = c.get("file", "")
                    line_no = int(c.get("line", 0) or 0)
                    depth = int(c.get("depth", 0) or 0)
                    kind = classify_kind(file_rel, func_name, line_no)
                    function_calls.append(
                        FunctionCallView(
                            index=idx,
                            file=file_rel,
                            function=func_name,
                            line=line_no,
                            depth=depth,
                            kind=kind,
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

                        kind = classify_kind(file_part, func_name, line_no)
                        function_calls.append(
                            FunctionCallView(
                                index=idx,
                                file=file_part,
                                function=func_name,
                                line=line_no,
                                depth=0,
                                kind=kind,
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

                        kind = classify_kind(file_part, func_name, 0)
                        function_calls.append(
                            FunctionCallView(
                                index=idx,
                                file=file_part,
                                function=func_name,
                                line=0,
                                depth=0,
                                kind=kind,
                            )
                        )

            # Re-order so the entrypoint module appears first in the execution order.
            function_calls = _reorder_by_entrypoint(function_calls)

            # Inject an explicit entrypoint row so the GUI always shows the
            # entry script (e.g. vgmini.py) as the root of execution.
            function_calls = _inject_entrypoint_row(function_calls)

            # External file I/O with filtering (unchanged)
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

    def __init__(
        self,
        code: str,
        client: OpenRouterClient,
        preset_id: Optional[str],
        meta: Optional[Dict[str, Any]],
    ):
        super().__init__()
        self._code = code
        self._client = client
        self._preset_id = preset_id
        self._meta = meta or {}

    def run(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                text = loop.run_until_complete(
                    self._client.summarize_function(self._code, preset_id=self._preset_id, meta=self._meta)
                )
            finally:
                loop.close()
            self.finished_with_result.emit(text)
        except Exception as exc:
            # Ensure we always emit a signal so the QThread can shut down cleanly.
            self.error_occurred.emit(str(exc))