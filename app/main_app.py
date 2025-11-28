import sys
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

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

        # LLM settings dialog
        self.action_llm_settings = QtWidgets.QAction("LLM &Summary Settings...", self)
        self.action_llm_settings.triggered.connect(self._on_llm_settings)

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
        config_menu.addSeparator()
        config_menu.addAction(self.action_llm_settings)

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

    def _on_llm_settings(self):
        """
        Open a dialog allowing the user to configure LLM summary settings.
        """
        if not hasattr(self.viewer, "_llm_client"):
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("LLM Summary Settings")

        layout = QtWidgets.QFormLayout(dlg)

        # ------------------------------------------------------------------
        # Model: editable combo box pre-populated with reasonable defaults.
        # ------------------------------------------------------------------
        model_combo = QtWidgets.QComboBox(dlg)
        model_combo.setEditable(True)

        # A small set of low-cost / reasonable models for summarization.
        default_models = [
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "openrouter/auto",
            "mistralai/mistral-small",
            "mistralai/mistral-nemo",
            "anthropic/claude-3.5-haiku",
            "google/gemini-1.5-flash",
            "meta-llama/llama-3.1-8b-instruct",
            "meta-llama/llama-3.1-70b-instruct",
            "cohere/command-r-plus",
        ]
        for m in default_models:
            model_combo.addItem(m)

        current_model = getattr(self.viewer, "_llm_model_override", None) or getattr(
            self.viewer._llm_client, "model", ""
        )
        if current_model and current_model not in default_models:
            model_combo.addItem(current_model)
        if current_model:
            idx = model_combo.findText(current_model)
            if idx >= 0:
                model_combo.setCurrentIndex(idx)
            else:
                model_combo.setEditText(current_model)
        layout.addRow("Model ID:", model_combo)

        # ------------------------------------------------------------------
        # Max tokens
        # ------------------------------------------------------------------
        max_tokens_edit = QtWidgets.QLineEdit(dlg)
        if getattr(self.viewer, "_llm_max_tokens", None) is not None:
            max_tokens_edit.setText(str(self.viewer._llm_max_tokens))
        layout.addRow("Max tokens (optional):", max_tokens_edit)

        # ------------------------------------------------------------------
        # Temperature: slider + live readout label
        # ------------------------------------------------------------------
        temp_container = QtWidgets.QWidget(dlg)
        temp_layout = QtWidgets.QHBoxLayout(temp_container)
        temp_layout.setContentsMargins(0, 0, 0, 0)

        temp_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, temp_container)
        temp_slider.setMinimum(0)
        temp_slider.setMaximum(100)  # 0.00 .. 1.00 in 0.01 steps
        current_temp = float(getattr(self.viewer, "_llm_temperature", 0.1))
        current_temp = max(0.0, min(1.0, current_temp))
        temp_slider.setValue(int(round(current_temp * 100)))

        temp_label = QtWidgets.QLabel(f"{current_temp:.2f}", temp_container)
        temp_label.setMinimumWidth(40)

        def _on_temp_changed(value: int):
            temp_label.setText(f"{value / 100.0:.2f}")

        temp_slider.valueChanged.connect(_on_temp_changed)

        temp_layout.addWidget(temp_slider, stretch=1)
        temp_layout.addWidget(temp_label, stretch=0)

        layout.addRow("Temperature:", temp_container)

        # ------------------------------------------------------------------
        # Prompt template: preset combo + editable text box
        # ------------------------------------------------------------------
        prompt_presets = {
            "Concise technical summary": (
                "You are an expert Python engineer. Summarize the purpose and behavior "
                "of the following function in concise, technical prose. Focus on:\n"
                "- Overall purpose\n"
                "- Key inputs and outputs\n"
                "- Important side effects (I/O, network, database, etc.)\n"
                "- Non-obvious edge cases or constraints\n\n"
                "Function source:\n"
                "```python\n"
                "{code}\n"
                "```"
            ),
            "High-level explanation for a new team member": (
                "You are helping onboard a new engineer to this codebase. Explain what "
                "the following function does in clear, approachable language. Focus on:\n"
                "- What problem it solves in the overall system\n"
                "- How it fits into the execution flow\n"
                "- Any assumptions or preconditions\n"
                "- Gotchas or areas where changes are risky\n\n"
                "Function source:\n"
                "```python\n"
                "{code}\n"
                "```"
            ),
            "Behavior + inputs/outputs only": (
                "Summarize the behavior of the following function, focusing strictly on:\n"
                "- Inputs (parameters and important global state)\n"
                "- Outputs (return values and changes to state)\n"
                "- Invariants the function relies on\n\n"
                "Avoid restating the code line-by-line.\n\n"
                "Function source:\n"
                "```python\n"
                "{code}\n"
                "```"
            ),
            "Potential bugs / edge cases": (
                "Review the following function for potential bugs and edge cases. "
                "Provide a short explanation that covers:\n"
                "- Any obvious or likely bugs\n"
                "- Edge cases that might fail (e.g., empty inputs, None, large values)\n"
                "- Error handling or lack thereof\n"
                "- Suggestions for tests that should be added\n\n"
                "Function source:\n"
                "```python\n"
                "{code}\n"
                "```"
            ),
        }

        prompt_combo = QtWidgets.QComboBox(dlg)
        prompt_combo.setEditable(False)
        for name in prompt_presets.keys():
            prompt_combo.addItem(name)
        prompt_combo.addItem("Custom")

        prompt_edit = QtWidgets.QPlainTextEdit(dlg)
        current_prompt = getattr(self.viewer, "_llm_prompt_template", None)
        if not current_prompt:
            current_prompt = getattr(self.viewer._llm_client, "prompt_template", "")
        prompt_edit.setPlainText(current_prompt)

        # Try to match current prompt to one of the presets.
        initial_index = prompt_combo.findText("Custom")
        for i, (name, template) in enumerate(prompt_presets.items()):
            if current_prompt.strip() == template.strip():
                initial_index = i
                break
        prompt_combo.setCurrentIndex(initial_index)

        def _on_preset_changed(index: int):
            name = prompt_combo.itemText(index)
            if name in prompt_presets:
                prompt_edit.setPlainText(prompt_presets[name])

        prompt_combo.currentIndexChanged.connect(_on_preset_changed)

        layout.addRow("Prompt preset:", prompt_combo)
        layout.addRow("Prompt template (use {code} placeholder):", prompt_edit)

        # ------------------------------------------------------------------
        # Buttons
        # ------------------------------------------------------------------
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=dlg,
        )
        layout.addRow(button_box)

        def _on_accept():
            # Apply changes back to the viewer; the next summarize call will
            # push these into the LLM client.

            # Model
            model_text = model_combo.currentText().strip()
            self.viewer._llm_model_override = model_text or None

            # Max tokens
            max_tokens_text = max_tokens_edit.text().strip()
            if max_tokens_text:
                try:
                    self.viewer._llm_max_tokens = max(0, int(max_tokens_text))
                except ValueError:
                    self.viewer._llm_max_tokens = None
            else:
                self.viewer._llm_max_tokens = None

            # Temperature from slider
            slider_value = temp_slider.value()
            self.viewer._llm_temperature = slider_value / 100.0

            # Prompt template
            prompt_text = prompt_edit.toPlainText().strip()
            self.viewer._llm_prompt_template = prompt_text or None

            dlg.accept()

        def _on_reject():
            dlg.reject()

        button_box.accepted.connect(_on_accept)
        button_box.rejected.connect(_on_reject)

        dlg.exec_()

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