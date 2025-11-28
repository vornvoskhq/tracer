import sys
from pathlib import Path
from typing import Optional

from PyQt5 import QtCore, QtWidgets

from .trace_viewer import TraceViewerWidget
from .llm_config_store import save_llm_config, DEFAULT_CONFIG


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, initial_codebase: Path | None = None):
        super().__init__()

        # Basic window properties; size will be adjusted if persisted UI config is present.
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

        # Verbose LLM logging toggle
        self.action_llm_verbose_logging = QtWidgets.QAction("Verbose &LLM Logging", self)
        self.action_llm_verbose_logging.setCheckable(True)
        if hasattr(self, "viewer") and hasattr(self.viewer, "_llm_verbose_logging"):
            self.action_llm_verbose_logging.setChecked(bool(self.viewer._llm_verbose_logging))
        self.action_llm_verbose_logging.toggled.connect(self._on_toggle_llm_verbose_logging)

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
        # Place verbose logging before the settings dialog so that
        # "LLM Summary Settings..." appears last in the menu.
        config_menu.addAction(self.action_llm_verbose_logging)
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

    def _on_toggle_llm_verbose_logging(self, checked: bool):
        """
        Toggle verbose LLM logging (whether to include full file contents in logs).
        """
        if hasattr(self.viewer, "_llm_verbose_logging"):
            self.viewer._llm_verbose_logging = bool(checked)
        # Update config via the viewer helper so it is persisted.
        if hasattr(self.viewer, "save_ui_state"):
            self.viewer.save_ui_state()

    def _on_llm_settings(self):
        """
        Open a dialog allowing the user to configure LLM summary settings.
        """
        if not hasattr(self.viewer, "_llm_client"):
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("LLM Summary Settings")

        # Current app configuration (including any persisted UI state).
        cfg = getattr(self.viewer, "_llm_config", {}) or {}

        layout = QtWidgets.QFormLayout(dlg)

        # ------------------------------------------------------------------
        # Model: editable combo box pre-populated with configurable defaults.
        # ------------------------------------------------------------------
        model_combo = QtWidgets.QComboBox(dlg)
        model_combo.setEditable(True)

        # Load the list of known models from the shared app config, falling
        # back to the DEFAULT_CONFIG models if none are stored yet.
        cfg_models = []
        try:
            cfg_models = list((cfg.get("models") or []))  # type: ignore[assignment]
        except Exception:
            cfg_models = []

        if not cfg_models:
            cfg_models = list(DEFAULT_CONFIG.get("models", []))

        for m in cfg_models:
            model_combo.addItem(str(m))

        current_model = getattr(self.viewer, "_llm_model_override", None) or getattr(
            self.viewer._llm_client, "model", ""
        )
        # Ensure the current model appears in the list even if it was not
        # present in the stored configuration.
        if current_model and current_model not in cfg_models:
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
        prompt_combo = QtWidgets.QComboBox(dlg)
        prompt_combo.setEditable(False)

        # Pull presets from the viewer's configuration
        presets = getattr(self.viewer, "_llm_presets", {}) or {}
        current_preset_id = getattr(self.viewer, "_llm_current_preset_id", None)
        preset_ids_by_index = []
        labels = []

        for pid, config in presets.items():
            label = str(config.get("label", pid))
            labels.append(label)
            preset_ids_by_index.append(pid)
            prompt_combo.addItem(label)

        # Add a generic "Custom" entry
        prompt_combo.addItem("Custom")
        custom_index = prompt_combo.count() - 1

        prompt_edit = QtWidgets.QPlainTextEdit(dlg)
        current_prompt = getattr(self.viewer, "_llm_prompt_template", None)
        if not current_prompt and current_preset_id and current_preset_id in presets:
            current_prompt = presets[current_preset_id].get("template", "")
        if not current_prompt:
            current_prompt = getattr(self.viewer._llm_client, "prompt_template", "")
        prompt_edit.setPlainText(current_prompt)

        # Select the current preset if we can match it; otherwise fall back to "Custom".
        initial_index = custom_index
        if current_preset_id and current_preset_id in presets:
            for idx, pid in enumerate(preset_ids_by_index):
                if pid == current_preset_id:
                    initial_index = idx
                    break
        prompt_combo.setCurrentIndex(initial_index)

        def _on_preset_changed(index: int):
            # If a known preset is selected, update the template editor from it.
            if 0 <= index < len(preset_ids_by_index):
                pid = preset_ids_by_index[index]
                cfg = presets.get(pid) or {}
                tmpl = cfg.get("template", "")
                if tmpl:
                    prompt_edit.setPlainText(tmpl)
            # If "Custom" is selected, leave the text as-is.

        prompt_combo.currentIndexChanged.connect(_on_preset_changed)

        layout.addRow("Prompt preset:", prompt_combo)
        # Use a two-line label so the dialog header does not stretch excessively.
        layout.addRow("Prompt template\n(use {code} placeholder):", prompt_edit)

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
            # push these into the LLM client and be persisted to app_config.json.

            # Model
            model_text = model_combo.currentText().strip() or None
            # Max tokens
            max_tokens_value: Optional[int]
            max_tokens_text = max_tokens_edit.text().strip()
            if max_tokens_text:
                try:
                    max_tokens_value = max(0, int(max_tokens_text))
                except ValueError:
                    max_tokens_value = None
            else:
                max_tokens_value = None

            # Temperature from slider
            slider_value = temp_slider.value()
            temperature_value = slider_value / 100.0

            # Prompt template
            prompt_text = prompt_edit.toPlainText()

            # Determine selected preset ID (if any)
            index = prompt_combo.currentIndex()
            preset_id: Optional[str] = None
            if 0 <= index < len(preset_ids_by_index):
                preset_id = preset_ids_by_index[index]
            # If "Custom" is selected, we leave preset_id as None; the template
            # will still be used, but it won't overwrite a named preset.

            if hasattr(self.viewer, "persist_llm_settings"):
                self.viewer.persist_llm_settings(
                    model=model_text,
                    max_tokens=max_tokens_value,
                    temperature=temperature_value,
                    preset_id=preset_id,
                    prompt_template=prompt_text,
                )

            # Persist dialog size and model list for next time.
            try:
                size = dlg.size()
                cfg = getattr(self.viewer, "_llm_config", {}) or {}
                ui_state = dict(cfg.get("ui") or {})
                ui_state["llm_dialog_size"] = [int(size.width()), int(size.height())]
                cfg["ui"] = ui_state

                # Update stored model list from the combo box contents so the
                # list of models is kept in the shared app_config.json.
                models_from_ui = []
                for i in range(model_combo.count()):
                    text = model_combo.itemText(i).strip()
                    if text and text not in models_from_ui:
                        models_from_ui.append(text)
                if models_from_ui:
                    cfg["models"] = models_from_ui

                save_llm_config(cfg)
                # Keep the viewer's cached config and UI state in sync with what
                # we just wrote so that subsequent save_ui_state() calls do not
                # drop the dialog size or models.
                self.viewer._llm_config = cfg
                if hasattr(self.viewer, "_ui_state"):
                    self.viewer._ui_state = dict(cfg.get("ui") or {})
            except Exception:
                pass

            dlg.accept()

        def _on_reject():
            # Persist dialog size and model list even on cancel so the layout is sticky.
            try:
                size = dlg.size()
                cfg = getattr(self.viewer, "_llm_config", {}) or {}
                ui_state = dict(cfg.get("ui") or {})
                ui_state["llm_dialog_size"] = [int(size.width()), int(size.height())]
                cfg["ui"] = ui_state

                models_from_ui = []
                for i in range(model_combo.count()):
                    text = model_combo.itemText(i).strip()
                    if text and text not in models_from_ui:
                        models_from_ui.append(text)
                if models_from_ui:
                    cfg["models"] = models_from_ui

                save_llm_config(cfg)
                self.viewer._llm_config = cfg
                if hasattr(self.viewer, "_ui_state"):
                    self.viewer._ui_state = dict(cfg.get("ui") or {})
            except Exception:
                pass
            dlg.reject()

        button_box.accepted.connect(_on_accept)
        # Route all dialog rejections (Cancel button, ESC, window close \"X\") through
        # a single handler so the dialog size is always persisted.
        dlg.rejected.connect(_on_reject)

        # Restore last dialog size if we have one stored in config, after the
        # layout is constructed so Qt honors the requested size.
        try:
            ui_state = cfg.get("ui") or {}
            dlg_size = ui_state.get("llm_dialog_size")
            if isinstance(dlg_size, list) and len(dlg_size) == 2:
                w, h = int(dlg_size[0]), int(dlg_size[1])
                if w > 0 and h > 0:
                    dlg.resize(w, h)
        except Exception:
            pass

        dlg.exec_()

    def closeEvent(self, event):
        """
        Ensure background threads are stopped cleanly before the app exits,
        and persist the current UI layout (splitter sizes, etc.) to the
        shared app_config.json file.
        """
        if hasattr(self, "viewer"):
            if hasattr(self.viewer, "save_ui_state"):
                try:
                    self.viewer.save_ui_state()
                except Exception:
                    pass
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