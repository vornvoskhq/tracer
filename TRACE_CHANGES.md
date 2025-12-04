# Tracer Changes (Session Summary)

This document summarizes the changes made to the tracer application in this session, and suggests potential future improvements.

---

## Changes Implemented

### 1. Robust Entry Point Detection

**New file:** `app/entry_detector.py`

Introduced a dedicated module for entry-point detection with two main functions:

- `detect_entry_from_command(command: str, target_dir: Path, codebase_name: str) -> tuple[Optional[str], list[str]]`
  - Parses the Run Trace command using `shlex.split`.
  - Handles:
    - `python script.py [args...]`
    - `python -m package.module [args...]`
    - Direct script execution: `./script.py` or `script.py`
    - Executable wrappers in `target_dir` (symlink or Python shebang scripts).
  - Returns:
    - a specific entry file (relative to `target_dir`) and the argument list, or
    - `(None, tokens)` if it cannot determine an entry point from the command (no guessing).

- `detect_entry_from_conventions(target_dir: Path, codebase_name: str) -> str`
  - Checks only conventional names in the codebase root:
    - `<codebase_name>.py`
    - `main.py`, `app.py`, `run.py`, `__main__.py`
    - `<codebase_name>`, `run`, `start` (executables)
  - Returns the filename if found, otherwise raises `FileNotFoundError`.
  - **No "largest .py file" fallback.**

Includes helpers:

- `_is_python_executable`
- `_is_within_target`
- `_resolve_script`
- `_resolve_module`

These helpers ensure that only files within the target codebase are considered valid entry points.

---

### 2. Execution Tracer Integration with Entry Detector

**Modified file:** `app/execution_tracer.py`

#### a. Imports

Added:

```python
from .entry_detector import detect_entry_from_command, detect_entry_from_conventions
```

#### b. Optional auto-detected `entry_point`

In `MainExecutionTracer.__init__`:

- Previously: `self.entry_point = self.detect_entry_point()`
- Now:

  ```python
  try:
      self.entry_point = self.detect_entry_point()
  except FileNotFoundError:
      self.entry_point = None
  ```

This keeps automatic detection for informational purposes (e.g. `--list-codebases`), but does not make it a hard requirement: the runtime entry point is determined from the command.

#### c. Conventional-only `detect_entry_point`

Replaced the prior heuristic (which fell back to the "largest .py file") with:

```python
def detect_entry_point(self) -> str:
    """
    Auto-detect a conventional main entry point for the codebase.

    This uses only conventional filenames and does not fall back to
    heuristics like "largest Python file". It is primarily used for
    informational purposes (e.g. --list-codebases).
    """
    return detect_entry_from_conventions(self.target_dir, self.codebase_name)
```

This is used in non-critical paths such as `--list-codebases`, and never guesses based on file size.

#### d. Command-driven `trace_command`

Rewrote `trace_command` to derive the entry point directly from the user-provided command:

```python
def trace_command(self, command: str) -> ExecutionTrace:
    """
    Trace a command by first determining a precise entry point.

    The entry point is derived from the command itself (python script.py,
    python -m package.module, ./script.py, or an executable wrapper inside
    the target directory). If no clear entry point can be determined from
    the command, this method raises an error instead of guessing.
    """
    # Determine entry point and arguments from the command
    entry_point, args_for_entry = detect_entry_from_command(
        command, self.target_dir, self.codebase_name
    )
    if entry_point is None:
        raise RuntimeError(
            f"Could not determine a Python entry point from command: {command!r}. "
            f"Please provide a command that directly runs the Python script "
            f"(e.g. 'python main.py') or an executable Python script inside "
            f"the target codebase."
        )

    # Compact summary of what is being traced, to reduce console clutter.
    print(
        f"TRACE | cmd={command} | "
        f"codebase={self.codebase_name} | "
        f"entry={entry_point} | "
        f"python={self.target_python}"
    )

    # Create tracer script (using standalone template helper)
    tracer_script = build_tracer_script(
        str(self.target_dir),
        self.codebase_name,
        entry_point,
        args_for_entry,
    )

    ...
```

Key behavior:

- For `python reporter_cli.py`, the entry point is now definitively `reporter_cli.py`, not some unrelated file.
- If no clear Python entry file can be inferred from the command (e.g. a complex wrapper), the tracer fails loudly with a clear error instead of silently guessing.

---

### 3. Entry Script Shown Properly in Traces

The dynamic tracer script now compiles the entry source with the real filename, so the entry script/module appears as a first-class participant in the trace (with correct file and line numbers).

Within the `TRACER_SCRIPT_TEMPLATE` in `app/execution_tracer.py`, the execution block was changed from:

```python
# Execute entry point by running the file content with __name__ == "__main__"
print(f"📚 Loading $ENTRY_POINT source...")
with open("$ENTRY_POINT", "r") as f:
    source_code = f.read()

print(f"🚀 Executing $ENTRY_POINT with __name__ == '__main__'...")
print("=" * 60)

# Create execution environment that simulates running the entry point directly
exec_globals = {
    "__name__": "__main__",
    "__file__": os.path.abspath("$ENTRY_POINT"),
    "__package__": None,
}

# Execute source code
exec(source_code, exec_globals)
```

to:

```python
# Execute entry point by running the file content with __name__ == "__main__"
print(f"📚 Loading $ENTRY_POINT source...")
with open("$ENTRY_POINT", "r") as f:
    source_code = f.read()

print(f"🚀 Executing $ENTRY_POINT with __name__ == '__main__'...")
print("=" * 60)

# Create execution environment that simulates running the entry point directly
exec_globals = {
    "__name__": "__main__",
    "__file__": os.path.abspath("$ENTRY_POINT"),
    "__package__": None,
}

# Compile with the real filename so the entry module appears distinctly in traces
code_obj = compile(source_code, os.path.abspath("$ENTRY_POINT"), "exec")

# Execute source code
exec(code_obj, exec_globals)
```

This ensures that:

- `frame.f_code.co_filename` for entry-script code points to the actual entry file path.
- The trace tree can show the entry script as a caller (e.g. `reporter_cli.py::<module>:line`) rather than skipping directly to imported modules.

---

## Suggested Future Improvements

Here are five concrete improvements that would further strengthen the tracer and its usability.

### 1. Explicit `--entry` Override and Validation

Add a CLI flag to specify the entry point explicitly:

```bash
trcr --entry reporter_cli.py "python reporter_cli.py --some-arg"
```

Implementation:

- Extend `main()` in `app/execution_tracer.py`:

  ```python
  parser.add_argument(
      "--entry",
      help="Explicit entry point file relative to target dir (e.g. 'reporter_cli.py')",
  )
  ```

- In `trace_command`, if an explicit `entry` is provided:
  - Validate the file exists under `target_dir`.
  - Use it directly, bypassing `detect_entry_from_command`.

Benefit:

- Provides a precise escape hatch for complex setups (multi-layer wrappers, custom launchers) without reintroducing heuristics.

---

### 2. Separate Trace Reporting and Core Tracing Logic

Currently `app/execution_tracer.py` still contains:

- Tracer script generation
- Subprocess orchestration
- Data collection
- Report formatting (`format_trace_report`)

Refactor by introducing `trace_report.py`:

- Move `format_trace_report` (and any future reporting logic) into `app/trace_report.py`.
- Keep `execution_tracer.py` focused on:
  - entry/command detection
  - running the trace
  - collecting raw data

Benefits:

- Smaller, more maintainable modules.
- Easier to add alternative views or export formats (e.g. JSON/Graphviz) without touching core tracing logic.

---

### 3. Configurable Trace Filters and Patterns

Right now, `should_trace` uses hard-coded patterns:

- Inclusion: `$CODEBASE_NAME_LOWER`, `src/`, `$TARGET_DIR`, specific names.
- Exclusion: `site-packages`, `__pycache__`, `.venv`, etc.

Improvements:

- Move tracing patterns into a config file (e.g. `app_config.json`) or a dedicated `trace_config.py`.
- Expose options such as:
  - `include_paths`, `exclude_paths`
  - `max_depth`
  - `include_entry_module` (explicit control over tracing the entry module)

Benefits:

- Users with different project layouts (monorepos, nested src, multiple apps) can tune tracing without modifying code.
- Less risk of hard-coded patterns missing valid project files or including noisy ones.

---

### 4. Richer Tree View with Explicit Root Entry Node

Enhance the UI/formatting layer (`TraceViewerWidget` and/or `format_trace_report`) to:

- Always render an explicit root node for the entry script/module, even if the first traced call is in another module.
- Show the entry node as:
  - `Entry: reporter_cli.py::<module> (line X)`
- Treat module-import time execution in the entry script as its own step in the visual tree.

Implementation directions:

- Once the entry script is correctly identified and its frames have the correct filename (now supported via `compile(..., filename, "exec")`), the viewer can:
  - Group calls by root script frame.
  - Show `__main__` module as a distinct node at depth 0.

Benefit:

- Clarifies the origin of the execution flow, aligning the visual trace with how you actually run the program (e.g. `python reporter_cli.py`).

---

### 5. Exportable Trace Data (JSON/Graph-friendly)

The tracer already writes `trace_output.json` inside the target codebase during execution and then reads it back. Consider:

- Adding a stable, documented JSON schema for this trace file.
- Exposing a CLI flag, e.g.:

  ```bash
  trcr "python reporter_cli.py" --export trace_raw.json
  ```

  that copies or writes the raw trace to a specified path without deleting it.

- Optionally add a helper script to convert this JSON into:
  - DOT/Graphviz
  - A simple HTML report
  - A call graph format consumable by other tools

Benefits:

- Enables further offline analysis of call graphs.
- Makes the tracer more useful in automated pipelines (CI, profiling, regression detection).

---

If you’d like, the next steps could be:

- Implementing the `--entry` override and wiring it into `MainExecutionTracer`.
- Extracting `format_trace_report` into its own module and adding a simple JSON export mode.