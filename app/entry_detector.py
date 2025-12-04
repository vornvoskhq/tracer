from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import List, Optional, Tuple


def _is_python_executable(token: str) -> bool:
    """Return True if the token looks like a Python interpreter."""
    name = os.path.basename(token)
    return name.startswith("python")


def _is_within_target(file_path: Path, target_dir: Path) -> bool:
    """Check that file_path is inside target_dir."""
    try:
        file_path.relative_to(target_dir)
        return True
    except ValueError:
        return False


def _resolve_script(
    script_token: str,
    args: List[str],
    target_dir: Path,
) -> Tuple[Optional[str], List[str]]:
    """
    Resolve a script token to a Python file inside target_dir.

    Returns (entry_point_relative_to_target, args) or (None, args) if not resolvable.
    """
    script_path = Path(script_token)

    if not script_path.is_absolute():
        candidate = (target_dir / script_path).resolve()
    else:
        candidate = script_path

    if candidate.exists() and candidate.is_file() and _is_within_target(candidate, target_dir):
        rel = candidate.relative_to(target_dir)
        return str(rel), args

    # Not resolvable -> no entry point from this token
    return None, args


def _resolve_module(module_name: str, target_dir: Path) -> str:
    """
    Resolve a Python module name (for `python -m`) to a file inside target_dir.

    Returns the entry point relative to target_dir or raises FileNotFoundError.
    """
    module_path = module_name.replace(".", "/")
    candidate_file = target_dir / f"{module_path}.py"
    candidate_pkg_main = target_dir / module_path / "__main__.py"

    for candidate in (candidate_file, candidate_pkg_main):
        if candidate.exists() and candidate.is_file() and _is_within_target(candidate, target_dir):
            return str(candidate.relative_to(target_dir))

    raise FileNotFoundError(
        f"Could not resolve module '{module_name}' to a Python file in {target_dir}"
    )


def detect_entry_from_conventions(target_dir: Path, codebase_name: str) -> str:
    """
    Detect an entry point using conventional filenames only.

    This is used for informational purposes (e.g. --list-codebases). It does not
    perform heuristic fallbacks like "largest .py file".
    """
    possible_entry_points = [
        target_dir / f"{codebase_name}.py",
        target_dir / "main.py",
        target_dir / "app.py",
        target_dir / "run.py",
        target_dir / "__main__.py",
        target_dir / f"{codebase_name}",
        target_dir / "run",
        target_dir / "start",
    ]

    for entry_point in possible_entry_points:
        if entry_point.exists():
            return entry_point.name

    raise FileNotFoundError(f"No conventional entry point found for {codebase_name} in {target_dir}")


def detect_entry_from_command(
    command: str,
    target_dir: Path,
    codebase_name: str,
) -> Tuple[Optional[str], List[str]]:
    """
    Detect the entry point and arguments from the Run Trace command.

    This function is deliberately conservative: it only returns an entry point
    when there is clear, direct evidence from the command. Otherwise it
    returns (None, tokens) and the caller should decide how to handle that
    (usually by raising an error rather than guessing).
    """
    tokens = shlex.split(command)
    if not tokens:
        return None, []

    t0 = tokens[0]

    # Case 1: python script.py ...
    if _is_python_executable(t0) and len(tokens) >= 2 and tokens[1].endswith(".py"):
        script = tokens[1]
        entry, args = _resolve_script(script, tokens[2:], target_dir)
        return entry, args

    # Case 2: python -m package.module ...
    if _is_python_executable(t0) and "-m" in tokens:
        idx = tokens.index("-m")
        if idx + 1 >= len(tokens):
            raise ValueError("python -m used without a module name")
        module = tokens[idx + 1]
        args = tokens[idx + 2:]
        entry = _resolve_module(module, target_dir)
        return entry, args

    # Case 3: direct script ./foo.py or foo.py
    if t0.endswith(".py") or "/" in t0:
        entry, args = _resolve_script(t0, tokens[1:], target_dir)
        return entry, args

    # Case 4: executable wrapper in target dir (symlink or Python script)
    candidate = (target_dir / t0).resolve()
    if candidate.exists() and candidate.is_file() and _is_within_target(candidate, target_dir):
        # Symlink directly to a .py under target_dir
        if candidate.is_symlink():
            target = candidate.resolve()
            if (
                target.suffix == ".py"
                and target.is_file()
                and _is_within_target(target, target_dir)
            ):
                rel = target.relative_to(target_dir)
                return str(rel), tokens[1:]

        # Plain file with python shebang -> treat this file as the entry
        try:
            with candidate.open("r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline()
        except OSError:
            first_line = ""

        if "python" in first_line:
            rel = candidate.relative_to(target_dir)
            return str(rel), tokens[1:]

    # No explicit entry identified from the command
    return None, tokens