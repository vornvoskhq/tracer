import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class FunctionLocation:
    file_path: Path
    name: str
    start_line: int
    end_line: int


def find_enclosing_function(
    file_path: Path, line_number: int, function_name: Optional[str] = None
) -> Optional[FunctionLocation]:
    """
    Given a Python file and a line number, return the enclosing function or method.

    If function_name is provided, it will be used as a hint but line_number
    is still the primary selector.
    """
    try:
        source = file_path.read_text(encoding="utf-8")
    except OSError:
        return None

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError:
        return None

    best_match: Optional[Tuple[str, int, int]] = None

    class FuncVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef):  # type: ignore[override]
            self._maybe_update_best(node)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):  # type: ignore[override]
            self._maybe_update_best(node)
            self.generic_visit(node)

        def _maybe_update_best(self, node):
            nonlocal best_match
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None)

            if start is None or end is None:
                return

            if not (start <= line_number <= end):
                return

            # Prefer more specific (smaller) spans
            span = end - start
            if best_match is None or span < (best_match[2] - best_match[1]):
                if function_name and getattr(node, "name", None) != function_name:
                    # If function_name is given and doesn't match, still keep
                    # this as a candidate, but only if we don't have any match yet.
                    if best_match is None:
                        best_match = (node.name, start, end)
                else:
                    best_match = (node.name, start, end)

    FuncVisitor().visit(tree)

    if best_match is None:
        return None

    name, start, end = best_match
    return FunctionLocation(
        file_path=file_path, name=name, start_line=start, end_line=end
    )


def extract_source_segment(
    file_path: Path, start_line: int, end_line: int
) -> str:
    """Return the text for the given line span (inclusive)."""
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""

    # ast line numbers are 1-based and inclusive
    start_idx = max(start_line - 1, 0)
    end_idx = min(end_line, len(lines))
    segment = lines[start_idx:end_idx]
    return "\n".join(segment)