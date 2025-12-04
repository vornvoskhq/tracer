import os
from string import Template
from typing import List


TRACER_SCRIPT_TEMPLATE = Template(
    """
import os
import sys
import json
import time
import runpy
from collections import Counter

# Change to target directory
os.chdir("$TARGET_DIR")
sys.path.insert(0, "$TARGET_DIR")

print("🚀 $CODEBASE_TITLE Execution Tracer")
print(f"Python: {sys.executable}")
print("Entry Point: $ENTRY_POINT")
print(f"Args: $COMMAND_ARGS")
print("=" * 60)

# Global trace storage
TRACE_DATA = {
    "start_time": time.time(),
    "calls": [],
    "functions": Counter(),
    "files": Counter(),
    "sequence": [],
    "patches_applied": [],
    "current_depth": 0,
    "file_accesses": [],
    "files_opened": Counter(),
}

def should_trace(filename, function_name):
    \"\"\"Determine if we should trace this call.\"\"\"
    if not filename:
        return False

    # Always trace the entrypoint script itself
    try:
        entry_abs = os.path.abspath("$ENTRY_POINT")
    except Exception:
        entry_abs = "$ENTRY_POINT"
    if (
        filename == entry_abs
        or filename.endswith("/$ENTRY_POINT")
        or filename.endswith("\\\\$ENTRY_POINT")
    ):
        return True

    # Must be a Python file
    if not filename.endswith(".py"):
        return False

    # Trace only codebase-specific files (not standard library)
    trace_patterns = [
        "$CODEBASE_NAME_LOWER",
        "src/",
        "/src/",
    ]

    # Skip standard library and external packages
    skip_patterns = [
        "__pycache__",
        "site-packages",
        "/lib/python",
        "/logging/",
        "/warnings/",
        "/.venv/",
        "/venv/",
        "/env/",
        "/Lib/",
        "/Scripts/",
    ]
    for skip in skip_patterns:
        if skip in filename:
            return False

    # MUCH MORE PERMISSIVE: Trace any Python file that looks like our codebase
    include_patterns = [
        "$CODEBASE_NAME_LOWER",  # project name
        "/src/",
        "src/",
        "$TARGET_DIR",
    ]

    for pattern in include_patterns:
        if pattern in filename:
            skip_functions = {
                "__enter__",
                "__exit__",
                "__del__",
                "__getattribute__",
                "__setattr__",
            }
            if function_name in skip_functions:
                return False
            return True

    # ENHANCED: Also trace any .py files in the current working directory
    try:
        rel_filename = os.path.relpath(filename)
        if (
            rel_filename.endswith(".py")
            and not rel_filename.startswith("..")
            and "/.venv/" not in rel_filename
        ):
            skip_functions = {
                "__enter__",
                "__exit__",
                "__del__",
                "__getattribute__",
                "__setattr__",
            }
            if function_name in skip_functions:
                return False
            return True
    except (ValueError, OSError):
        # os.path.relpath can fail in some cases, ignore
        pass

    return False

def trace_calls(frame, event, arg):
    \"\"\"Main trace function.\"\"\"
    global TRACE_DATA

    if event not in ("call", "return"):
        return trace_calls

    filename = frame.f_code.co_filename
    function_name = frame.f_code.co_name
    line_no = frame.f_lineno

    # Depth tracking
    if event == "call":
        TRACE_DATA["current_depth"] += 1
    elif event == "return":
        TRACE_DATA["current_depth"] -= 1
        if TRACE_DATA["current_depth"] < 0:
            TRACE_DATA["current_depth"] = 0
        return trace_calls

    # Limit depth to avoid excessive recursion noise
    if TRACE_DATA["current_depth"] > 50:
        TRACE_DATA["current_depth"] -= 1
        return trace_calls

    # Only trace relevant files
    if not should_trace(filename, function_name):
        return trace_calls

    try:
        # Get clean relative path (dynamic based on codebase)
        rel_path = filename
        if "$TARGET_DIR" in filename:
            rel_path = filename.split("$TARGET_DIR" + "/")[-1]
        elif filename.endswith("$CODEBASE_NAME.py"):
            rel_path = "$CODEBASE_NAME.py"
        elif filename.endswith("$ENTRY_POINT"):
            rel_path = "$ENTRY_POINT"
        elif "/src/" in filename:
            parts = filename.split("/src/")
            if len(parts) > 1:
                rel_path = "src/" + parts[-1]
        elif "/lib/" in filename:
            parts = filename.split("/lib/")
            if len(parts) > 1:
                rel_path = "lib/" + parts[-1]

        func_key = f"{rel_path}::{function_name}"

        # Update counters
        TRACE_DATA["functions"][func_key] += 1
        TRACE_DATA["files"][rel_path] += 1
        TRACE_DATA["sequence"].append(func_key)

        # Store call details with line number
        call_info = {
            "function": function_name,
            "file": rel_path,
            "line": line_no,
            "timestamp": time.time() - TRACE_DATA["start_time"],
            "depth": TRACE_DATA["current_depth"],
        }
        TRACE_DATA["calls"].append(call_info)

        # Also store in sequence with line number for easy access
        func_with_line = f"{rel_path}::{function_name}:{line_no}"
        sequence_with_lines = TRACE_DATA.get("sequence_with_lines")
        if sequence_with_lines is None:
            sequence_with_lines = []
            TRACE_DATA["sequence_with_lines"] = sequence_with_lines
        sequence_with_lines.append(func_with_line)

    except Exception:
        # Never break the target program because of tracing issues
        pass

    return trace_calls

def save_trace_data(filename):
    \"\"\"Save trace data to file.\"\"\"
    try:
        TRACE_DATA["end_time"] = time.time()
        TRACE_DATA["duration"] = TRACE_DATA["end_time"] - TRACE_DATA["start_time"]

        # Convert Counter objects for JSON serialization
        output_data = dict(TRACE_DATA)
        output_data["functions"] = dict(TRACE_DATA["functions"])
        output_data["files"] = dict(TRACE_DATA["files"])
        output_data["files_opened"] = dict(TRACE_DATA["files_opened"])

        with open(filename, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        print(
            f"💾 Saved {len(TRACE_DATA['calls'])} traced calls and "
            f"{len(TRACE_DATA['file_accesses'])} file accesses"
        )

    except Exception as e:
        print(f"❌ Error saving trace: {e}")

def main():
    try:
        # Set up sys.argv for target app
        sys.argv = ["$ENTRY_POINT"] + $COMMAND_ARGS
        print(f"🎯 Setting sys.argv: {sys.argv}")

        # Install tracer and file access hooks
        print("🔧 Installing tracer and file access monitoring...")
        sys.settrace(trace_calls)

        # Hook file operations to track file access
        original_open = open

        def tracked_open(file, mode="r", *args, **kwargs):
            \"\"\"Wrapped open function to track file access.\"\"\"
            try:
                # Get clean file path
                file_str = str(file)
                if "$TARGET_DIR" in file_str:
                    rel_file = file_str.replace("$TARGET_DIR" + "/", "")
                else:
                    rel_file = file_str

                # Identify caller (file, function, line) for better attribution
                src_file = ""
                src_func = ""
                src_line = 0
                try:
                    frame = sys._getframe(1)
                    src_file = frame.f_code.co_filename or ""
                    src_func = frame.f_code.co_name or ""
                    src_line = frame.f_lineno or 0
                    if "$TARGET_DIR" in src_file:
                        src_file = src_file.replace("$TARGET_DIR" + "/", "")
                except Exception:
                    pass

                # Track the access
                if "w" in mode or "a" in mode or "+" in mode:
                    access_type = "write"
                else:
                    access_type = "read"

                TRACE_DATA["file_accesses"].append(
                    {
                        "file": rel_file,
                        "mode": mode,
                        "access_type": access_type,
                        "timestamp": time.time() - TRACE_DATA["start_time"],
                        "src_file": src_file,
                        "src_func": src_func,
                        "src_line": src_line,
                    }
                )
                TRACE_DATA["files_opened"][rel_file] += 1

                print(
                    "📁 FILE ACCESS: {} ({}) from {}::{}:{}".format(
                        rel_file, mode, src_file, src_func, src_line
                    )
                )

            except Exception:
                # Don't break file operations if tracking fails
                pass

            # Call original open
            return original_open(file, mode, *args, **kwargs)

        # Replace built-in open
        import builtins

        builtins.open = tracked_open

        # Also hook pickle operations
        try:
            import pickle

            original_pickle_load = pickle.load
            original_pickle_dump = pickle.dump

            def tracked_pickle_load(file):
                \"\"\"Wrapped pickle.load to track file access.\"\"\"
                try:
                    file_name = getattr(file, "name", str(file))
                    if "$TARGET_DIR" in file_name:
                        rel_file = file_name.replace("$TARGET_DIR" + "/", "")
                    else:
                        rel_file = file_name

                    # Caller info
                    src_file = ""
                    src_func = ""
                    src_line = 0
                    try:
                        frame = sys._getframe(1)
                        src_file = frame.f_code.co_filename or ""
                        src_func = frame.f_code.co_name or ""
                        src_line = frame.f_lineno or 0
                        if "$TARGET_DIR" in src_file:
                            src_file = src_file.replace("$TARGET_DIR" + "/", "")
                    except Exception:
                        pass

                    TRACE_DATA["file_accesses"].append(
                        {
                            "file": rel_file,
                            "mode": "pickle_load",
                            "access_type": "read",
                            "timestamp": time.time() - TRACE_DATA["start_time"],
                            "src_file": src_file,
                            "src_func": src_func,
                            "src_line": src_line,
                        }
                    )
                    TRACE_DATA["files_opened"][rel_file] += 1
                    print(
                        "🥒 PICKLE LOAD: {} from {}::{}:{}".format(
                            rel_file, src_file, src_func, src_line
                        )
                    )
                except Exception:
                    pass
                return original_pickle_load(file)

            def tracked_pickle_dump(obj, file):
                \"\"\"Wrapped pickle.dump to track file access.\"\"\"
                try:
                    file_name = getattr(file, "name", str(file))
                    if "$TARGET_DIR" in file_name:
                        rel_file = file_name.replace("$TARGET_DIR" + "/", "")
                    else:
                        rel_file = file_name

                    # Caller info
                    src_file = ""
                    src_func = ""
                    src_line = 0
                    try:
                        frame = sys._getframe(1)
                        src_file = frame.f_code.co_filename or ""
                        src_func = frame.f_code.co_name or ""
                        src_line = frame.f_lineno or 0
                        if "$TARGET_DIR" in src_file:
                            src_file = src_file.replace("$TARGET_DIR" + "/", "")
                    except Exception:
                        pass

                    TRACE_DATA["file_accesses"].append(
                        {
                            "file": rel_file,
                            "mode": "pickle_dump",
                            "access_type": "write",
                            "timestamp": time.time() - TRACE_DATA["start_time"],
                            "src_file": src_file,
                            "src_func": src_func,
                            "src_line": src_line,
                        }
                    )
                    TRACE_DATA["files_opened"][rel_file] += 1
                    print(
                        "🥒 PICKLE SAVE: {} from {}::{}:{}".format(
                            rel_file, src_file, src_func, src_line
                        )
                    )
                except Exception:
                    pass
                return original_pickle_dump(obj, file)

            pickle.load = tracked_pickle_load
            pickle.dump = tracked_pickle_dump

        except ImportError:
            # pickle not available
            pass

        # Apply patches to prevent restart
        print("🐒 Applying patches...")

        # Patch os.execv to prevent process replacement
        import os as _os

        original_execv = getattr(_os, "execv", None)

        def patched_execv(*args, **kwargs):
            \"\"\"Enhanced patch that detects and simulates venv restart.\"\"\"
            print(f"🔧 os.execv intercepted: {args}")
            
            # Check if this looks like a venv restart
            if len(args) >= 2:
                python_path = args[0]
                argv_list = args[1]
                
                # If it's restarting with venv python, we need to handle this properly
                if '.venv' in python_path and len(argv_list) > 1:
                    print("🔄 Detected venv restart - simulating successful restart")
                    sys.executable = python_path
                    TRACE_DATA["patches_applied"].append("os.execv (venv restart simulated)")
                    return
            
            print(f"🔧 os.execv patched - preventing restart: {args}")
            TRACE_DATA["patches_applied"].append("os.execv")
            return

        if original_execv is not None:
            _os.execv = patched_execv

        # Execute entry point by running it as a normal script using runpy.
        # This preserves normal Python __main__ behavior and ensures that the
        # entry script appears as a proper module in traces.
        print(f"🚀 Executing $ENTRY_POINT as __main__ via runpy.run_path...")
        print("=" * 60)

        entry_abspath = os.path.abspath("$ENTRY_POINT")
        runpy.run_path(entry_abspath, run_name="__main__")

        print("=" * 60)
        print("✅ $CODEBASE_TITLE execution completed")

    except SystemExit as e:
        print(f"🏁 Target exited with code: {e.code}")
    except Exception as e:
        print(f"❌ $CODEBASE_TITLE error: {e}")
        TRACE_DATA["error"] = str(e)
        import traceback

        traceback.print_exc()
    finally:
        # Stop tracing and save data
        sys.settrace(None)
        save_trace_data("trace_output.json")

if __name__ == "__main__":
    main()
"""
)


def build_tracer_script(
    target_dir: str,
    codebase_name: str,
    entry_point: str,
    command_args: List[str],
) -> str:
    """Build the tracer script for a given target codebase."""
    target_dir_escaped = target_dir.replace("\\", "\\\\")
    command_args_repr = repr(command_args)

    return TRACER_SCRIPT_TEMPLATE.substitute(
        TARGET_DIR=target_dir_escaped,
        CODEBASE_TITLE=codebase_name.title(),
        ENTRY_POINT=entry_point,
        CODEBASE_NAME=codebase_name,
        CODEBASE_NAME_LOWER=codebase_name.lower(),
        COMMAND_ARGS=command_args_repr,
    )