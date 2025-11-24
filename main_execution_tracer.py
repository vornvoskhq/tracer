#!/usr/bin/env python3
"""
Main Execution Tracer

This tracer properly executes VGMini's main logic by simulating the
if __name__ == "__main__" condition to capture complete execution.
"""

import os
import sys
import json
import time
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Set, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from string import Template


@dataclass
class ExecutionTrace:
    """Contains execution trace data."""
    command: str
    start_time: float
    end_time: float
    duration: float
    total_calls: int
    unique_functions: int
    unique_files: int
    function_calls: List[Dict]
    file_usage: Dict[str, int]
    function_usage: Dict[str, int]
    call_sequence: List[str]
    patches_applied: List[str] = None
    venv_used: str = ""
    file_accesses: List[Dict] = None
    files_opened: Dict[str, int] = None
    error_output: str = ""
    success: bool = True
    # Full call records (including depth, file, function, line, timestamp)
    calls: Optional[List[Dict]] = None


class MainExecutionTracer:
    """
    Tracer that works with any Python codebase in target/ directory.
    """
    
    def __init__(self, target_dir: str = None, auto_detect: bool = True):
        if target_dir is None and auto_detect:
            # Auto-detect available codebases in target/
            target_base = Path("target").resolve()
            if target_base.exists():
                codebases = [d for d in target_base.iterdir() if d.is_dir() and not d.name.startswith('.')]
                if len(codebases) == 1:
                    self.target_dir = codebases[0]
                    self.codebase_name = codebases[0].name
                elif len(codebases) > 1:
                    print(f"🔍 Multiple codebases found: {[d.name for d in codebases]}")
                    print(f"📁 Using default: {codebases[0].name}")
                    self.target_dir = codebases[0]
                    self.codebase_name = codebases[0].name
                else:
                    raise FileNotFoundError("No codebases found in target/ directory")
            else:
                raise FileNotFoundError("target/ directory not found")
        else:
            # Use specified directory
            if target_dir is None:
                target_dir = "target/vgmini"  # fallback
            self.target_dir = Path(target_dir).resolve()
            self.codebase_name = self.target_dir.name
        
        if not self.target_dir.exists():
            raise FileNotFoundError(f"Codebase directory not found: {self.target_dir}")
        
        # Detect entry point
        self.entry_point = self.detect_entry_point()
        
        # Detect virtual environment
        self.target_venv = self.target_dir / ".venv"
        self.target_python = None
        
        if self.target_venv.exists():
            possible_pythons = [
                self.target_venv / "bin" / "python",
                self.target_venv / "bin" / "python3", 
                self.target_venv / "Scripts" / "python.exe",
                self.target_venv / "Scripts" / "python3.exe"
            ]
            
            for python_path in possible_pythons:
                if python_path.exists():
                    self.target_python = str(python_path)
                    break
        
        if not self.target_python:
            self.target_python = sys.executable
    
    def detect_entry_point(self) -> str:
        """
        Auto-detect the main entry point for the codebase.
        """
        # Common entry point patterns
        possible_entry_points = [
            # Direct Python files
            self.target_dir / f"{self.codebase_name}.py",
            self.target_dir / "main.py",
            self.target_dir / "app.py",
            self.target_dir / "run.py",
            self.target_dir / "__main__.py",
            # Executable scripts
            self.target_dir / f"{self.codebase_name}",
            self.target_dir / "run",
            self.target_dir / "start",
        ]
        
        for entry_point in possible_entry_points:
            if entry_point.exists():
                return entry_point.name
        
        # If no obvious entry point, look for the largest .py file in root
        py_files = list(self.target_dir.glob("*.py"))
        if py_files:
            largest_file = max(py_files, key=lambda f: f.stat().st_size)
            print(f"⚠️  No obvious entry point found, using largest Python file: {largest_file.name}")
            return largest_file.name
        
        raise FileNotFoundError(f"No entry point found for {self.codebase_name}")
    
    def create_main_execution_tracer(self, command_args: List[str]) -> str:
        """
        Create a tracer script that properly executes any codebase's main logic.

        This version uses string.Template for substitution so that we can use
        normal Python dictionaries and formatting inside the generated script
        without fighting f-string brace escaping.
        """
        target_dir_str = str(self.target_dir)
        codebase_name = self.codebase_name
        entry_point = self.entry_point

        # Escape backslashes for safe use inside quoted strings in the script
        target_dir_escaped = target_dir_str.replace("\\", "\\\\")

        # Represent command_args as a Python list literal
        command_args_repr = repr(command_args)

        template = Template(
            """
import os
import sys
import json
import time
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

    # Trace only codebase-specific files (not standard library)
    trace_patterns = [
        "$CODEBASE_NAME_LOWER", "src/", "/src/"
    ]

    # Must be a Python file
    if not filename.endswith(".py"):
        return False

    # Skip standard library and external packages
    skip_patterns = [
        "__pycache__", "site-packages", "/lib/python", "/logging/", "/warnings/",
        "/.venv/", "/venv/", "/env/", "/Lib/", "/Scripts/"
    ]
    for skip in skip_patterns:
        if skip in filename:
            return False

    # Only trace files that match our codebase patterns
    for pattern in trace_patterns:
        if pattern in filename.lower():
            # Skip some very noisy functions
            skip_functions = {"__enter__", "__exit__", "__del__", "__getattribute__", "__setattr__"}
            if function_name in skip_functions:
                return False
            return True

    # Also trace files in the target directory (absolute path check)
    if "$TARGET_DIR" in filename:
        skip_functions = {"__enter__", "__exit__", "__del__", "__getattribute__", "__setattr__"}
        if function_name in skip_functions:
            return False
        return True

    return False

def trace_calls(frame, event, arg):
    \"\"\"Main trace function.\"\"\"
    global TRACE_DATA

    if event not in ('call', 'return'):
        return trace_calls

    filename = frame.f_code.co_filename
    function_name = frame.f_code.co_name
    line_no = frame.f_lineno

    # Depth tracking
    if event == 'call':
        TRACE_DATA["current_depth"] += 1
    elif event == 'return':
        TRACE_DATA["current_depth"] -= 1
        return trace_calls

    # Limit depth
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
        TRACE_DATA["sequence_with_lines"] = TRACE_DATA.get("sequence_with_lines", [])
        TRACE_DATA["sequence_with_lines"].append(func_with_line)

    except Exception:
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

        print(f"💾 Saved {len(TRACE_DATA['calls'])} traced calls and {len(TRACE_DATA['file_accesses'])} file accesses")

    except Exception as e:
        print(f"❌ Error saving trace: {e}")

# Main execution
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

        def tracked_open(file, mode='r', *args, **kwargs):
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
                access_type = "read" if 'r' in mode else "write" if 'w' in mode or 'a' in mode else "read"
                TRACE_DATA["file_accesses"].append({
                    "file": rel_file,
                    "mode": mode,
                    "access_type": access_type,
                    "timestamp": time.time() - TRACE_DATA["start_time"],
                    "src_file": src_file,
                    "src_func": src_func,
                    "src_line": src_line,
                })
                TRACE_DATA["files_opened"][rel_file] += 1

                print("📁 FILE ACCESS: {} ({}) from {}::{}:{}".format(rel_file, mode, src_file, src_func, src_line))

            except Exception:
                pass  # Don't break file operations if tracking fails

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
                    file_name = getattr(file, 'name', str(file))
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

                    TRACE_DATA["file_accesses"].append({
                        "file": rel_file,
                        "mode": "pickle_load",
                        "access_type": "read",
                        "timestamp": time.time() - TRACE_DATA["start_time"],
                        "src_file": src_file,
                        "src_func": src_func,
                        "src_line": src_line,
                    })
                    TRACE_DATA["files_opened"][rel_file] += 1
                    print("🥒 PICKLE LOAD: {} from {}::{}:{}".format(rel_file, src_file, src_func, src_line))
                except Exception:
                    pass
                return original_pickle_load(file)

            def tracked_pickle_dump(obj, file):
                \"\"\"Wrapped pickle.dump to track file access.\"\"\"
                try:
                    file_name = getattr(file, 'name', str(file))
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

                    TRACE_DATA["file_accesses"].append({
                        "file": rel_file,
                        "mode": "pickle_dump",
                        "access_type": "write",
                        "timestamp": time.time() - TRACE_DATA["start_time"],
                        "src_file": src_file,
                        "src_func": src_func,
                        "src_line": src_line,
                    })
                    TRACE_DATA["files_opened"][rel_file] += 1
                    print("🥒 PICKLE SAVE: {} from {}::{}:{}".format(rel_file, src_file, src_func, src_line))
                except Exception:
                    pass
                return original_pickle_dump(obj, file)

            pickle.load = tracked_pickle_load
            pickle.dump = tracked_pickle_dump

        except ImportError:
            pass  # pickle not available

        # Apply patches to prevent restart
        print("🐒 Applying patches...")

        # Patch os.execv to prevent process replacement
        import os
        original_execv = getattr(os, 'execv', None)
        def patched_execv(*args, **kwargs):
            print(f"🔧 os.execv patched - preventing restart: {args}")
            TRACE_DATA["patches_applied"].append("os.execv")
            return
        os.execv = patched_execv

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

        script = template.substitute(
            TARGET_DIR=target_dir_escaped,
            CODEBASE_TITLE=codebase_name.title(),
            ENTRY_POINT=entry_point,
            CODEBASE_NAME=codebase_name,
            CODEBASE_NAME_LOWER=codebase_name.lower(),
            COMMAND_ARGS=command_args_repr,
        )
        return script
    
    def trace_command(self, command: str) -> ExecutionTrace:
        """
        Trace VGMini command with proper main execution.
        """
        print(f"🔍 Tracing: {command}")
        print(f"📁 Codebase: {self.codebase_name} ({self.target_dir})")
        print(f"🎯 Entry Point: {self.entry_point}")
        print(f"🐍 Using Python: {self.target_python}")
        
        # Parse command
        cmd_parts = command.strip().split()
        if cmd_parts[0].startswith('./'):
            vg_args = cmd_parts[1:] if len(cmd_parts) > 1 else []
        else:
            vg_args = cmd_parts
        
        # Create tracer script
        tracer_script = self.create_main_execution_tracer(vg_args)
        
        # Write script to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(tracer_script)
            script_path = f.name
        
        trace_output_file = self.target_dir / "trace_output.json"
        
        start_time = time.time()
        success = True
        error_output = ""
        
        try:
            print(f"🚀 Starting {self.codebase_name} with main execution...")
            
            # Run the tracer script using target's Python
            process = subprocess.Popen(
                [self.target_python, script_path],
                cwd=str(self.target_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Progress indicator
            dots = 0
            start_wait = time.time()
            while process.poll() is None:
                elapsed = time.time() - start_wait
                dots = (dots + 1) % 4
                dot_str = "." * dots + " " * (3 - dots)
                print(f"\r⏳ Executing{dot_str} ({elapsed:.1f}s)", end="", flush=True)
                time.sleep(0.5)
                
                if elapsed > 300:  # 5 minute timeout
                    process.terminate()
                    break
            
            stdout, stderr = process.communicate()
            elapsed = time.time() - start_wait
            print(f"\r✅ Completed ({elapsed:.1f}s)     ")
            
            # Only show output if there are errors
            if process.returncode != 0 and stdout.strip():
                print(f"📋 {self.codebase_name.title()} Output (error):")
                lines = stdout.strip().split('\n')
                for line in lines[-10:]:  # Show last 10 lines on error
                    print(f"  {line}")
            
            if process.returncode != 0:
                success = False
                error_output = f"STDOUT:\\n{stdout}\\nSTDERR:\\n{stderr}"
                if stderr.strip():
                    print(f"⚠️  Error output: {stderr.strip()[:300]}...")
                    
        except Exception as e:
            success = False
            error_output = str(e)
            print(f"\r❌ Error: {e}")
        
        end_time = time.time()
        
        # Read trace data
        trace_output_file = self.target_dir / "trace_output.json"
        trace_data = None
        if trace_output_file.exists():
            try:
                with open(trace_output_file, 'r') as f:
                    trace_data = json.load(f)
                # Minimal output - just essentials
                
                # Clean up trace file
                trace_output_file.unlink()
            except Exception as e:
                print(f"❌ Failed to read trace: {e}")
        
        # Clean up script file
        try:
            os.unlink(script_path)
        except:
            pass
        
        # Build result
        if trace_data:
            function_calls = []
            for func, count in trace_data.get('functions', {}).items():
                if '::' in func:
                    file_name, func_name = func.split('::', 1)
                    function_calls.append({
                        'function': func_name,
                        'file': file_name,
                        'count': count
                    })
            
            execution_trace = ExecutionTrace(
                command=command,
                start_time=trace_data.get('start_time', start_time),
                end_time=trace_data.get('end_time', end_time),
                duration=trace_data.get('duration', end_time - start_time),
                total_calls=len(trace_data.get('calls', [])),
                unique_functions=len(trace_data.get('functions', {})),
                unique_files=len(trace_data.get('files', {})),
                function_calls=function_calls,
                file_usage=trace_data.get('files', {}),
                function_usage=trace_data.get('functions', {}),
                call_sequence=trace_data.get('sequence', []),
                patches_applied=trace_data.get('patches_applied', []),
                venv_used=self.target_python,
                file_accesses=trace_data.get('file_accesses', []),
                files_opened=trace_data.get('files_opened', {}),
                error_output=error_output,
                success=success,
                calls=trace_data.get('calls', []),
            )
            
            # Add line number sequence if available
            if 'sequence_with_lines' in trace_data:
                execution_trace.call_sequence_with_lines = trace_data['sequence_with_lines']
        else:
            execution_trace = ExecutionTrace(
                command=command,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                total_calls=0,
                unique_functions=0,
                unique_files=0,
                function_calls=[],
                file_usage={},
                function_usage={},
                call_sequence=[],
                patches_applied=[],
                venv_used=self.target_python,
                file_accesses=[],
                files_opened={},
                error_output=error_output,
                success=success,
                calls=[],
            )
        
        return execution_trace


def format_trace_report(trace: ExecutionTrace, detailed: bool = False) -> str:
    """Format trace report focused on execution order."""
    output = []
    
    # Brief summary
    output.append(f"📊 {trace.command} - {trace.duration:.1f}s - {trace.total_calls} calls, {trace.unique_functions} functions, {trace.unique_files} files")
    output.append("")
    
    # Main focus: Execution Order
    if trace.call_sequence:
        if detailed:
            # Show detailed call sequence (all calls, not just unique)
            output.append(f"📞 Complete Call Sequence (showing all {len(trace.call_sequence)} calls):")
            for i, call in enumerate(trace.call_sequence):
                output.append(f"  {i+1:4d}. {call}")
        else:
            # Show execution order with line numbers
            output.append("📞 Function Execution Order:")
            seen_functions = set()
            execution_order = []
            
            # Use call sequence with line numbers if available, otherwise fall back to regular sequence
            sequence_to_use = getattr(trace, 'call_sequence_with_lines', trace.call_sequence)
            
            for call in sequence_to_use:
                # Extract function name without line number for uniqueness check
                if "::" in call and ":" in call.split("::")[-1]:
                    # Format: file::function:line -> extract file::function
                    func_without_line = "::".join(call.split("::")[:-1]) + "::" + call.split("::")[-1].split(":")[0]
                else:
                    func_without_line = call
                
                if func_without_line not in seen_functions:
                    seen_functions.add(func_without_line)
                    execution_order.append(call)
                    if len(execution_order) >= 100:  # Show first 100 unique functions
                        break
            
            for i, call in enumerate(execution_order):
                # Format the display: extract line number if present
                if "::" in call and ":" in call.split("::")[-1]:
                    parts = call.split("::")
                    if len(parts) == 2:
                        file_part = parts[0]
                        func_line_part = parts[1]
                        if ":" in func_line_part:
                            func_name, line_num = func_line_part.split(":", 1)
                            output.append(f"  {i+1:3d}. {file_part}::{func_name} (line {line_num})")
                        else:
                            output.append(f"  {i+1:3d}. {call}")
                    else:
                        output.append(f"  {i+1:3d}. {call}")
                else:
                    output.append(f"  {i+1:3d}. {call}")
            
            remaining_unique = len(set([call.split(":")[0] if ":" in call.split("::")[-1] else call for call in sequence_to_use])) - len(execution_order)
            if remaining_unique > 0:
                output.append(f"  ... and {remaining_unique} more unique functions")
        output.append("")
    
    # File access summary
    if trace.files_opened:
        output.append("📁 Files Accessed:")
        sorted_files = sorted(trace.files_opened.items(), key=lambda x: x[1], reverse=True)
        for file_path, count in sorted_files:
            output.append(f"  {count:2d}x - {file_path}")
        output.append("")
    
    # Compact Python file summary (only if not detailed)
    if not detailed and trace.file_usage:
        output.append("🐍 Python Files Used:")
        sorted_files = sorted(trace.file_usage.items(), key=lambda x: x[1], reverse=True)
        for file_path, count in sorted_files:
            output.append(f"  {count:4d} - {file_path}")
        output.append("")
    
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description="Universal Python codebase execution tracer")
    parser.add_argument("command", nargs="?", help="Command to trace (e.g., './script args' or 'python main.py')")
    parser.add_argument("--detailed", action="store_true", help="Show detailed call sequence")
    parser.add_argument("--target", help="Target directory (default: auto-detect in target/)")
    parser.add_argument("--list-codebases", action="store_true", help="List available codebases in target/")
    
    args = parser.parse_args()
    
    if args.list_codebases:
        target_base = Path("target")
        if target_base.exists():
            codebases = [d.name for d in target_base.iterdir() if d.is_dir() and not d.name.startswith('.')]
            print("📁 Available codebases:")
            for codebase in codebases:
                entry_point_info = ""
                try:
                    tracer = MainExecutionTracer(f"target/{codebase}")
                    entry_point_info = f" (entry: {tracer.entry_point})"
                except:
                    pass
                print(f"  • {codebase}{entry_point_info}")
        else:
            print("❌ target/ directory not found")
        return
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        tracer = MainExecutionTracer(target_dir=args.target, auto_detect=True)
        trace = tracer.trace_command(args.command)
        
        report = format_trace_report(trace, detailed=args.detailed)
        print(report)
        
    except Exception as e:
        print(f"❌ Tracing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()