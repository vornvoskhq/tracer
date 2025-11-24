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
                    # Condensed summary of auto-selected codebase
                    default = codebases[0]
                    others = ", ".join(d.name for d in codebases[1:])
                    if others:
                        print(f"📁 Codebase: target/{default.name} (others: {others})")
                    else:
                        print(f"📁 Codebase: target/{default.name}")
                    self.target_dir = default
                    self.codebase_name = default.name
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
        Create a tracer that properly executes any codebase's main logic.
        """
        # Get values that need to be inserted into the script
        target_dir_str = str(self.target_dir)
        codebase_name = self.codebase_name
        entry_point = self.entry_point
        
        script = f'''
import os
import sys
import json
import time
from collections import Counter

# Change to target directory
os.chdir("{target_dir_str}")
sys.path.insert(0, "{target_dir_str}")

print("🚀 {codebase_name.title()} Execution Tracer")
print(f"Python: {{sys.executable}}")
print(f"Entry Point: {entry_point}")
print(f"Args: {command_args}")
print("=" * 60)

# Global trace storage
TRACE_DATA = {{
    "start_time": time.time(),
    "calls": [],
    "functions": Counter(),
    "files": Counter(),
    "sequence": [],
    "patches_applied": [],
    "current_depth": 0,
    "file_accesses": [],
    "files_opened": Counter()
}}

def should_trace(filename, function_name):
    """Determine if we should trace this call."""
    if not filename:
        return False
    
    # Trace only codebase-specific files (not standard library)
    trace_patterns = [
        "{codebase_name.lower()}", "src/", "/src/"
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
            skip_functions = {{"__enter__", "__exit__", "__del__", "__getattribute__", "__setattr__"}}
            if function_name in skip_functions:
                return False
            return True
    
    # Also trace files in the target directory (absolute path check)
    if "{target_dir_str}" in filename:
        skip_functions = {{"__enter__", "__exit__", "__del__", "__getattribute__", "__setattr__"}}
        if function_name in skip_functions:
            return False
        return True
    
    return False

def trace_calls(frame, event, arg):
    """Main trace function."""
    global TRACE_DATA
    
    if event not in ('call', 'return'):
        return trace_calls
    
    filename = frame.f_code.co_filename
    function_name = frame.f_code.co_name
    line_no = frame.f_lineno

    # Decide if this frame should be traced
    trace_this = should_trace(filename, function_name)
    
    # Depth tracking only for traced frames
    if event == 'call':
        if not trace_this:
            return trace_calls
        TRACE_DATA["current_depth"] += 1
        # Limit depth
        if TRACE_DATA["current_depth"] > 50:
            TRACE_DATA["current_depth"] -= 1
            return trace_calls
    elif event == 'return':
        if trace_this and TRACE_DATA["current_depth"] > 0:
            TRACE_DATA["current_depth"] -= 1
        return trace_calls
    
    if not trace_this:
        return trace_calls
    
    try:
        # Get clean relative path (dynamic based on codebase)
        rel_path = filename
        if "{target_dir_str}" in filename:
            rel_path = filename.split("{target_dir_str}" + "/")[-1]
        elif filename.endswith("{codebase_name}.py"):
            rel_path = "{codebase_name}.py"
        elif filename.endswith("{entry_point}"):
            rel_path = "{entry_point}"
        elif "/src/" in filename:
            parts = filename.split("/src/")
            if len(parts) > 1:
                rel_path = "src/" + parts[-1]
        elif "/lib/" in filename:
            parts = filename.split("/lib/")
            if len(parts) > 1:
                rel_path = "lib/" + parts[-1]
        
        func_key = f"{{rel_path}}::{{function_name}}"
        
        # Update counters
        TRACE_DATA["functions"][func_key] += 1
        TRACE_DATA["files"][rel_path] += 1
        TRACE_DATA["sequence"].append(func_key)
        
        # Store call details with line number
        call_info = {{
            "function": function_name,
            "file": rel_path,
            "line": line_no,
            "timestamp": time.time() - TRACE_DATA["start_time"],
            "depth": TRACE_DATA["current_depth"]
        }}
        TRACE_DATA["calls"].append(call_info)
        
        # Also store in sequence with line number for easy access
        func_with_line = f"{{rel_path}}::{{function_name}}:{{line_no}}"
        TRACE_DATA["sequence_with_lines"] = TRACE_DATA.get("sequence_with_lines", [])
        TRACE_DATA["sequence_with_lines"].append(func_with_line)
        
    except Exception as e:
        pass
    
    return trace_calls

def save_trace_data(filename):
    """Save trace data to file."""
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
        
        print(f"💾 Saved {{len(TRACE_DATA['calls'])}} traced calls and {{len(TRACE_DATA['file_accesses'])}} file accesses")
        
    except Exception as e:
        print(f"❌ Error saving trace: {{e}}")

# Main execution
def main():
    try:
        # Set up sys.argv for VGMini
        sys.argv = ["vgmini.py"] + {command_args}
        print(f"🎯 Setting sys.argv: {{sys.argv}}")
        
        # Install tracer and file access hooks
        print("🔧 Installing tracer and file access monitoring...")
        sys.settrace(trace_calls)

        # Hook file operations to track file access
        original_open = open
        
        def tracked_open(file, mode='r', *args, **kwargs):
            """Wrapped open function to track file access."""
            try:
                # Get clean file path
                file_str = str(file)
                if "{target_dir_str}" in file_str:
                    rel_file = file_str.replace("{target_dir_str}/", "")
                else:
                    rel_file = file_str
                
                # Track the access
                access_type = "read" if 'r' in mode else "write" if 'w' in mode or 'a' in mode else "read"
                TRACE_DATA["file_accesses"].append({{
                    "file": rel_file,
                    "mode": mode,
                    "access_type": access_type,
                    "timestamp": time.time() - TRACE_DATA["start_time"]
                }})
                TRACE_DATA["files_opened"][rel_file] += 1
                
                print(f"📁 {{access_type.upper()}}: {{rel_file}} ({{mode}})")
                
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
                """Wrapped pickle.load to track file access."""
                try:
                    file_name = getattr(file, 'name', str(file))
                    if "{target_dir_str}" in file_name:
                        rel_file = file_name.replace("{target_dir_str}/", "")
                    else:
                        rel_file = file_name
                    
                    TRACE_DATA["file_accesses"].append({{
                        "file": rel_file,
                        "mode": "pickle_load",
                        "access_type": "read",
                        "timestamp": time.time() - TRACE_DATA["start_time"]
                    }})
                    TRACE_DATA["files_opened"][rel_file] += 1
                    print(f"🥒 PICKLE LOAD: {{rel_file}}")
                except Exception:
                    pass
                return original_pickle_load(file)
            
            def tracked_pickle_dump(obj, file):
                """Wrapped pickle.dump to track file access."""
                try:
                    file_name = getattr(file, 'name', str(file))
                    if "{target_dir_str}" in file_name:
                        rel_file = file_name.replace("{target_dir_str}/", "")
                    else:
                        rel_file = file_name
                    
                    TRACE_DATA["file_accesses"].append({{
                        "file": rel_file,
                        "mode": "pickle_dump",
                        "access_type": "write",
                        "timestamp": time.time() - TRACE_DATA["start_time"]
                    }})
                    TRACE_DATA["files_opened"][rel_file] += 1
                    print(f"🥒 PICKLE SAVE: {{rel_file}}")
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
            print(f"🔧 os.execv patched - preventing restart: {{args}}")
            TRACE_DATA["patches_applied"].append("os.execv")
            return
        os.execv = patched_execv
        
        # Execute entry point by running the file content with __name__ == "__main__"
        print(f"📚 Loading {entry_point} source...")
        with open("{entry_point}", "r") as f:
            source_code = f.read()
        
        print(f"🚀 Executing {entry_point} with __name__ == '__main__'...")
        print("=" * 60)
        
        # Create execution environment that simulates running the entry point directly
        exec_globals = {{
            "__name__": "__main__",
            "__file__": os.path.abspath("{entry_point}"),
            "__package__": None
        }}
        
        # Execute source code
        exec(source_code, exec_globals)
        
        print("=" * 60)
        print(f"✅ {codebase_name.title()} execution completed")
        
    except SystemExit as e:
        print(f"🏁 VGMini exited with code: {{e.code}}")
    except Exception as e:
        print(f"❌ {codebase_name.title()} error: {{e}}")
        TRACE_DATA["error"] = str(e)
        import traceback
        traceback.print_exc()
    finally:
        # Stop tracing and save data
        sys.settrace(None)
        save_trace_data("trace_output.json")

if __name__ == "__main__":
    main()
'''
        return script
    
    def trace_command(self, command: str) -> ExecutionTrace:
        """
        Trace VGMini command with proper main execution.
        """
        print(f"🔍 Tracing: {command}")
        # Compact startup summary: show codebase folder and entry point location
        # Emphasize that the target directory was auto-detected correctly.
        print(f"📁 target/{self.codebase_name} | entry: {self.entry_point}")
        
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
                error_output=error_output,
                success=success
            )
            
            # Attach full call list for detailed reporting
            execution_trace.calls = trace_data.get('calls', [])
            
            # Add line number sequence if available
            if 'sequence_with_lines' in trace_data:
                execution_trace.call_sequence_with_lines = trace_data['sequence_with_lines']
                
            # Add file access data if available
            if 'file_accesses' in trace_data:
                execution_trace.file_accesses = trace_data['file_accesses']
            if 'files_opened' in trace_data:
                execution_trace.files_opened = trace_data['files_opened']
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
                error_output=error_output,
                success=success
            )
        
        return execution_trace


def format_trace_report(trace: ExecutionTrace, detailed: bool = False) -> str:
    """Format trace report focused on execution order."""
    output = []
    
    # Brief summary
    output.append(f"📊 {trace.command} - {trace.duration:.1f}s - {trace.total_calls} calls, {trace.unique_functions} functions, {trace.unique_files} files")
    output.append("")
    
    # Main focus: Execution Order as a table (no truncation)
    raw_calls = getattr(trace, "calls", []) or []
    # Hide lambda frames in the report
    calls = [c for c in raw_calls if c.get("function") != "<lambda>"]
    if calls:
        output.append("📞 Function Execution Order:")
        # Header with short names to improve alignment
        output.append("Call\tLvl\tFile\tFunc\tLine")
        for idx, call in enumerate(calls, start=1):
            file_name = call.get("file", "")
            func_name = call.get("function", "")
            line_no = call.get("line", "")
            depth = call.get("depth", 0) or 0
            # Indent function name by depth for intuitive visual hierarchy
            indent = "" if depth <= 1 else "  " * (depth - 1)
            func_display = f"{indent}{func_name}"
            output.append(f"{idx}\t{depth}\t{file_name}\t{func_display}\t{line_no}")
        output.append("")
    
    # External file I/O events (text, config, pickle, images, etc.)
    file_accesses = getattr(trace, "file_accesses", []) or []
    # Filter out .venv and .py files (keep configs, text, pickle, images, etc.)
    filtered_accesses = []
    for a in file_accesses:
        file_path = a.get("file", "") or ""
        if ".venv/" in file_path:
            continue
        if file_path.endswith(".py"):
            continue
        filtered_accesses.append(a)
    
    if filtered_accesses:
        # Build a mapping from access -> calling function/file using timestamps
        caller_info = []
        call_index = 0
        n_calls = len(calls)
        for access in filtered_accesses:
            access_ts = access.get("timestamp", 0.0)
            # Advance call_index while the call timestamp is <= access timestamp
            while call_index + 1 &lt; n_calls and calls[call_index + 1].get("timestamp", 0.0) &lt;= access_ts:
                call_index += 1
            if n_calls &gt; 0:
                caller = calls[call_index]
                caller_info.append((
                    access,
                    caller.get("file", ""),
                    caller.get("function", ""),
                    caller.get("line", "")
                ))
            else:
                caller_info.append((access, "", "", ""))
        
        output.append("📁 External File I/O:")
        # No separate type column; show mode (r/w/a, pickle_load, etc.) and put File last
        output.append("Order\tMode\tSrcFile\tSrcFunc\tSrcLine\tFile")
        for idx, (access, c_file, c_func, c_line) in enumerate(caller_info, start=1):
            mode = access.get("mode", "")
            file_path = access.get("file", "")
            output.append(f"{idx}\t{mode}\t{c_file}\t{c_func}\t{c_line}\t{file_path}")
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