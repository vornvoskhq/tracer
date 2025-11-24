#!/bin/bash
# activate venv and run tracer_app
source /home/vorn/mithra/tracer/.venv/bin/activate

# Pass the current working directory as the initial codebase folder so that
# invoking the global "trcr" symlink from any directory causes that directory
# to be opened in the UI as the codebase.
python /home/vorn/mithra/tracer/tracer_app.py "$(pwd)" "$@"