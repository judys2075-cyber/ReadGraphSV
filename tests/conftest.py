"""Shared pytest helpers for ReadGraphSV."""

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_cli(script_name, *args, cwd=PROJECT_ROOT):
    """Run a ReadGraphSV CLI script and fail with captured output on error."""
    command = [sys.executable, str(PROJECT_ROOT / script_name), *map(str, args)]
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if result.returncode != 0:
        raise AssertionError(
            "Command failed:\n"
            f"{' '.join(command)}\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )
    return result
