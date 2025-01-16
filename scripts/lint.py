#!/usr/bin/env python3
"""Script to run code style checks and auto-fix issues."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def run_ruff(fix: bool = False, unsafe_fixes: bool = False, directories: List[str] = None) -> int:
    """Run ruff code style checks and optionally fix issues.

    Args:
        fix: If True, attempt to automatically fix issues
        unsafe_fixes: If True, enable additional automated fixes
        directories: List of directories to check

    Returns:
        Exit code from ruff

    """
    project_root = Path(__file__).parent.parent

    # Base command with configuration
    cmd = [
        sys.executable,  # Use current Python interpreter
        "-m", "ruff", "check",
        "--no-cache",  # Disable cache to avoid any config caching issues
        "--line-length=100",
        "--target-version=py38",
        "--select=E,F,W,I,N,D,Q",
        # Ignore more rules that are too strict or conflict
        "--ignore=UP015,UP009,D100,D101,D102,D103,D107,D203,D212,D400,D415,D213",
        "--exclude=.git,__pycache__,.ruff_cache,build,dist",
    ] + directories

    # Add fix flags
    if fix:
        cmd.append("--fix")
        if unsafe_fixes:
            cmd.append("--unsafe-fixes")

    # Run ruff with explicit environment to avoid config file lookup
    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
        env={"NO_COLOR": "1"}  # Disable color output for cleaner logs
    )

    # Print output
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    error_count = len(result.stdout.strip().split("\n")) if result.stdout else 0
    action = "Found" if not fix else "Remaining"
    print(f"\n{action} {error_count} style issues")

    return result.returncode

def main() -> int:
    """Run the linting process."""
    parser = argparse.ArgumentParser(description="Run Ruff style checks and auto-fix issues.")
    parser.add_argument(
        "directories",
        metavar="DIR",
        type=str,
        nargs="*",
        help="Directories to run checks on. If none provided, uses default directories.",
        default=["legion", "tests", "examples"]
    )
    args = parser.parse_args()

    directories = args.directories

    if directories:
        print(f"Running checks on directories: {', '.join(directories)}")

    # First try safe fixes
    print("Attempting safe auto-fixes...")
    run_ruff(fix=True, unsafe_fixes=False, directories=directories)

    # Then try unsafe fixes
    print("\nAttempting additional auto-fixes...")
    run_ruff(fix=True, unsafe_fixes=True, directories=directories)

    # Final check
    print("\nRunning final style check...")
    return run_ruff(fix=False, directories=directories)

if __name__ == "__main__":
    sys.exit(main())
