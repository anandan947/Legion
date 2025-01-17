#!/usr/bin/env python3
"""Script to set up development environment."""

import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def setup_environment() -> int:
    """Set up the development environment."""
    try:
        project_root = Path(__file__).parent.parent
        venv_path = project_root / "venv"

        print("Setting up development environment...")

        # Deactivate current virtual environment if active
        if "VIRTUAL_ENV" in os.environ:
            print("\n1. Deactivating current virtual environment...")
            # Clear the VIRTUAL_ENV variable
            del os.environ["VIRTUAL_ENV"]

        # Remove existing virtual environment if it exists
        if venv_path.exists():
            print("\n2. Removing existing virtual environment...")
            shutil.rmtree(venv_path)

        # Create virtual environment
        print("\n3. Creating virtual environment...")
        venv.create(venv_path, with_pip=True)

        # Determine the Python executable path
        if platform.system() == "Windows":
            python_path = venv_path / "Scripts" / "python.exe"
            activate_path = venv_path / "Scripts" / "activate"
        else:
            python_path = venv_path / "bin" / "python"
            activate_path = venv_path / "bin" / "activate"

        # Upgrade pip
        print("\n4. Upgrading pip...")
        run_command([str(python_path), "-m", "pip", "install", "--upgrade", "pip"])

        # Install dependencies
        print("\n5. Installing dependencies...")
        run_command([str(python_path), "-m", "pip", "install", "-r", str(project_root / "requirements.txt")])

        # Install Legion package in editable mode
        print("\n6. Installing Legion package in development mode...")
        run_command([str(python_path), "-m", "pip", "install", "-e", str(project_root)])

        # Install pre-commit
        print("\n7. Installing pre-commit...")
        run_command([str(python_path), "-m", "pip", "install", "pre-commit"])

        # Clear pre-commit cache
        print("\n8. Clearing pre-commit cache...")
        run_command([str(python_path), "-m", "pre_commit", "clean"])

        # Install pre-commit hooks
        print("\n9. Installing pre-commit hooks...")
        run_command([str(python_path), "-m", "pre_commit", "install"])

        print("\nEnvironment setup complete! 🎉")
        print("\nTo activate the virtual environment:")
        if platform.system() == "Windows":
            print(f"    {activate_path}")
        else:
            print(f"    source {activate_path}")

        return 0

    except subprocess.CalledProcessError as e:
        print(f"\nError during setup: {e}", file=sys.stderr)
        print(f"Command output: {e.output}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(setup_environment())
