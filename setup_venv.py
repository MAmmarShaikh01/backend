#!/usr/bin/env python3
"""
Setup script to create and configure a Python virtual environment
to avoid running pip as root user.
"""

import os
import sys
import subprocess
import venv


def create_virtual_environment():
    """Create a virtual environment in the project directory."""
    venv_path = os.path.join(os.getcwd(), "venv")

    print("Creating virtual environment...")
    venv.create(venv_path, with_pip=True)
    print(f"Virtual environment created at: {venv_path}")

    # Upgrade pip in the virtual environment
    pip_path = os.path.join(venv_path, "Scripts", "pip.exe") if os.name == "nt" else os.path.join(venv_path, "bin", "pip")
    print("Upgrading pip in virtual environment...")
    subprocess.check_call([pip_path, "install", "--upgrade", "pip"])

    print("\nVirtual environment setup complete!")
    print("\nTo activate the virtual environment:")
    if os.name == "nt":  # Windows
        print("  Windows: venv\\Scripts\\activate")
    else:  # Unix/Linux/macOS
        print(f"  Linux/macOS: source venv/bin/activate")

    print("\nAfter activation, install dependencies with:")
    print("  pip install -r requirements.txt")

    print("\nTo deactivate the virtual environment:")
    print("  deactivate")

    return venv_path


def main():
    """Main function to create virtual environment."""
    print("Setting up Python virtual environment...")
    print(f"Current directory: {os.getcwd()}")

    try:
        venv_path = create_virtual_environment()
        print(f"\nSuccess! Virtual environment created at: {venv_path}")
        print("\nRemember to activate it before running pip commands!")
    except Exception as e:
        print(f"Error creating virtual environment: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()