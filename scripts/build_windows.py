#!/usr/bin/env python3
"""
Build script for HARO Windows executable.

Creates a folder-based distribution using PyInstaller.
Output: dist/HARO/ containing haro.exe and all dependencies.

Usage:
    python scripts/build_windows.py
    python scripts/build_windows.py --clean
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def check_python_version() -> bool:
    """Check that Python 3.11+ is being used."""
    if sys.version_info < (3, 11):
        print(f"Error: Python 3.11+ required, found {sys.version}")
        return False
    print(f"Python version: {sys.version}")
    return True


def check_pyinstaller() -> bool:
    """Check that PyInstaller is installed."""
    try:
        import PyInstaller
        print(f"PyInstaller version: {PyInstaller.__version__}")
        return True
    except ImportError:
        print("Error: PyInstaller not installed.")
        print("Install with: pip install pyinstaller")
        print("Or install dev dependencies: pip install -e '.[dev]'")
        return False


def clean_build_dirs(project_root: Path) -> None:
    """Remove previous build artifacts."""
    dirs_to_clean = ['build', 'dist']
    for dir_name in dirs_to_clean:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"Removing {dir_path}")
            shutil.rmtree(dir_path)


def run_pyinstaller(project_root: Path) -> bool:
    """Run PyInstaller with the spec file."""
    spec_file = project_root / 'haro.spec'

    if not spec_file.exists():
        print(f"Error: Spec file not found: {spec_file}")
        return False

    print(f"Building with spec file: {spec_file}")
    print("-" * 60)

    result = subprocess.run(
        [sys.executable, '-m', 'PyInstaller', str(spec_file), '--noconfirm'],
        cwd=project_root,
    )

    return result.returncode == 0


def post_build_setup(project_root: Path) -> None:
    """Set up the dist folder after build."""
    dist_dir = project_root / 'dist' / 'HARO'
    internal_dir = dist_dir / '_internal'

    if not dist_dir.exists():
        print("Warning: dist/HARO not found, skipping post-build setup")
        return

    # Create empty models directory
    models_dir = dist_dir / 'models'
    models_dir.mkdir(exist_ok=True)
    print(f"Created: {models_dir}")

    # Copy user-facing files from _internal to dist root for easier access
    # (The app will find them in either location)
    if internal_dir.exists():
        # Copy .env.example
        src_env = internal_dir / '.env.example'
        if src_env.exists():
            shutil.copy(src_env, dist_dir / '.env.example')
            print(f"Copied: .env.example")

        # Copy config directory
        src_config = internal_dir / 'config'
        dst_config = dist_dir / 'config'
        if src_config.exists() and not dst_config.exists():
            shutil.copytree(src_config, dst_config)
            print(f"Copied: config/")

        # Copy .context template
        src_context = internal_dir / '.context'
        dst_context = dist_dir / '.context'
        if src_context.exists() and not dst_context.exists():
            shutil.copytree(src_context, dst_context)
            print(f"Copied: .context/")

    # Write README
    readme_path = dist_dir / 'README-WINDOWS.txt'
    readme_content = """\
HARO Voice Assistant - Windows Build
=====================================

Quick Start:
------------
1. Set up your API key:
   - Copy .env.example to .env
   - Edit .env and add your OpenRouter or Anthropic API key

2. Check system status:
   haro.exe status

3. Download speech models (first time only):
   haro.exe download-model whisper tiny
   haro.exe download-model piper en_US-lessac-medium

4. Run HARO:
   haro.exe run

Available Commands:
-------------------
  haro.exe --help           Show all available commands
  haro.exe status           Check system and dependency status
  haro.exe run              Start the voice assistant
  haro.exe test-audio       Test audio input/output
  haro.exe test-stt         Test speech-to-text
  haro.exe test-tts         Test text-to-speech
  haro.exe test-wake        Test wake word detection
  haro.exe models           List installed speech models
  haro.exe download-model   Download a speech model
  haro.exe init-context     Initialize .context directory

Configuration:
--------------
  config/default.yaml       Default configuration (bundled)
  .env                      API keys (create from .env.example)
  .context/                 Session data and knowledge base

Models are downloaded to: %USERPROFILE%\\.cache\\haro\\models\\

For more information, see: https://github.com/haro-project/haro
"""
    readme_path.write_text(readme_content)
    print(f"Created: {readme_path}")


def main() -> int:
    """Main build entry point."""
    parser = argparse.ArgumentParser(description='Build HARO Windows executable')
    parser.add_argument('--clean', action='store_true', help='Clean build directories first')
    args = parser.parse_args()

    # Find project root (parent of scripts/)
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent

    print("=" * 60)
    print("HARO Windows Build")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print()

    # Pre-flight checks
    if not check_python_version():
        return 1

    if not check_pyinstaller():
        return 1

    print()

    # Clean if requested
    if args.clean:
        print("Cleaning previous build...")
        clean_build_dirs(project_root)
        print()

    # Run PyInstaller
    print("Running PyInstaller...")
    if not run_pyinstaller(project_root):
        print()
        print("Build FAILED")
        return 1

    print()
    print("-" * 60)

    # Post-build setup
    print("Post-build setup...")
    post_build_setup(project_root)

    print()
    print("=" * 60)
    print("Build SUCCESSFUL")
    print("=" * 60)
    print()
    print(f"Output: {project_root / 'dist' / 'HARO'}")
    print()
    print("Next steps:")
    print("  1. cd dist\\HARO")
    print("  2. copy .env.example .env")
    print("  3. Edit .env with your API key")
    print("  4. haro.exe status")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
