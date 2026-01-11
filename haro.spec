# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for HARO Windows build.

Build with: pyinstaller haro.spec
Or use: python scripts/build_windows.py

Output: dist/HARO/ folder containing haro.exe and all dependencies
"""

import os
import sys
from pathlib import Path

# Get the project root directory
project_root = Path(SPECPATH)

# Block cipher for encryption (None = no encryption)
block_cipher = None

# Collect data files
datas = [
    # Bundle default configuration
    (str(project_root / 'config' / 'default.yaml'), 'config'),
    # Bundle .context template directory
    (str(project_root / '.context'), '.context'),
    # Bundle .env.example for users
    (str(project_root / '.env.example'), '.'),
]

# Hidden imports that PyInstaller might miss
hiddenimports = [
    # Audio libraries
    'sounddevice',
    '_sounddevice_data',
    'sounddevice._sounddevice',

    # NumPy (required by audio/ML)
    'numpy',
    'numpy.core._methods',
    'numpy.lib.format',

    # Speech recognition (faster-whisper)
    'faster_whisper',
    'ctranslate2',

    # Text-to-speech (piper)
    'piper_tts',
    'onnxruntime',

    # API clients
    'anthropic',
    'anthropic._client',
    'anthropic.resources',
    'httpx',
    'httpx._transports',
    'httpx._transports.default',
    'httpcore',

    # Configuration
    'yaml',
    'dotenv',

    # CLI and UI
    'click',
    'click.core',
    'click.decorators',
    'rich',
    'rich.console',
    'rich.table',
    'rich.panel',
    'rich.progress',

    # Logging
    'structlog',
    'structlog.processors',
    'structlog.stdlib',

    # File watching
    'watchdog',
    'watchdog.observers',
    'watchdog.events',

    # SSL/TLS for API calls
    'ssl',
    'certifi',
]

# Collect binaries from packages that need them
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# Collect sounddevice data (PortAudio binaries)
try:
    datas += collect_data_files('sounddevice')
    datas += collect_data_files('_sounddevice_data')
except Exception:
    pass

# Collect faster-whisper/ctranslate2 binaries
try:
    datas += collect_data_files('faster_whisper')
    datas += collect_data_files('ctranslate2')
except Exception:
    pass

# Collect piper-tts data
try:
    datas += collect_data_files('piper_tts')
except Exception:
    pass

# Collect onnxruntime binaries
try:
    datas += collect_data_files('onnxruntime')
except Exception:
    pass

# Collect certifi CA bundle for HTTPS
try:
    datas += collect_data_files('certifi')
except Exception:
    pass

# Analysis
a = Analysis(
    [str(project_root / 'src' / 'haro' / '__main__.py')],
    pathex=[str(project_root / 'src')],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary modules to reduce size
        'tkinter',
        'matplotlib',
        'scipy',
        'pandas',
        'PIL',
        'cv2',
        'torch',  # We use faster-whisper with ctranslate2, not torch
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Create PYZ archive
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Create executable
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # Required for COLLECT (folder mode)
    name='haro',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # CLI app needs console
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if desired: 'assets/haro.ico'
)

# Collect all files into folder (not single file)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='HARO',  # Output folder name: dist/HARO/
)
