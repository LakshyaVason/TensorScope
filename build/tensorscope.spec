# PyInstaller spec for TensorScope.
#
# Usage:
#   pip install pyinstaller
#   pyinstaller build/tensorscope.spec
#
# Produces a dist/TensorScope/ folder (--onedir) with a TensorScope executable.
# --onefile is not recommended: torch has too many large shared libraries for a
# single-file bundle to extract on each launch without unacceptable startup cost.
#
# To bundle GPU torch instead of CPU-only, install torch from the CUDA channel first:
#   pip install --index-url https://download.pytorch.org/whl/cu124 torch
# then re-run this spec.

import sys
from pathlib import Path

ROOT = Path(SPECPATH).parent  # one level up from build/

block_cipher = None

a = Analysis(
    [str(ROOT / "TensorScope.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[],
    hiddenimports=[
        "platformdirs",
        "numpy",
        "matplotlib",
        "matplotlib.backends.backend_qt5agg",
        "PyQt5",
        "PyQt5.QtCore",
        "PyQt5.QtWidgets",
        "PyQt5.QtGui",
        "sqlite3",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TensorScope",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # no terminal window on Windows
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="TensorScope",
)
