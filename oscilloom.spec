# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for bundling the Oscilloom backend into a single executable.
# Build: pyinstaller oscilloom.spec

import os

block_cipher = None

# Collect MNE data files (channel locations, etc.)
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

mne_datas = collect_data_files("mne")
mne_hiddenimports = collect_submodules("mne")

a = Analysis(
    ["backend/main.py"],
    pathex=["."],
    binaries=[],
    datas=[
        ("backend/templates", "backend/templates"),
        ("backend/registry", "backend/registry"),
    ]
    + mne_datas,
    hiddenimports=[
        "uvicorn",
        "uvicorn.logging",
        "uvicorn.loops",
        "uvicorn.loops.auto",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",
        "multipart",
        "jinja2",
        "matplotlib",
        "matplotlib.backends.backend_agg",
        "numpy",
        "scipy",
        "sklearn",
        "yasa",
        "autoreject",
        "edfio",
        "pybv",
    ]
    + mne_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="oscilloom-server",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="branding/icon.ico" if os.path.exists("branding/icon.ico") else None,
)
