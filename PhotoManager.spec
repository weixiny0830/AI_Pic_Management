# PhotoManager.spec (ONEDIR, stable)
# Build: pyinstaller PhotoManager.spec

# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all

block_cipher = None

clip_datas, clip_binaries, clip_hidden = collect_all("clip")
heif_datas, heif_binaries, heif_hidden = collect_all("pillow_heif")

datas = (
    clip_datas
    + heif_datas
    + [
        ("assets/icon.ico", "assets"),
    ]
)

binaries = clip_binaries + heif_binaries

hiddenimports = clip_hidden + heif_hidden + [
    "torch.distributed",
    "torch.testing",
]

a = Analysis(
    ["app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tensorboard",
        "torch.utils.tensorboard",
        "matplotlib",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# IMPORTANT: exclude_binaries=True means binaries go to COLLECT, not inside the exe
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PhotoManager",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon="assets/icon.ico",
    version="version.txt",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="PhotoManager",
)
