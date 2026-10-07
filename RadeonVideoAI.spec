# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller Spec File for RadeonVideoAI Portable Suite.
Target: Windows 10/11 64-bit with any DirectX 12 GPU (NVIDIA/AMD/Intel, via
ONNX Runtime DirectML) and any CPU. Bundles PyQt6, PyTorch (CPU, used only
to load/export AI weights), ONNX Runtime DirectML, and standalone FFmpeg
binaries.
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs

block_cipher = None
PROJECT_DIR = os.path.abspath(SPECPATH)

# 1. Collect PyTorch data and DLL dependencies
#
# Deliberately NOT bundling models/ (models/weights/*): every AI checkpoint
# under there — Real-ESRGAN .pth files, the Stable Diffusion ONNX export
# cache, the RIFE Vulkan binary — is downloaded on first use at runtime, not
# meant to ship in the portable build. It used to be blanket-included here,
# which silently made every build bundle whatever happened to already be
# cached on the machine doing the build (the Stable Diffusion onnx_cache/
# alone reached 4+ GB during development and made PyInstaller's file
# enumeration pass grind to a near-standstill).
datas = [
    (os.path.join(PROJECT_DIR, "core"), "core"),
    (os.path.join(PROJECT_DIR, "gui"), "gui"),
]
datas += collect_data_files('torch')
datas += collect_data_files('onnxruntime')

# Generative recreation tab (Stable Diffusion img2img): optimum/diffusers
# ship json/yaml config templates as package data, not just code, so these
# need explicit collection the way torch/onnxruntime do above. transformers
# is covered by its own community-maintained hook in pyinstaller-hooks-contrib.
datas += collect_data_files('diffusers')
datas += collect_data_files('optimum')
datas += collect_data_files('accelerate')
datas += collect_data_files('tokenizers')
datas += collect_data_files('huggingface_hub')

# 2. Binaries: Include local ffmpeg and ffprobe if present
binaries = []
for binary_name in ["ffmpeg.exe", "ffprobe.exe"]:
    bin_path = os.path.join(PROJECT_DIR, binary_name)
    if os.path.isfile(bin_path):
        binaries.append((bin_path, "."))

binaries += collect_dynamic_libs('torch')
binaries += collect_dynamic_libs('onnxruntime')
binaries += collect_dynamic_libs('tokenizers')
binaries += collect_dynamic_libs('safetensors')

# 3. Hidden imports to prevent runtime missing module errors
hiddenimports = [
    'PyQt6',
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'onnx',
    'onnxruntime',
    'requests',
    'psutil',
    'numpy',
    'PIL',
    'json',
    'platform',
    'ctypes',
    'subprocess',
    # Generative recreation tab
    'optimum',
    'optimum.onnxruntime',
    'optimum.exporters',
    'optimum.exporters.onnx',
    'optimum.utils',
    'diffusers',
    'diffusers.pipelines.stable_diffusion',
    'accelerate',
    'huggingface_hub',
    'safetensors',
    'tokenizers',
]
hiddenimports += collect_submodules('torch')
hiddenimports += collect_submodules('onnxruntime')
hiddenimports += collect_submodules('optimum')
hiddenimports += collect_submodules('diffusers')
hiddenimports += collect_submodules('accelerate')

a = Analysis(
    ['main.py'],
    pathex=[PROJECT_DIR],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'scipy', 'scapy', 'IPython', 'notebook'],
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
    name='RadeonVideoAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # Desktop GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='RadeonVideoAI',
)
