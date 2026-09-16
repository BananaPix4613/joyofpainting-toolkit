#!/usr/bin/env pwsh
# Build a single-file Windows executable with PyInstaller.
# Run from the repo root:  .\scripts\build_exe.ps1
$ErrorActionPreference = "Stop"
# --onefile bundles nothing it is not told about, so icon.png needs --add-data
# or app._asset() finds nothing at runtime. --icon sets the exe's own icon;
# PyInstaller converts the PNG using Pillow, which is already a dependency.
#
# The two scipy modules are named explicitly rather than swept in with
# --collect-submodules scipy. That flag pulls every scipy submodule, including
# scipy._lib.array_api_compat.torch, which drags in torch, torchvision, numba,
# llvmlite and matplotlib, plus scipy's entire test suite. Measured: 277 MiB and
# a 281 s build, against 83 MiB and 45 s for the same app.
#
# `python -m` rather than the bare command: the console script only resolves if
# the interpreter's Scripts directory is on PATH, which it is not inside an
# unactivated venv.
python -m PyInstaller --noconfirm --onefile --windowed `
    --name JoyOfPaintingToolkit `
    --icon icon.png `
    --add-data "icon.png;." `
    --hidden-import scipy.ndimage `
    --hidden-import scipy.spatial `
    --exclude-module torch --exclude-module torchvision `
    --exclude-module matplotlib --exclude-module numba --exclude-module llvmlite `
    --exclude-module sympy --exclude-module pytest --exclude-module tensorflow `
    --exclude-module pandas --exclude-module IPython --exclude-module notebook `
    jop_toolkit/__main__.py
Write-Host "Built dist/JoyOfPaintingToolkit.exe"