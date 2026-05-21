#!/usr/bin/env pwsh
# Build a single-file Windows executable with PyInstaller.
# Run from the repo root:  .\scripts\build_exe.ps1
$ErrorActionPreference = "Stop"
pyinstaller --noconfirm --onefile --windowed `
    --name JoyOfPaintingToolkit `
    --collect-submodules scipy `
    jop_toolkit/__main__.py
Write-Host "Built dist/JoyOfPaintingToolkit.exe"
