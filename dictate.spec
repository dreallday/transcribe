# PyInstaller spec for the Windows build. From the build dir: pyinstaller dictate.spec
# Produces dist/dictate/dictate.exe (onedir - onefile would re-extract ~1 GB of CUDA on
# every launch). Model weights are NOT bundled; they download to the HF cache on first run.
import os
import shutil
import site
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
for pkg in ("nicegui", "faster_whisper", "ctranslate2", "onnxruntime", "av", "tokenizers"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

ffmpeg = shutil.which("ffmpeg")
if not ffmpeg:
    raise SystemExit("ffmpeg.exe not on PATH - winget install Gyan.FFmpeg")
binaries.append((ffmpeg, "."))

# CUDA runtime from the nvidia-*-cu12 wheels, flat next to the exe so the loader finds it
cuda = [dll for pkgs in site.getsitepackages()
        for dll in Path(pkgs).glob("nvidia/*/bin/*.dll")]
if not cuda:
    raise SystemExit("no CUDA DLLs - pip install nvidia-cublas-cu12 nvidia-cudnn-cu12")
binaries += [(str(dll), ".") for dll in cuda]

a = Analysis(
    ["gui.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + ["transcribe"],
    excludes=["tkinter", "matplotlib"],
)
pyz = PYZ(a.pure)
# DICTATE_CONSOLE=1 builds a console exe so startup errors are visible
exe = EXE(pyz, a.scripts, exclude_binaries=True, name="dictate",
          console=bool(os.environ.get("DICTATE_CONSOLE")))
coll = COLLECT(exe, a.binaries, a.datas, name="dictate")
