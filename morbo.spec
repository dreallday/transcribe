# PyInstaller spec for the Windows build. From the build dir: pyinstaller morbo.spec
# Produces dist/morbo/morbo.exe (onedir - onefile would re-extract the CUDA libraries on
# every launch). Model weights are NOT bundled; they download to the HF cache on first run.
import os
import shutil
import site
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

# The only CUDA libraries a transcription actually loads, measured by listing the modules of
# a running morbo.exe. Everything else the wheels ship - the cuDNN engines, ops, adv and
# nvrtc - is never touched by ctranslate2's whisper path and costs 1 GB.
# If a future model or compute type fails with "Library ... is not found", add it here.
CUDA_KEEP = {"cublas64_12.dll", "cublaslt64_12.dll"}
CUDA_PREFIX = ("cublas", "cudnn", "nvrtc", "cufft", "curand", "cusolver", "cusparse", "nccl")

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
        for dll in Path(pkgs).glob("nvidia/*/bin/*.dll")
        if dll.name.lower() in CUDA_KEEP]
if not cuda:
    raise SystemExit("no CUDA DLLs - pip install nvidia-cublas-cu12 nvidia-cudnn-cu12")
binaries += [(str(dll), ".") for dll in cuda]


def keep(dest):
    """Drop CUDA libraries we do not need, and the second copy of the ones we do.

    PyInstaller's dependency scan pulls the whole nvidia/ tree in as well, so without this
    every kept library is bundled twice.
    """
    dest = Path(dest)
    if not dest.name.lower().startswith(CUDA_PREFIX):
        return True
    if dest.parent.name == "ctranslate2":
        return True  # ctranslate2 loads the cudnn64_9.dll sitting in its own directory
    return dest.parent == Path(".") and dest.name.lower() in CUDA_KEEP


a = Analysis(
    ["morbo.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + ["transcribe"],
    excludes=["tkinter", "matplotlib"],
)
a.binaries = [entry for entry in a.binaries if keep(entry[0])]

pyz = PYZ(a.pure)
# MORBO_CONSOLE=1 builds a console exe so startup errors are visible
exe = EXE(pyz, a.scripts, exclude_binaries=True, name="morbo",
          console=bool(os.environ.get("MORBO_CONSOLE")))
coll = COLLECT(exe, a.binaries, a.datas, name="morbo")
