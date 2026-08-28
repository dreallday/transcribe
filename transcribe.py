#!/usr/bin/env python3
"""Transcribe audio/video files with faster-whisper. Any format ffmpeg reads."""
import argparse
import ctypes
import sys
from pathlib import Path

# ctranslate2 dlopens libcublas/libcudnn by soname; pip wheels hide them in
# site-packages/nvidia/*/lib, which the loader does not search. Preload by path,
# cublas first (cudnn links against it).
for _lib in sorted(Path(__file__).resolve().parent.glob(".venv/lib/*/site-packages/nvidia/*/lib/*.so.*")):
    try:
        ctypes.CDLL(str(_lib), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass

from faster_whisper import BatchedInferencePipeline, WhisperModel
from faster_whisper.utils import format_timestamp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("-m", "--model", default="large-v3-turbo")
    p.add_argument("-l", "--language", default=None, help="e.g. en, ro. Default: autodetect")
    p.add_argument("-d", "--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("-c", "--compute-type", default="default",
                   help="int8, int8_float16, float16, float32")
    p.add_argument("-b", "--batch-size", type=int, default=16, help="lower if VRAM runs out")
    args = p.parse_args()

    model = BatchedInferencePipeline(
        WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    )

    for f in args.files:
        if not f.is_file():
            print(f"skip (not a file): {f}", file=sys.stderr)
            continue
        segments, info = model.transcribe(str(f), language=args.language,
                                          batch_size=args.batch_size)
        print(f"== {f.name} [{info.language} {info.duration:.0f}s]", file=sys.stderr)

        lines = []
        for seg in segments:
            line = f"[{format_timestamp(seg.start, True)}] {seg.text.strip()}"
            print(line, file=sys.stderr)
            lines.append(line)

        out = f.with_suffix(f"{f.suffix}.txt")
        out.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
        print(f"-> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
