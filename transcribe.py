#!/usr/bin/env python3
"""Transcribe audio/video files with faster-whisper. Any format ffmpeg reads."""
import argparse
import ctypes
import subprocess
import sys
from pathlib import Path

import av
import numpy as np

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


def audio_streams(path):
    """[(index, codec, language, title)] for every audio track in the container."""
    with av.open(str(path), metadata_errors="ignore") as container:
        return [(i, s.codec_context.name, s.metadata.get("language", "und"),
                 s.metadata.get("title", ""))
                for i, s in enumerate(container.streams.audio)]


def decode_stream(path, index):
    """faster-whisper always decodes audio stream 0, so pull the wanted one via ffmpeg."""
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", f"0:a:{index}",
         "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if r.returncode or not r.stdout:
        sys.exit(f"ffmpeg failed on stream {index} of {path}: {r.stderr.decode().strip()}")
    return np.frombuffer(r.stdout, np.int16).astype(np.float32) / 32768.0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("-m", "--model", default="large-v3-turbo")
    p.add_argument("-l", "--language", default=None, help="e.g. en, ro. Default: autodetect")
    p.add_argument("-d", "--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("-c", "--compute-type", default="default",
                   help="int8, int8_float16, float16, float32")
    p.add_argument("-b", "--batch-size", type=int, default=16, help="lower if VRAM runs out")
    p.add_argument("-s", "--stream", type=int, default=None,
                   help="audio stream index to transcribe (default: 0). Listed per file")
    args = p.parse_args()

    model = BatchedInferencePipeline(
        WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    )

    for f in args.files:
        if not f.is_file():
            print(f"skip (not a file): {f}", file=sys.stderr)
            continue
        streams = audio_streams(f)
        if not streams:
            print(f"skip (no audio stream): {f}", file=sys.stderr)
            continue
        if len(streams) > 1:
            for i, codec, lang, title in streams:
                mark = "*" if i == (args.stream or 0) else " "
                print(f" {mark} stream {i}: {codec} {lang} {title}".rstrip(), file=sys.stderr)
        if args.stream is not None and args.stream >= len(streams):
            sys.exit(f"{f}: no audio stream {args.stream}, file has {len(streams)}")

        audio = str(f) if args.stream in (None, 0) else decode_stream(f, args.stream)
        segments, info = model.transcribe(audio, language=args.language,
                                          batch_size=args.batch_size)
        print(f"== {f.name} [{info.language} {info.duration:.0f}s]", file=sys.stderr)

        lines = []
        for seg in segments:
            line = f"[{format_timestamp(seg.start, True)}] {seg.text.strip()}"
            print(line, file=sys.stderr)
            lines.append(line)

        if not lines:
            print("no speech found - wrong stream?", file=sys.stderr)
        # per-stream name so transcribing both tracks of a call does not overwrite
        tag = f".a{args.stream or 0}" if len(streams) > 1 else ""
        out = f.with_suffix(f"{f.suffix}{tag}.txt")
        out.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
        print(f"-> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
