#!/usr/bin/env python3
"""Transcribe audio/video files with faster-whisper. Any format ffmpeg reads."""
import argparse
import ctypes
import queue
import shutil
import subprocess
import sys
import threading
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


# ponytail: RMS gate, not a real VAD. Cheap and tunable; if it clips speech in a noisy
# room, swap in faster_whisper.vad.get_speech_timestamps over the tail of the buffer.
def live(model, args):
    """Capture the mic, preview every --interval, commit the line once the speaker pauses."""
    fmt = args.input_format or {"linux": "pulse", "darwin": "avfoundation",
                                "win32": "dshow"}.get(sys.platform, "pulse")
    proc = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", fmt, "-i", args.input,
         "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
        stdout=subprocess.PIPE,
    )
    # a thread keeps the pipe drained while the GPU is busy, else pulse overruns and drops audio
    blocks = queue.Queue()
    threading.Thread(target=lambda: ([blocks.put(b) for b in iter(
        lambda: proc.stdout.read(8000), b"")], blocks.put(None)), daemon=True).start()

    out = open(args.out, "a", buffering=1, encoding="utf-8") if args.out else None
    preview_ok = sys.stdout.isatty()
    buf, silence, elapsed, next_preview = np.empty(0, np.float32), 0.0, 0.0, 0.0
    print(f"listening on {fmt}:{args.input} - ctrl-c to stop", file=sys.stderr)

    # batched + vad_filter=False only accepts audio under the model's 30s window, so cap below it
    cap = 25 * 16000

    def text_of(audio):
        # beam 1 and no VAD: ~20% faster than the defaults, and the RMS gate already trimmed silence
        segments, _ = model.transcribe(audio, language=args.language, beam_size=1,
                                       vad_filter=False, batch_size=args.batch_size)
        return " ".join(seg.text.strip() for seg in segments).strip()

    def show(text, at, final):
        if final:
            line = f"[{format_timestamp(at, True)}] {text}"
            print(f"\r\033[K{line}" if preview_ok else line, flush=True)
            if out:
                out.write(line + "\n")
        elif preview_ok:
            width = shutil.get_terminal_size().columns - 1
            print(f"\r\033[K{text[-width:]}", end="", flush=True)

    try:
        while (raw := blocks.get()) is not None:
            audio = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
            elapsed += len(audio) / 16000
            quiet = np.sqrt(np.mean(audio ** 2)) < args.threshold
            if quiet and not len(buf):
                continue  # still waiting for someone to speak
            buf = np.concatenate([buf, audio])
            silence = silence + len(audio) / 16000 if quiet else 0.0
            start = elapsed - len(buf) / 16000

            if silence >= args.pause or len(buf) >= cap:
                if text := text_of(buf):
                    show(text, start, True)
                buf, silence, next_preview = np.empty(0, np.float32), 0.0, 0.0
            elif preview_ok and elapsed >= next_preview and len(buf) >= 16000:
                next_preview = elapsed + args.interval
                show(text_of(buf), start, False)
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
        if len(buf) >= 16000 and (text := text_of(buf)):
            show(text, elapsed - len(buf) / 16000, True)
        if out:
            out.close()
            print(f"-> {args.out}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="*", type=Path)
    p.add_argument("--mic", action="store_true", help="live-transcribe the microphone")
    p.add_argument("-i", "--input", default="default",
                   help="mic device for --mic (list them: ffmpeg -sources pulse)")
    p.add_argument("--input-format", default=None, help="ffmpeg input format for --mic")
    p.add_argument("-o", "--out", type=Path, default=None, help="append --mic lines to a file")
    p.add_argument("--interval", type=float, default=1.0,
                   help="seconds between live previews of the sentence being spoken")
    p.add_argument("--pause", type=float, default=0.8,
                   help="seconds of silence that end an utterance")
    p.add_argument("--threshold", type=float, default=0.01,
                   help="RMS below this counts as silence. Raise in a noisy room")
    p.add_argument("-m", "--model", default="large-v3-turbo")
    p.add_argument("-l", "--language", default=None, help="e.g. en, ro. Default: autodetect")
    p.add_argument("-d", "--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("-c", "--compute-type", default="default",
                   help="int8, int8_float16, float16, float32")
    p.add_argument("-b", "--batch-size", type=int, default=16, help="lower if VRAM runs out")
    p.add_argument("-s", "--stream", type=int, default=None,
                   help="audio stream index to transcribe (default: 0). Listed per file")
    args = p.parse_args()

    if not args.files and not args.mic:
        p.error("give at least one file, or --mic")

    model = BatchedInferencePipeline(
        WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    )
    if args.mic:
        return live(model, args)

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
