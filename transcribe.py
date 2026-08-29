#!/usr/bin/env python3
"""Transcribe audio and video files, or the microphone, with faster-whisper."""
import argparse
import atexit
import ctypes
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import av
import numpy as np

SAMPLE_RATE = 16000
BLOCK = 800  # samples read at a time: 50ms, small enough to drive a live level meter
WINDOW = 25 * SAMPLE_RATE  # batched decoding without VAD only takes audio under 30s

HERE = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
# a PyInstaller build ships its own ffmpeg; a source checkout uses the one on PATH
FROZEN_WIN = getattr(sys, "frozen", False) and sys.platform == "win32"
FFMPEG = str(HERE / "ffmpeg.exe") if FROZEN_WIN else "ffmpeg"
# a windowed build has no console of its own, so each child would pop one - suppress them
NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}


def _find_cuda() -> None:
    """Make the CUDA libraries from the nvidia-*-cu12 wheels loadable.

    ctranslate2 asks the OS for them by bare name, and neither loader looks inside
    site-packages: on Windows a plain LoadLibrary searches PATH (add_dll_directory is
    not enough, it only covers Python's own extension loading), on Linux dlopen needs
    the file preloaded by absolute path, cuBLAS before cuDNN.
    """
    if sys.platform == "win32":
        dirs = [d for d in [HERE, *HERE.glob("nvidia/*/bin"),
                            *Path(sys.prefix).glob("Lib/site-packages/nvidia/*/bin")]
                if d.is_dir()]
        for d in dirs:
            os.add_dll_directory(str(d))
        os.environ["PATH"] = os.pathsep.join([*map(str, dirs), os.environ.get("PATH", "")])
        return
    for lib in sorted(HERE.glob(".venv/lib/*/site-packages/nvidia/*/lib/*.so.*")):
        try:
            ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


_find_cuda()

from faster_whisper import BatchedInferencePipeline, WhisperModel  # noqa: E402
from faster_whisper.utils import format_timestamp  # noqa: E402


def pcm(raw: bytes) -> np.ndarray:
    """Signed 16-bit little-endian bytes -> the float32 samples whisper wants."""
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


# --- files ------------------------------------------------------------------


def audio_streams(path: Path) -> list[tuple[int, str, str, str]]:
    """(index, codec, language, title) for every audio track in the container."""
    with av.open(str(path), metadata_errors="ignore") as container:
        return [(i, s.codec_context.name, s.metadata.get("language", "und"),
                 s.metadata.get("title", ""))
                for i, s in enumerate(container.streams.audio)]


def decode_stream(path: Path, index: int) -> np.ndarray:
    """faster-whisper always decodes audio stream 0, so pull the wanted one via ffmpeg."""
    r = subprocess.run(
        [FFMPEG, "-nostdin", "-v", "error", "-i", str(path), "-map", f"0:a:{index}",
         "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
        capture_output=True, **NO_WINDOW,
    )
    if r.returncode or not r.stdout:
        sys.exit(f"ffmpeg failed on stream {index} of {path}: {r.stderr.decode().strip()}")
    return pcm(r.stdout)


def transcribe_file(model, path: Path, args) -> None:
    """Write a timestamped transcript next to the input file."""
    streams = audio_streams(path)
    if not streams:
        print(f"skip (no audio stream): {path}", file=sys.stderr)
        return
    if len(streams) > 1:
        for i, codec, language, title in streams:
            mark = "*" if i == (args.stream or 0) else " "
            print(f" {mark} stream {i}: {codec} {language} {title}".rstrip(), file=sys.stderr)
    if args.stream is not None and args.stream >= len(streams):
        sys.exit(f"{path}: no audio stream {args.stream}, file has {len(streams)}")

    audio = str(path) if args.stream in (None, 0) else decode_stream(path, args.stream)
    segments, info = model.transcribe(audio, language=args.language,
                                      batch_size=args.batch_size)
    print(f"== {path.name} [{info.language} {info.duration:.0f}s]", file=sys.stderr)

    lines = []
    for seg in segments:
        line = f"[{format_timestamp(seg.start, True)}] {seg.text.strip()}"
        print(line, file=sys.stderr)
        lines.append(line)
    if not lines:
        print("no speech found - wrong stream?", file=sys.stderr)

    # per-stream name so transcribing both tracks of a call does not overwrite
    tag = f".a{args.stream or 0}" if len(streams) > 1 else ""
    out = path.with_suffix(f"{path.suffix}{tag}.txt")
    out.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {out}", file=sys.stderr)


# --- capture devices --------------------------------------------------------


def parse_dshow_devices(listing: str) -> list[tuple[str, str]]:
    """(friendly name, capture name) pairs out of ffmpeg's dshow device listing.

    The capture name is dshow's own "Alternative name", a PnP path. Friendly names can
    contain a colon ("Wave:XLR") and a colon is dshow's separator, so handing one back
    gives "Malformed dshow input string".
    """
    devices: list[list[str]] = []
    for raw in listing.splitlines():
        line = re.sub(r"^\[[^\]]+\]\s*", "", raw).strip()
        if m := re.fullmatch(r'"(.+)" \(audio\)', line):
            devices.append([m.group(1), m.group(1)])
        elif (m := re.fullmatch(r'Alternative name "(.+)"', line)) and devices:
            if devices[-1][0] == devices[-1][1]:  # this device has no capture name yet
                devices[-1][1] = m.group(1)
    return [(friendly, capture) for friendly, capture in devices]


def dshow_audio_devices() -> list[tuple[str, str]]:
    return parse_dshow_devices(subprocess.run(
        [FFMPEG, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
        capture_output=True, text=True, **NO_WINDOW,
    ).stderr)


# Core Audio knows the default recording device; dshow does not. Both identify an endpoint
# by the same GUID - it ends the Core Audio id and sits inside dshow's "wave_{...}" name.
DEFAULT_ENDPOINT_PS = """
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class DefaultCapture {
  [ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")] class Enumerator { }
  [ComImport, Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"),
   InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  interface IMMDeviceEnumerator {
    int EnumAudioEndpoints(int flow, int mask, out IntPtr col);
    int GetDefaultAudioEndpoint(int flow, int role, out IMMDevice dev);
  }
  [ComImport, Guid("D666063F-1587-4E43-81F1-B948E807363F"),
   InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  interface IMMDevice {
    int Activate(ref Guid iid, int ctx, IntPtr p, out IntPtr o);
    int OpenPropertyStore(int access, out IntPtr store);
    int GetId([MarshalAs(UnmanagedType.LPWStr)] out string id);
  }
  public static string Id() {
    var e = (IMMDeviceEnumerator)(new Enumerator());
    IMMDevice d;
    if (e.GetDefaultAudioEndpoint(1, 0, out d) != 0) return "";
    string id; d.GetId(out id); return id;
  }
}
"@
[DefaultCapture]::Id()
"""


def windows_default_endpoint() -> str | None:
    """GUID of the Windows default recording device, or None."""
    script = Path(tempfile.gettempdir()) / "dictate_default_device.ps1"
    script.write_text(DEFAULT_ENDPOINT_PS, encoding="utf-8")
    try:
        out = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-File", str(script)],
            capture_output=True, text=True, timeout=30, **NO_WINDOW,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.search(r"\{([0-9a-fA-F-]{36})\}\s*$", out.strip())
    return m.group(1).lower() if m else None


def dshow_input(wanted: str) -> str:
    """Resolve --input to something dshow accepts."""
    devices = dshow_audio_devices()
    if not devices:
        sys.exit("no dshow audio device found - check Windows microphone privacy settings")
    if wanted != "default":
        name = wanted.removeprefix("audio=")
        capture = next((c for friendly, c in devices if name in (friendly, c)), None)
        return f"audio={capture}" if capture else wanted  # unknown: hand it to ffmpeg as-is
    for friendly, _ in devices:
        print(f"   audio device: {friendly}", file=sys.stderr)
    endpoint = windows_default_endpoint()
    matched = [d for d in devices if endpoint and endpoint in d[1].lower()]
    # nothing matched (a virtual device with no Core Audio entry): prefer a name saying mic
    friendly, capture = (matched or [d for d in devices if "mic" in d[0].lower()] or devices)[0]
    print(f'   using "{friendly}" - override with -i "<name>"', file=sys.stderr)
    return f"audio={capture}"


def parse_pulse_sources(listing: str) -> list[tuple[str, str]]:
    """(value for --input, label) out of `ffmpeg -sources pulse`."""
    return [(m.group(1), m.group(2))
            for line in listing.splitlines()
            if (m := re.match(r"\s*\*?\s*(\S+)\s+\[(.*)\]", line))]


def audio_inputs() -> list[tuple[str, str]]:
    """(value for --input, label to show) for every capture device on this platform."""
    if sys.platform == "win32":
        return [("default", "Windows default"),
                *((f"audio={cap}", friendly) for friendly, cap in dshow_audio_devices())]
    if sys.platform != "linux":
        return [("default", "default")]
    listing = subprocess.run([FFMPEG, "-hide_banner", "-sources", "pulse"],
                             capture_output=True, text=True)
    return [("default", "default"), *parse_pulse_sources(listing.stdout + listing.stderr)]


# --- typing into whatever window has focus ----------------------------------


def quote(text: str) -> str:
    """Escape for a PowerShell single-quoted string."""
    return text.replace("'", "''")


def make_typer():
    """Return send(text), which inserts text wherever the keyboard focus is."""
    if sys.platform == "win32" or "microsoft" in os.uname().release.lower():
        # one long-lived shell: spawning powershell.exe per sentence costs ~1s
        ps = subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", "-"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1, **NO_WINDOW,
        )
        # block until the assembly is loaded, else the first sentence types seconds late
        ps.stdin.write("Add-Type -AssemblyName System.Windows.Forms; "
                       "$saved = Get-Clipboard -Raw; Write-Output READY\n")
        ps.stdout.readline()

        @atexit.register
        def restore_clipboard():
            try:
                ps.stdin.write("if ($saved) { Set-Clipboard -Value $saved }\nexit\n")
                ps.wait(timeout=5)
            except Exception:
                ps.kill()

        def send(text: str) -> None:
            # Paste, never SendKeys typing: its brace escapes send the unshifted key, so
            # "50% (net)" arrives as "505 9net0", and accents cannot be typed at all.
            # Restoring the clipboard per sentence truncates the paste - that happens on exit.
            ps.stdin.write(f"Set-Clipboard -Value '{quote(text)}'; "
                           f"[System.Windows.Forms.SendKeys]::SendWait('^v'); "
                           f"Start-Sleep -Milliseconds 150\n")

        return send

    if tool := shutil.which("xdotool"):  # X11
        return lambda text: subprocess.run([tool, "type", "--clearmodifiers", "--", text])
    if tool := shutil.which("wtype"):  # wayland
        return lambda text: subprocess.run([tool, "--", text])
    sys.exit("--type needs powershell.exe (WSL/Windows), xdotool or wtype")


# --- live capture -----------------------------------------------------------


def open_capture(args) -> tuple[subprocess.Popen, queue.Queue]:
    """Start ffmpeg on the chosen input; return it and a queue of raw audio blocks.

    A thread does the reading so the pipe keeps draining while the GPU is busy -
    otherwise the audio driver overruns its own buffer and drops sound.
    """
    fmt = args.input_format or {"linux": "pulse", "darwin": "avfoundation",
                                "win32": "dshow"}.get(sys.platform, "pulse")
    if fmt == "dshow":
        args.input = dshow_input(args.input)
    # a file standing in for the mic has to be paced, or it floods in at decode speed
    rate = [] if fmt in ("pulse", "dshow", "avfoundation") else ["-re"]
    proc = subprocess.Popen(
        [FFMPEG, "-nostdin", "-v", "error", *rate, "-f", fmt, "-i", args.input,
         "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
        stdout=subprocess.PIPE, **NO_WINDOW,
    )
    blocks: queue.Queue = queue.Queue()

    def pump() -> None:
        for raw in iter(lambda: proc.stdout.read(BLOCK * 2), b""):
            blocks.put(raw)
        blocks.put(None)  # end of stream

    threading.Thread(target=pump, daemon=True).start()
    print(f"listening on {fmt}:{args.input} - ctrl-c to stop", file=sys.stderr)
    return proc, blocks


# ponytail: RMS gate, not a real VAD. Cheap and tunable; if it clips speech in a noisy
# room, swap in faster_whisper.vad.get_speech_timestamps over the tail of the buffer.
def live(model, args, is_on=None, on_text=None, on_level=None, handle=None) -> None:
    """Transcribe the microphone until the input ends or ctrl-c.

    A sentence is whatever was said between two pauses: audio piles up while the level is
    above --threshold, is re-transcribed every --interval as a preview, and is committed
    after --pause seconds of silence.

    is_on: optional threading.Event - audio is discarded while it is clear.
    on_text: optional callback(text, final) - final=False while the sentence is still
        being spoken, True when it is committed.
    on_level: optional callback(level) with the smoothed input level, 20 times a second.
    handle: optional dict - the ffmpeg process is stored under "proc", so a caller can
        terminate it and end this call, e.g. to reopen on another device.
    """
    proc, blocks = open_capture(args)
    if handle is not None:
        handle["proc"] = proc
    out = open(args.out, "a", buffering=1, encoding="utf-8") if args.out else None
    typer = make_typer() if args.type else None
    redraw = sys.stdout.isatty()  # a terminal to redraw the unfinished sentence in
    previewing = redraw or on_text is not None  # ... or a GUI listening for it

    def text_of(audio: np.ndarray) -> str:
        # beam 1 and no VAD: ~20% faster than the defaults, and the gate already cut silence
        segments, _ = model.transcribe(audio, language=args.language, beam_size=1,
                                       vad_filter=False, batch_size=args.batch_size)
        return " ".join(seg.text.strip() for seg in segments).strip()

    def show(text: str, at: float, final: bool) -> None:
        if on_text:
            on_text(text, final)
        if not final:
            if redraw:
                width = shutil.get_terminal_size().columns - 1
                print(f"\r\033[K{text[-width:]}", end="", flush=True)
            return
        if typer:
            typer(text + " ")
        line = f"[{format_timestamp(at, True)}] {text}"
        print(f"\r\033[K{line}" if redraw else line, flush=True)
        if out:
            out.write(line + "\n")

    chunks: list[np.ndarray] = []  # a list, not one array: concatenating per block is O(n^2)
    buffered = 0
    silence = elapsed = next_preview = meter = 0.0
    try:
        while (raw := blocks.get()) is not None:
            audio = pcm(raw)
            elapsed += len(audio) / SAMPLE_RATE
            level = float(np.sqrt(np.mean(audio ** 2)))
            # fast attack, slow decay: a bare per-block level flickers, this reads as a meter
            meter = max(level, meter * 0.82)
            if on_level:
                on_level(meter)
            if is_on is not None and not is_on.is_set():
                chunks, buffered, silence = [], 0, 0.0  # muted: drop what was building
                continue
            quiet = level < args.threshold
            if quiet and not buffered:
                continue  # still waiting for someone to speak
            chunks.append(audio)
            buffered += len(audio)
            silence = silence + len(audio) / SAMPLE_RATE if quiet else 0.0
            start = elapsed - buffered / SAMPLE_RATE

            if silence >= args.pause or buffered >= WINDOW:
                if text := text_of(np.concatenate(chunks)):
                    show(text, start, True)
                chunks, buffered, silence, next_preview = [], 0, 0.0, 0.0
            elif previewing and elapsed >= next_preview and buffered >= SAMPLE_RATE:
                next_preview = elapsed + args.interval
                show(text_of(np.concatenate(chunks)), start, False)
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
        if buffered >= SAMPLE_RATE and (text := text_of(np.concatenate(chunks))):
            show(text, elapsed - buffered / SAMPLE_RATE, True)
        if out:
            out.close()
            print(f"-> {args.out}", file=sys.stderr)


# --- command line -----------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="*", type=Path)
    p.add_argument("--mic", action="store_true", help="live-transcribe the microphone")
    p.add_argument("-i", "--input", default="default",
                   help="mic device for --mic (list them: ffmpeg -sources pulse)")
    p.add_argument("--input-format", default=None, help="ffmpeg input format for --mic")
    p.add_argument("-o", "--out", type=Path, default=None, help="append --mic lines to a file")
    p.add_argument("--type", action="store_true",
                   help="with --mic, type each finished sentence into the focused textbox")
    p.add_argument("--interval", type=float, default=1.0,
                   help="seconds between live previews of the sentence being spoken")
    p.add_argument("--pause", type=float, default=0.8,
                   help="seconds of silence that end a sentence")
    p.add_argument("--threshold", type=float, default=0.01,
                   help="input level below this counts as silence. Raise in a noisy room")
    p.add_argument("-m", "--model", default="large-v3-turbo")
    p.add_argument("-l", "--language", default=None, help="e.g. en, ro. Default: autodetect")
    p.add_argument("-d", "--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("-c", "--compute-type", default="default",
                   help="int8, int8_float16, float16, float32")
    p.add_argument("-b", "--batch-size", type=int, default=16, help="lower if VRAM runs out")
    p.add_argument("-s", "--stream", type=int, default=None,
                   help="audio stream index to transcribe (default: 0). Listed per file")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.files and not args.mic:
        parser.error("give at least one file, or --mic")

    model = BatchedInferencePipeline(
        WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    )
    if args.mic:
        return live(model, args)
    for path in args.files:
        if path.is_file():
            transcribe_file(model, path, args)
        else:
            print(f"skip (not a file): {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
