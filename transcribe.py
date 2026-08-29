#!/usr/bin/env python3
"""Transcribe audio/video files with faster-whisper. Any format ffmpeg reads."""
import argparse
import atexit
import ctypes
import os
import platform
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

# a windowed build has no console of its own, so each child would pop one - suppress them
NO_WINDOW = ({"creationflags": subprocess.CREATE_NO_WINDOW}
             if sys.platform == "win32" else {})
FROZEN = getattr(sys, "frozen", False)
HERE = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
# a PyInstaller build ships its own ffmpeg; a source checkout uses the one on PATH
FFMPEG = str(HERE / "ffmpeg.exe") if FROZEN and sys.platform == "win32" else "ffmpeg"

if sys.platform == "win32":
    # CUDA DLLs sit next to the exe when frozen, in the nvidia wheels when not
    for _dir in [HERE, *HERE.glob("nvidia/*/bin"),
                 *Path(sys.prefix).glob("Lib/site-packages/nvidia/*/bin")]:
        if _dir.is_dir():
            os.add_dll_directory(str(_dir))
else:
    # ctranslate2 dlopens libcublas/libcudnn by soname; pip wheels hide them in
    # site-packages/nvidia/*/lib, which the loader does not search. Preload by path,
    # cublas first (cudnn links against it).
    for _lib in sorted(HERE.glob(".venv/lib/*/site-packages/nvidia/*/lib/*.so.*")):
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
        [FFMPEG, "-nostdin", "-v", "error", "-i", str(path), "-map", f"0:a:{index}",
         "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, **NO_WINDOW,
    )
    if r.returncode or not r.stdout:
        sys.exit(f"ffmpeg failed on stream {index} of {path}: {r.stderr.decode().strip()}")
    return np.frombuffer(r.stdout, np.int16).astype(np.float32) / 32768.0


def dshow_audio_devices():
    """[(friendly name, capture name)] from ffmpeg. The capture name is dshow's own
    "Alternative name" (a PnP path): friendly names contain colons - "Wave:XLR" - and a
    colon is dshow's separator, so passing one back gives "Malformed dshow input string"."""
    listing = subprocess.run(
        [FFMPEG, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
        capture_output=True, text=True, **NO_WINDOW,
    ).stderr
    devices = []
    for line in listing.splitlines():
        line = re.sub(r"^\[[^\]]+\]\s*", "", line).strip()
        if m := re.fullmatch(r'"(.+)" \(audio\)', line):
            devices.append([m.group(1), m.group(1)])
        elif m := re.fullmatch(r'Alternative name "(.+)"', line):
            if devices and devices[-1][0] == devices[-1][1]:
                devices[-1][1] = m.group(1)
    return [tuple(d) for d in devices]


# Core Audio knows the default recording device; dshow does not. Both identify an endpoint
# by the same GUID - it is the tail of the Core Audio id and sits inside dshow's "wave_{...}".
DEFAULT_ENDPOINT_PS = """
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class DefaultCapture {
  [ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")] class Enumerator { }
  [ComImport, Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  interface IMMDeviceEnumerator {
    int EnumAudioEndpoints(int flow, int mask, out IntPtr col);
    int GetDefaultAudioEndpoint(int flow, int role, out IMMDevice dev);
  }
  [ComImport, Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
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


def windows_default_endpoint():
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


def dshow_input(wanted):
    """Resolve --input to something dshow accepts."""
    devices = dshow_audio_devices()
    if not devices:
        sys.exit("no dshow audio device found - check Windows microphone privacy settings")
    if wanted != "default":
        name = wanted[len("audio="):] if wanted.startswith("audio=") else wanted
        for friendly, capture in devices:
            if name in (friendly, capture):
                return f"audio={capture}"
        return wanted  # not one of ours: hand it to ffmpeg untouched
    for friendly, _ in devices:
        print(f"   audio device: {friendly}", file=sys.stderr)
    endpoint = windows_default_endpoint()
    match = [d for d in devices if endpoint and endpoint in d[1].lower()]
    # no endpoint match (a virtual device with no Core Audio entry): prefer a name saying mic
    friendly, capture = (match or [d for d in devices if "mic" in d[0].lower()] or devices)[0]
    print(f'   using "{friendly}" - override with -i "<name>"', file=sys.stderr)
    return f"audio={capture}"


def audio_inputs():
    """[(value for --input, label to show)] of capture devices on this platform."""
    if sys.platform == "win32":
        return [("default", "Windows default")] + \
               [(f"audio={capture}", friendly) for friendly, capture in dshow_audio_devices()]
    if sys.platform != "linux":
        return [("default", "default")]
    listing = subprocess.run([FFMPEG, "-hide_banner", "-sources", "pulse"],
                             capture_output=True, text=True)  # linux only, no flags needed
    devices = [("default", "default")]
    for line in (listing.stdout + listing.stderr).splitlines():
        if m := re.match(r"\s*\*?\s*(\S+)\s+\[(.*)\]", line):
            devices.append((m.group(1), m.group(2)))
    return devices


def make_typer():
    """Return send(text): types into whatever window has focus right now."""
    if sys.platform == "win32" or "microsoft" in platform.uname().release.lower():
        # one long-lived shell: spawning powershell.exe per utterance costs ~1s
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

        def send(text):
            # Paste, never SendKeys-typing: its brace escapes send the unshifted key, so
            # "50% (net)" arrives as "505 9net0". Pasting is also layout- and accent-proof.
            # restoring the clipboard per sentence truncates the paste - it is put back on exit
            ps.stdin.write(
                f"Set-Clipboard -Value '{quote(text)}'; "
                f"[System.Windows.Forms.SendKeys]::SendWait('^v'); "
                f"Start-Sleep -Milliseconds 150\n")

        return send

    if shutil.which("xdotool"):  # X11
        return lambda text: subprocess.run(["xdotool", "type", "--clearmodifiers", "--", text])
    if shutil.which("wtype"):  # wayland
        return lambda text: subprocess.run(["wtype", "--", text])
    sys.exit("--type needs powershell.exe (WSL/Windows), xdotool or wtype")


def quote(text):
    """Escape for a PowerShell single-quoted string."""
    return text.replace("'", "''")



# ponytail: RMS gate, not a real VAD. Cheap and tunable; if it clips speech in a noisy
# room, swap in faster_whisper.vad.get_speech_timestamps over the tail of the buffer.
def live(model, args, is_on=None, on_text=None, on_level=None, handle=None):
    """Capture the mic, preview every --interval, commit the line once the speaker pauses.

    is_on: optional threading.Event - audio is discarded while it is clear.
    on_text: optional callback, called with every committed sentence.
    on_level: optional callback, called with the RMS of every 0.25s block.
    handle: optional dict - the ffmpeg process is stored under "proc" so a caller can
        terminate it (that ends this call, e.g. to reopen on another device).
    """
    fmt = args.input_format or {"linux": "pulse", "darwin": "avfoundation",
                                "win32": "dshow"}.get(sys.platform, "pulse")
    if fmt == "dshow":
        args.input = dshow_input(args.input)
    proc = subprocess.Popen(
        [FFMPEG, "-nostdin", "-v", "error", "-f", fmt, "-i", args.input,
         "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
        stdout=subprocess.PIPE, **NO_WINDOW,
    )
    if handle is not None:
        handle["proc"] = proc
    # a thread keeps the pipe drained while the GPU is busy, else pulse overruns and drops audio
    blocks = queue.Queue()
    threading.Thread(target=lambda: ([blocks.put(b) for b in iter(
        lambda: proc.stdout.read(1600), b"")], blocks.put(None)), daemon=True).start()

    out = open(args.out, "a", buffering=1, encoding="utf-8") if args.out else None
    typer = make_typer() if args.type else None
    preview_ok = sys.stdout.isatty()
    chunks, buffered, silence, elapsed, next_preview, meter = [], 0, 0.0, 0.0, 0.0, 0.0
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
            if on_text:
                on_text(text)
            if typer:
                typer(text + " ")
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
            level = float(np.sqrt(np.mean(audio ** 2)))
            # fast attack, slow decay: a bare per-block RMS flickers, this reads as a meter
            meter = max(level, meter * 0.82)
            if on_level:
                on_level(meter)
            if is_on is not None and not is_on.is_set():
                chunks, buffered, silence = [], 0, 0.0  # muted: drop what was building
                continue
            quiet = level < args.threshold
            if quiet and not buffered:
                continue  # still waiting for someone to speak
            chunks.append(audio)  # kept as a list: concatenating per block is O(n^2)
            buffered += len(audio)
            silence = silence + len(audio) / 16000 if quiet else 0.0
            start = elapsed - buffered / 16000

            if silence >= args.pause or buffered >= cap:
                if text := text_of(np.concatenate(chunks)):
                    show(text, start, True)
                chunks, buffered, silence, next_preview = [], 0, 0.0, 0.0
            elif preview_ok and elapsed >= next_preview and buffered >= 16000:
                next_preview = elapsed + args.interval
                show(text_of(np.concatenate(chunks)), start, False)
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
        if buffered >= 16000 and (text := text_of(np.concatenate(chunks))):
            show(text, elapsed - buffered / 16000, True)
        if out:
            out.close()
            print(f"-> {args.out}", file=sys.stderr)


def build_parser():
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
    return p


def main():
    p = build_parser()
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
