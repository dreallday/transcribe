"""Checks for the fiddly parsing. Run: .venv/bin/python test_transcribe.py"""
import argparse
import os
import tempfile
from pathlib import Path

from transcribe import (
    Hotkey,
    apply_saved,
    cached_models,
    delete_cached_model,
    human_size,
    load_settings,
    parse_dshow_devices,
    parse_hotkey,
    parse_pulse_sources,
    quote,
    save_settings,
)

DSHOW = r'''[in#0 @ 0001] "Mic In (2- Elgato Wave:XLR)" (audio)
[in#0 @ 0001]   Alternative name "@device_cm_{33D9A762}\wave_{505FE611}"
[in#0 @ 0001] "Chat Mix (Elgato Virtual Audio)" (audio)
[in#0 @ 0001]   Alternative name "@device_cm_{33D9A762}\wave_{93FB1807}"
[in#0 @ 0001] "Some Webcam" (video)
[in#0 @ 0001] "Legacy Device" (audio)
'''

devices = parse_dshow_devices(DSHOW)
assert [d[0] for d in devices] == ["Mic In (2- Elgato Wave:XLR)",
                                   "Chat Mix (Elgato Virtual Audio)",
                                   "Legacy Device"], devices
# the capture name is the alternative name, because a friendly name with a colon in it
# makes ffmpeg answer "Malformed dshow input string"
assert devices[0][1] == r"@device_cm_{33D9A762}\wave_{505FE611}"
assert devices[1][1] == r"@device_cm_{33D9A762}\wave_{93FB1807}"
assert devices[2][1] == "Legacy Device"  # no alternative name: fall back to the friendly one

PULSE = """Auto-detected sources for pulse:
  RDPSink.monitor [Monitor of RDP Sink] (none)
* RDPSource [RDP Source] (none)
"""
assert parse_pulse_sources(PULSE) == [("RDPSink.monitor", "Monitor of RDP Sink"),
                                      ("RDPSource", "RDP Source")]

# hotkey specs: modifier flags for RegisterHotKey, then the key or the mouse button
assert parse_hotkey("ctrl+alt+space") == Hotkey(0x2 | 0x1, key=0x20)
assert parse_hotkey("Ctrl+Shift+F9") == Hotkey(0x2 | 0x4, key=0x78)
assert parse_hotkey("win+d") == Hotkey(0x8, key=ord("D"))
# mouse buttons, which take the low-level hook path instead of RegisterHotKey
assert parse_hotkey("m4") == Hotkey(0, button=4)
assert parse_hotkey("ctrl+m3") == Hotkey(0x2, button=3)
assert parse_hotkey("forward") == Hotkey(0, button=5)
for bad in ("ctrl+nope", "ctrl+shift", "m1", "m2", "ctrl+m4+space"):
    try:
        parse_hotkey(bad)
        raise AssertionError(f"{bad!r} should not parse")
    except ValueError:
        pass

# saved settings round-trip, and the rule that a flag typed on the command line wins
os.environ["MORBO_SETTINGS"] = str(Path(tempfile.mkdtemp()) / "settings.json")
assert load_settings() == {}, "no file yet means no settings"
save_settings({"hotkey": "ctrl+alt+k", "pause": 1.5})
assert load_settings() == {"hotkey": "ctrl+alt+k", "pause": 1.5}

parser = argparse.ArgumentParser()
parser.add_argument("--hotkey", default="ctrl+alt+m")
parser.add_argument("--pause", type=float, default=0.8)
args = parser.parse_args(["--pause", "2.5"])          # pause typed, hotkey not
apply_saved(args, parser, load_settings())
assert args.hotkey == "ctrl+alt+k", "untouched option takes the saved value"
assert args.pause == 2.5, "an explicit flag must survive"

# the model cache: what is downloaded, how big, and deleting one
cache = Path(tempfile.mkdtemp())
os.environ["HF_HUB_CACHE"] = str(cache)
assert cached_models() == {}, "empty cache means no models"
for repo, size in (("models--Systran--faster-whisper-tiny", 1024),
                   ("models--mobiuslabsgmbh--faster-whisper-large-v3-turbo", 4096)):
    blobs = cache / repo / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "model.bin").write_bytes(b"x" * size)
(cache / "models--NeoQuasar--Kronos-small").mkdir()  # not a whisper model, must be ignored

assert cached_models() == {"tiny": 1024, "large-v3-turbo": 4096}
delete_cached_model("tiny")
assert cached_models() == {"large-v3-turbo": 4096}, "deleting removes just that model"

assert human_size(512) == "512 B"
assert human_size(1536) == "2 KB"
assert human_size(3 * 1024**3) == "3.0 GB"

assert quote("don't") == "don''t"
assert quote("plain 50% (net)") == "plain 50% (net)"

print("ok")
