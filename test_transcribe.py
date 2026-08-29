"""Checks for the fiddly parsing. Run: .venv/bin/python test_transcribe.py"""
from transcribe import parse_dshow_devices, parse_pulse_sources, quote

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

assert quote("don't") == "don''t"
assert quote("plain 50% (net)") == "plain 50% (net)"

print("ok")
