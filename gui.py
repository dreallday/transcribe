#!/usr/bin/env python3
"""Floating dictation panel: mic on/off, close. Run: .venv/bin/python gui.py"""
import importlib.util
import multiprocessing
import subprocess
import sys
import threading

multiprocessing.freeze_support()  # native mode relaunches this exe for the webview process

from faster_whisper import BatchedInferencePipeline, WhisperModel
from nicegui import app, ui

import transcribe

PORT = 8171
EDGE = "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
# a real frameless window where pywebview is available (Windows builds), browser otherwise
NATIVE = importlib.util.find_spec("webview") is not None

# parse_known_args, not parse_args: the webview child is launched with --multiprocessing-fork
argv = [a for a in sys.argv[1:] if a != "--no-window"]
args, _ = transcribe.build_parser().parse_known_args(["--mic", "--type", *argv])
listening = threading.Event()
state = {"ready": False, "text": "", "level": 0.0}
capture = {}  # holds the running ffmpeg process, so a device switch can end it
inputs = {}  # filled at startup: enumerating shells out to ffmpeg, too slow to redo per render
switching = threading.Event()


def worker():
    model = BatchedInferencePipeline(
        WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    )
    state["ready"] = True
    while True:
        transcribe.live(model, args, is_on=listening, on_text=lambda t: state.update(text=t),
                        on_level=lambda v: state.update(level=v), handle=capture)
        if not switching.is_set():  # the input ended on its own
            return
        switching.clear()


def pick_input(value):
    """Reopen capture on another device without reloading the model."""
    args.input = value
    switching.set()
    if proc := capture.get("proc"):
        proc.terminate()


# an app window has no title bar to grab, so the panel drags the window itself
DRAG = """
<script>
let last = null;
addEventListener('pointerdown', e => {
  if (e.target.closest('button')) return;
  last = e; document.body.setPointerCapture(e.pointerId);
});
addEventListener('pointermove', e => {
  if (!last) return;
  try { window.moveBy(e.screenX - last.screenX, e.screenY - last.screenY); } catch (_) {}
  last = e;
});
addEventListener('pointerup', () => last = null);
</script>
"""


CSS = """
<style>
  body { background: transparent; overflow: hidden; user-select: none; }
  .shell { background: #15171c; border: 1px solid #2b303a; border-radius: 14px;
           box-shadow: 0 10px 30px rgba(0,0,0,.45); }
  .rule { height: 1px; background: #232830; }
  .said { color: #a7afbd; font-size: 11.5px; }
  .mic-live { box-shadow: 0 0 0 0 rgba(239,68,68,.65); animation: pulse 1.8s infinite; }
  @keyframes pulse {
    70% { box-shadow: 0 0 0 12px rgba(239,68,68,0); }
    100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
  }
  .meter { background: #22262e; border-radius: 999px; height: 5px; overflow: hidden; }
  .meter > div { height: 100%; border-radius: 999px; transition: width .1s linear; }
  .said { line-height: 1.4; min-height: 2.8em; max-height: 2.8em; overflow: hidden;
          display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
  .device .q-field__native { color: #9aa3b2; font-size: 11px; padding: 0; min-height: 0; }
  .device .q-field__control, .device .q-field__marginal { height: 18px; min-height: 18px; }
  .device .q-icon { font-size: 15px; color: #6b7280; }
</style>
"""


@ui.page("/")
def panel():
    ui.add_head_html(CSS)
    if not NATIVE:
        ui.add_body_html(DRAG)
    ui.query(".nicegui-content").classes("p-0 gap-0")

    def toggle():
        if state["ready"]:
            listening.clear() if listening.is_set() else listening.set()

    with ui.element("div").classes("shell w-full h-full px-4 py-3 flex flex-col gap-3 "
                                   "cursor-move pywebview-drag-region"):
        with ui.row().classes("items-center gap-3 no-wrap w-full"):
            mic = ui.button(on_click=toggle).props("round unelevated size=14px")
            with ui.column().classes("gap-1 min-w-0 grow"):
                status = ui.label().classes("text-sm font-semibold leading-none")
                device = ui.select(inputs, value=args.input,
                                   on_change=lambda e: pick_input(e.value))
                device.props("dense options-dense borderless").classes("device w-full")
            ui.button(icon="close", on_click=app.shutdown) \
                .props("flat round dense size=9px color=grey-7").classes("self-start")

        with ui.element("div").classes("meter w-full"):
            bar = ui.element("div").style("width: 0%; background: #6b7280")

        ui.element("div").classes("rule w-full")
        said = ui.label().classes("said w-full")
        ui.query("body").style("background: #15171c")

    def refresh():
        on = listening.is_set()
        mic.props(f'icon={"mic" if on else "mic_off"} color={"red-6" if on else "grey-8"}')
        mic.classes(replace="mic-live" if on else "")
        status.text = ("Listening" if on else "Paused") if state["ready"] else "Loading model"
        status.classes(replace="text-sm font-semibold leading-none "
                       + ("text-red-4" if on else "text-grey-3"))
        # meter is log-ish: speech sits around 0.02-0.2 RMS, so a linear bar barely moves
        pct = min(100, int((state["level"] ** 0.5) * 260))
        bar.style(f"width: {pct}%; background: {'#ef4444' if on else '#6b7280'}")
        said.text = state["text"] or ("Say something - it lands in the focused textbox"
                                      if on else "Press the mic to start")

    ui.timer(0.06, refresh)  # meter needs to move, not tick


@app.on_startup
def start():
    inputs.update(transcribe.audio_inputs())
    threading.Thread(target=worker, daemon=True).start()
    if not NATIVE and "--no-window" not in sys.argv:
        try:
            subprocess.Popen([EDGE, f"--app=http://localhost:{PORT}", "--window-size=380,160"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            print(f"open http://localhost:{PORT} yourself", file=sys.stderr)


# __main__ only: the spawned webview child imports this module as __mp_main__ and
# must not start a second server, model and microphone capture
if __name__ == "__main__":
    if NATIVE:
        app.native.window_args.update(frameless=True, easy_drag=True, on_top=True,
                                      background_color="#15171c")
    options = dict(host="127.0.0.1", port=PORT, show=False, reload=False,
                   title="dictate", favicon="🎤", dark=True)
    if NATIVE:  # passing window_size at all switches NiceGUI into native mode
        options.update(native=True, window_size=(400, 158))
    ui.run(**options)
