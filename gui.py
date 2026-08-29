#!/usr/bin/env python3
"""Floating dictation panel: mic on/off, input picker, live text. Run: python gui.py"""
import importlib.util
import multiprocessing
import sys
import threading
from dataclasses import dataclass, field

multiprocessing.freeze_support()  # native mode relaunches this program for the webview

from faster_whisper import BatchedInferencePipeline, WhisperModel  # noqa: E402
from nicegui import app, ui  # noqa: E402

import transcribe  # noqa: E402

PORT = 8171
# a real frameless window where pywebview is installed (the Windows build), browser otherwise
NATIVE = importlib.util.find_spec("webview") is not None
FLAGS = {"--no-type", "--start-listening"}  # ours, everything else belongs to transcribe.py

# parse_known_args, not parse_args: the webview child is launched with --multiprocessing-fork
forced = ["--mic"] + ([] if "--no-type" in sys.argv else ["--type"])
args, _ = transcribe.build_parser().parse_known_args(
    [*forced, *(a for a in sys.argv[1:] if a not in FLAGS)])


@dataclass
class State:
    """What the panel shows. The worker thread writes it, the UI timer reads it."""
    ready: bool = False       # model loaded
    text: str = ""            # last committed sentence
    partial: str = ""         # sentence still being spoken
    level: float = 0.0        # input level, 0-1
    inputs: dict = field(default_factory=dict)  # device value -> label
    capture: dict = field(default_factory=dict)  # holds the running ffmpeg process


state = State()
listening = threading.Event()
switching = threading.Event()


def worker() -> None:
    """Load the model once, then keep capturing until the panel closes."""
    state.ready = False
    model = BatchedInferencePipeline(
        WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    )
    state.ready = True
    while True:
        transcribe.live(model, args, is_on=listening, handle=state.capture,
                        on_text=on_text, on_level=lambda v: setattr(state, "level", v))
        if not switching.is_set():
            return  # the input ended on its own
        switching.clear()


def on_text(text: str, final: bool) -> None:
    """Previews arrive every --interval; the committed sentence replaces them."""
    state.text, state.partial = (text, "") if final else (state.text, text)


def pick_input(value: str) -> None:
    """Reopen capture on another device, without reloading the model."""
    args.input = value
    switching.set()
    if proc := state.capture.get("proc"):
        proc.terminate()


CSS = """
<style>
  body { background: #15171c; overflow: hidden; user-select: none; }
  .shell { background: #15171c; border: 1px solid #2b303a; border-radius: 14px;
           box-shadow: 0 10px 30px rgba(0,0,0,.45); }
  .rule { height: 1px; background: #232830; }
  .said { color: #a7afbd; font-size: 11.5px; line-height: 1.4; overflow: hidden;
          min-height: 2.8em; max-height: 2.8em;
          display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
  .partial { color: #7e8797; font-style: italic; }
  .mic-live { box-shadow: 0 0 0 0 rgba(239,68,68,.65); animation: pulse 1.8s infinite; }
  @keyframes pulse {
    70% { box-shadow: 0 0 0 12px rgba(239,68,68,0); }
    100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
  }
  .meter { background: #22262e; border-radius: 999px; height: 5px; overflow: hidden; }
  .meter > div { height: 100%; border-radius: 999px; transition: width .1s linear; }
  .device .q-field__native { color: #9aa3b2; font-size: 11px; padding: 0; min-height: 0; }
  .device .q-field__control, .device .q-field__marginal { height: 18px; min-height: 18px; }
  .device .q-icon { font-size: 15px; color: #6b7280; }
</style>
"""


@ui.page("/")
def panel() -> None:
    ui.add_head_html(CSS)
    ui.query(".nicegui-content").classes("p-0 gap-0")

    def toggle() -> None:
        if state.ready:
            listening.clear() if listening.is_set() else listening.set()

    with ui.element("div").classes("shell w-full h-full px-4 py-3 flex flex-col gap-3 "
                                   "cursor-move pywebview-drag-region"):
        with ui.row().classes("items-center gap-3 no-wrap w-full"):
            mic = ui.button(on_click=toggle).props("round unelevated size=14px")
            with ui.column().classes("gap-1 min-w-0 grow"):
                status = ui.label().classes("text-sm font-semibold leading-none")
                ui.select(state.inputs, value=args.input,
                          on_change=lambda e: pick_input(e.value)) \
                    .props("dense options-dense borderless").classes("device w-full")
            ui.button(icon="close", on_click=app.shutdown) \
                .props("flat round dense size=9px color=grey-7").classes("self-start")

        with ui.element("div").classes("meter w-full"):
            bar = ui.element("div").style("width: 0%")
        ui.element("div").classes("rule w-full")
        said = ui.label().classes("said w-full")

    def refresh() -> None:
        on = listening.is_set()
        mic.props(f'icon={"mic" if on else "mic_off"} color={"red-6" if on else "grey-8"}')
        mic.classes(replace="mic-live" if on else "")
        status.text = ("Listening" if on else "Paused") if state.ready else "Loading model"
        status.classes(replace="text-sm font-semibold leading-none "
                       + ("text-red-4" if on else "text-grey-3"))
        # levels are log-ish - speech sits around 0.02-0.2, so a linear bar barely moves
        bar.style(f"width: {min(100, int(state.level ** 0.5 * 260))}%; "
                  f"background: {'#ef4444' if on else '#6b7280'}")
        said.text = state.partial or state.text or (
            "Say something - it lands in the focused textbox" if on else "Press the mic to start")
        said.classes(replace="said w-full" + (" partial" if state.partial else ""))

    ui.timer(0.06, refresh)  # the meter has to move, not tick


@app.on_startup
def start() -> None:
    state.inputs.update(transcribe.audio_inputs())
    state.inputs.setdefault(args.input, args.input)  # a -i the enumeration does not know
    if "--start-listening" in sys.argv:
        listening.set()
    threading.Thread(target=worker, daemon=True).start()


# __main__ only: the spawned webview child imports this module as __mp_main__ and must not
# start a second server, model and microphone capture
if __name__ == "__main__":
    options = dict(host="127.0.0.1", port=PORT, show=False, reload=False,
                   title="dictate", favicon="🎤", dark=True)
    if NATIVE:
        app.native.window_args.update(frameless=True, easy_drag=True, on_top=True,
                                      background_color="#15171c")
        # passing window_size at all is what switches NiceGUI into native mode
        options.update(native=True, window_size=(400, 158))
    ui.run(**options)
