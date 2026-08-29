#!/usr/bin/env python3
"""Morbo, the dictation panel: mic on/off, input picker, live text. Run: python morbo.py"""
import argparse
import importlib.util
import multiprocessing
import threading
from dataclasses import dataclass, field

multiprocessing.freeze_support()  # native mode relaunches this program for the webview

from faster_whisper import BatchedInferencePipeline, WhisperModel  # noqa: E402
from faster_whisper.utils import available_models  # noqa: E402
from nicegui import app, ui  # noqa: E402

import transcribe  # noqa: E402

PORT = 8171
# what the browser calls a key, in the words parse_hotkey understands
KEY_NAMES = {" ": "space", "Escape": "esc", "ArrowLeft": "left", "ArrowRight": "right",
             "ArrowUp": "up", "ArrowDown": "down", "PageUp": "pageup", "PageDown": "pagedown",
             "Enter": "enter", "Tab": "tab", "Backspace": "backspace", "Delete": "delete",
             "Insert": "insert", "Home": "home", "End": "end"}
SIZE = (400, 158)          # the panel alone
SIZE_SETTINGS = (400, 560)  # ... and with the settings open
# a real frameless window where pywebview is installed (the Windows build), browser otherwise
NATIVE = importlib.util.find_spec("webview") is not None

# the panel's own options; everything else is handed to transcribe.py's parser. Both use
# parse_known_args because the webview child is launched with --multiprocessing-fork
panel_parser = argparse.ArgumentParser(add_help=False)
panel_parser.add_argument("--hotkey", default="ctrl+alt+m",
                          help='toggle the mic from any window. "none" disables it')
panel_parser.add_argument("--no-type", action="store_true", help="show text, type nothing")
panel_parser.add_argument("--start-listening", action="store_true", help="mic on at launch")
panel, rest = panel_parser.parse_known_args()
engine_parser = transcribe.build_parser()
args, _ = engine_parser.parse_known_args(
    ["--mic", *([] if panel.no_type else ["--type"]), *rest])

# settings the cog remembers between runs. Add a name here and it is saved and restored;
# a flag given on the command line still wins for that run.
SAVED = ("hotkey", "input", "model", "device", "compute_type", "type",
         "pause", "interval", "threshold")
_saved = transcribe.load_settings()
transcribe.apply_saved(panel, panel_parser, _saved)
transcribe.apply_saved(args, engine_parser, _saved)


def remember() -> None:
    """Write the current settings out, so the next launch starts where this one left off."""
    values = {key: getattr(args, key) for key in SAVED if hasattr(args, key)}
    values.update({key: getattr(panel, key) for key in SAVED if hasattr(panel, key)})
    transcribe.save_settings(values)


@dataclass
class State:
    """What the panel shows. The worker thread writes it, the UI timer reads it."""
    ready: bool = False       # model loaded
    text: str = ""            # last committed sentence
    partial: str = ""         # sentence still being spoken
    level: float = 0.0        # input level, 0-1
    recording: bool = False   # waiting for the user to press a new hotkey
    inputs: dict = field(default_factory=dict)  # device value -> label
    capture: dict = field(default_factory=dict)  # holds the running ffmpeg process
    hotkey_error: str = ""    # why the hotkey did not bind, if it did not


state = State()
listening = threading.Event()
switching = threading.Event()   # reopen the capture
reloading = threading.Event()   # ... and load a different model first
unbind_hotkey = None  # set once the global hotkey is registered


def worker() -> None:
    """Load the model and keep capturing, reloading whenever the model is changed."""
    model = None
    while True:
        state.ready = False
        switching.clear()
        model = None  # drop the old model first, so its memory is free for the next one
        model = BatchedInferencePipeline(
            WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
        )
        state.ready = True
        reloading.clear()
        while True:
            transcribe.live(model, args, is_on=listening, handle=state.capture,
                            on_text=on_text, on_level=lambda v: setattr(state, "level", v))
            if reloading.is_set():
                break  # load the newly chosen model
            if not switching.is_set():
                return  # the input ended on its own
            switching.clear()


def toggle_mic() -> None:
    """Start or stop transcribing. Driven by the mic button and by the hotkey."""
    if state.ready:
        listening.clear() if listening.is_set() else listening.set()


def on_text(text: str, final: bool) -> None:
    """Previews arrive every --interval; the committed sentence replaces them."""
    state.text, state.partial = (text, "") if final else (state.text, text)


def restart_capture() -> None:
    """End the current ffmpeg so the worker reopens it - the model stays loaded."""
    switching.set()
    if proc := state.capture.get("proc"):
        proc.terminate()


def pick_input(value: str) -> None:
    args.input = value
    remember()
    restart_capture()


def set_model(name: str) -> None:
    """Load a different model. The first use of one downloads it, which takes a while."""
    if name == args.model:
        return
    args.model = name
    remember()
    reloading.set()
    restart_capture()


def set_hotkey(spec: str, save: bool = True) -> None:
    """Rebind the global hotkey while running. save=False for the initial bind, so simply
    launching with a flag does not write it into the saved settings."""
    global unbind_hotkey
    spec = spec.strip()
    off = spec.lower() in ("none", "off", "")
    if not off:
        try:  # a spec that does not parse keeps the current binding instead of losing it
            transcribe.parse_hotkey(spec)
        except ValueError as e:
            state.hotkey_error = str(e)
            return
    if unbind_hotkey:
        unbind_hotkey()
        unbind_hotkey = None
    panel.hotkey = spec
    state.hotkey_error = ""
    if save:
        remember()
    if off:
        return
    unbind_hotkey = transcribe.watch_hotkey(
        spec, toggle_mic,
        on_status=lambda ok, msg: setattr(state, "hotkey_error", "" if ok else msg))


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


def resize(size) -> None:
    """Grow the window for the settings menu - it is taller than the panel itself."""
    if NATIVE and (window := app.native.main_window):
        window.resize(*size)


def model_options() -> list[tuple[str, str]]:
    """Every model faster-whisper knows, with the size of the ones already downloaded."""
    cached = transcribe.cached_models()
    names = sorted(set(available_models()) | set(cached))
    return [(name, f"{name}  ({transcribe.human_size(cached[name])})" if name in cached else name)
            for name in names]


def remove_model(name: str) -> None:
    """Delete a downloaded model, after asking - it is a big file to fetch again."""
    size = transcribe.human_size(transcribe.cached_models().get(name, 0))
    with ui.dialog() as confirm, ui.card().classes("bg-[#1a1d23] p-3 gap-2"):
        ui.label(f"Delete {name}?").classes("text-sm font-semibold")
        ui.label(f"Frees {size}. Choosing it again downloads it again.") \
            .classes("text-xs text-grey-5")
        with ui.row().classes("gap-2 justify-end w-full"):
            ui.button("Cancel", on_click=confirm.close).props("flat dense size=sm color=grey")
            ui.button("Delete", on_click=lambda: (transcribe.delete_cached_model(name),
                                                  confirm.close(), downloaded.refresh())) \
                .props("flat dense size=sm color=red")
    confirm.open()


@ui.refreshable
def downloaded() -> None:
    """The models on disk, with a way to throw away the ones you no longer use."""
    cached = transcribe.cached_models()
    if not cached:
        ui.label("no models downloaded yet").classes("text-[10px] text-grey-6")
        return
    total = transcribe.human_size(sum(cached.values()))
    ui.label(f"DOWNLOADED - {total}").classes("text-[10px] tracking-widest text-grey-6")
    for name, size in sorted(cached.items()):
        with ui.row().classes("items-center gap-2 no-wrap w-full"):
            in_use = name == args.model
            ui.label(name + (" (in use)" if in_use else "")) \
                .classes("text-[11px] grow " + ("text-grey-3" if in_use else "text-grey-5"))
            ui.label(transcribe.human_size(size)).classes("text-[10px] text-grey-6")
            if not in_use:
                ui.button(icon="delete_outline", on_click=lambda n=name: remove_model(n)) \
                    .props("flat round dense size=8px color=grey-7")


def settings() -> None:
    """The cog: everything you set once and forget."""
    with ui.button(icon="settings").props("flat round dense size=9px color=grey-7"):
        menu = ui.menu().props("dark").classes("bg-[#1a1d23] p-3 w-80 max-h-[520px]")
        # the menu is taller than the panel, so the window has to make room for it
        menu.on("show", lambda: (downloaded.refresh(), resize(SIZE_SETTINGS)))
        menu.on("hide", lambda: resize(SIZE))
        with menu, ui.column().classes("gap-2 w-full"):
            ui.label("SETTINGS").classes("text-[10px] tracking-widest text-grey-6")

            hotkey_field = ui.input("Global hotkey", value=panel.hotkey,
                                    on_change=lambda e: apply_hotkey(e.value)) \
                .props("dense outlined debounce=700").classes("w-full text-xs") \
                .tooltip('ctrl / alt / shift / win plus a key, or a mouse button: '
                         'm3 middle, m4 back, m5 forward. "none" turns it off')

            def apply_hotkey(spec: str) -> None:
                set_hotkey(spec)
                hotkey_field.value = panel.hotkey  # a rejected spec snaps back

            def recorded(event) -> None:
                """The next key pressed becomes the hotkey."""
                if not state.recording or not event.action.keydown:
                    return
                name = event.key.name
                if name in ("Control", "Alt", "Shift", "Meta"):
                    return  # a modifier on its own is not a hotkey - wait for the real key
                mods = [word for word, held in (("ctrl", event.modifiers.ctrl),
                                                ("alt", event.modifiers.alt),
                                                ("shift", event.modifiers.shift),
                                                ("win", event.modifiers.meta)) if held]
                stop_recording()
                apply_hotkey("+".join([*mods, KEY_NAMES.get(name, name.lower())]))

            keyboard = ui.keyboard(on_key=recorded, active=False)

            def stop_recording() -> None:
                state.recording = False
                keyboard.active = False

            def start_recording() -> None:
                state.recording = True
                keyboard.active = True

            with ui.row().classes("items-center gap-1 no-wrap w-full"):
                ui.button("press a key", icon="keyboard", on_click=start_recording) \
                    .props("flat dense size=sm color=grey-5").classes("text-[10px]") \
                    .bind_text_from(state, "recording",
                                    lambda on: "waiting..." if on else "press a key")
                ui.space()
                for button, tip in (("m3", "middle"), ("m4", "back"), ("m5", "forward")):
                    ui.button(button, on_click=lambda b=button: apply_hotkey(b)) \
                        .props("flat dense size=sm color=grey-5").classes("text-[10px]") \
                        .tooltip(f"{tip} mouse button")
                ui.button("none", on_click=lambda: apply_hotkey("none")) \
                    .props("flat dense size=sm color=grey-5").classes("text-[10px]")
            ui.label().bind_text_from(state, "hotkey_error").classes("text-[10px] text-red-4")

            ui.switch("Type into the focused window", value=args.type,
                      on_change=lambda e: (setattr(args, "type", e.value), remember(),
                                           restart_capture())) \
                .props("dense").classes("text-xs")

            with ui.row().classes("gap-2 no-wrap w-full"):
                ui.number("Pause (s)", value=args.pause, step=0.1, min=0.2, max=5,
                          on_change=lambda e: (setattr(args, "pause", e.value or 0.8),
                                       remember())) \
                    .props("dense outlined").classes("text-xs grow")
                ui.number("Preview (s)", value=args.interval, step=0.5, min=0.2, max=10,
                          on_change=lambda e: (setattr(args, "interval", e.value or 1.0),
                                       remember())) \
                    .props("dense outlined").classes("text-xs grow")
                ui.number("Silence", value=args.threshold, step=0.005, min=0, max=0.5,
                          format="%.3f",
                          on_change=lambda e: (setattr(args, "threshold", e.value or 0.01),
                                       remember())) \
                    .props("dense outlined").classes("text-xs grow")

            ui.select({name: label for name, label in model_options()},
                      value=args.model, label="Model", with_input=True,
                      on_change=lambda e: set_model(e.value)) \
                .props("dense options-dense outlined").classes("w-full text-xs") \
                .tooltip("a model you have not used before is downloaded on first use")
            downloaded()
            ui.label(f"{args.device} ({args.compute_type})") \
                .classes("text-[10px] text-grey-6")


@ui.page("/")
def show() -> None:
    ui.add_head_html(CSS)
    ui.query(".nicegui-content").classes("p-0 gap-0 h-full")

    with ui.element("div").classes("shell w-full h-full px-4 py-3 flex flex-col gap-3 "
                                   "cursor-move pywebview-drag-region"):
        with ui.row().classes("items-center gap-3 no-wrap w-full"):
            mic = ui.button(on_click=toggle_mic).props("round unelevated size=14px")
            with ui.column().classes("gap-1 min-w-0 grow"):
                status = ui.label().classes("text-sm font-semibold leading-none")
                ui.select(state.inputs, value=args.input,
                          on_change=lambda e: pick_input(e.value)) \
                    .props("dense options-dense borderless").classes("device w-full")
            with ui.row().classes("gap-0 no-wrap self-start"):
                settings()
                ui.button(icon="close", on_click=app.shutdown) \
                    .props("flat round dense size=9px color=grey-7")

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
        hint = "Press the mic to start"
        if panel.hotkey.lower() not in ("none", "off") and not state.hotkey_error:
            hint += f" - or {panel.hotkey} from any window"
        said.text = state.partial or state.text or (
            "Say something - it lands in the focused textbox" if on else hint)
        said.classes(replace="said w-full" + (" partial" if state.partial else ""))

    ui.timer(0.06, refresh)  # the meter has to move, not tick


@app.on_startup
def start() -> None:
    state.inputs.update(transcribe.audio_inputs())
    state.inputs.setdefault(args.input, args.input)  # a -i the enumeration does not know
    if panel.start_listening:
        listening.set()
    set_hotkey(panel.hotkey, save=False)
    threading.Thread(target=worker, daemon=True).start()


# __main__ only: the spawned webview child imports this module as __mp_main__ and must not
# start a second server, model and microphone capture
if __name__ == "__main__":
    options = dict(host="127.0.0.1", port=PORT, show=False, reload=False,
                   title="Morbo", favicon="🎤", dark=True)
    if NATIVE:
        app.native.window_args.update(frameless=True, easy_drag=True, on_top=True,
                                      background_color="#15171c")
        # passing window_size at all is what switches NiceGUI into native mode
        options.update(native=True, window_size=SIZE)
    ui.run(**options)
