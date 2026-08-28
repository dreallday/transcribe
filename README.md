# transcribe

Meeting transcription with [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

## Setup

**1. [uv](https://github.com/astral-sh/uv)** - the Python installer used below
([install docs](https://docs.astral.sh/uv/getting-started/installation/)).
Skip if `uv --version` already answers:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # Linux/macOS/WSL
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows
```

Restart the shell afterwards so `~/.local/bin` lands on PATH. uv brings its own Python, so
nothing else is needed. Prefer stock tooling? Skip this step - step 3 has a `venv` + `pip`
line that works the same.

**2. ffmpeg** - needed on PATH to pull a chosen audio track out of a container:

```bash
sudo apt install ffmpeg      # Debian/Ubuntu/WSL
brew install ffmpeg          # macOS
winget install ffmpeg        # Windows
ffmpeg -version              # check
```

**3. faster-whisper and the rest:**

```bash
uv venv && uv pip install -r requirements.txt   # or: python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

On a CPU-only machine, delete the two `nvidia-*` lines from `requirements.txt` first - they
are ~700 MB of CUDA libraries and do nothing without a GPU. Then run with `-d cpu -c int8`.

No model download step: the weights land in `~/.cache/huggingface` on first use (~1.6 GB for
`large-v3-turbo`).

## Audio

wav, mp3, m4a, flac, ogg, opus - anything ffmpeg reads.

```bash
.venv/bin/python transcribe.py meeting.wav                    # autodetect device + language
.venv/bin/python transcribe.py -d cuda -c float16 meeting.wav # GPU, fastest
.venv/bin/python transcribe.py -l en *.mp3                    # force language, batch of files
```

Writes `meeting.wav.txt` next to the input: `[HH:MM:SS] text` paragraphs, one per ~30s chunk.
Same lines stream to stderr while it runs.

## Video

mp4, mkv, mov, webm. Same command, no extra step - the audio track is pulled straight out
of the container.

```bash
.venv/bin/python transcribe.py -d cuda -c float16 standup.mp4   # -> standup.mp4.txt
```

## Multiple audio tracks

A recording with a separate mic and system track lists its streams first and marks the
selected one:

```
 * stream 0: aac eng Mic
   stream 1: aac eng System
```

Pick one with `-s`, run it twice to keep both - each track gets its own file:

```bash
.venv/bin/python transcribe.py call.mkv        # -> call.mkv.a0.txt
.venv/bin/python transcribe.py -s 1 call.mkv   # -> call.mkv.a1.txt
```

`no speech found - wrong stream?` means that track was silent - try the other index.

## Flags

| flag | default | notes |
|---|---|---|
| `-m` | `large-v3-turbo` | also `tiny`, `base`, `small`, `medium`, `large-v3` |
| `-d` | `auto` | `cpu`, `cuda` |
| `-c` | `default` | `int8` (CPU), `float16` (GPU) |
| `-l` | autodetect | `en`, `ro`, ... |
| `-b` | `16` | batch size, lower if VRAM runs out |
| `-s` | `0` | audio stream index, for multi-track files |

Speed: 30 min of audio in ~15s on an RTX 4070 Ti SUPER (`large-v3-turbo`, float16).

## GPU not used?

`RuntimeError: Library libcublas.so.12 is not found` means the `nvidia-*` wheels are missing.
Install them; `transcribe.py` preloads the libs out of `site-packages`, so no `LD_LIBRARY_PATH` needed.
