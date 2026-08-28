# transcribe

Meeting transcription with [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
Audio or video, any format ffmpeg reads: wav, mp3, m4a, flac, mp4, mkv, mov, webm.
Video needs no extra step - the audio track is pulled straight out of the container.

## Setup

```bash
uv venv && uv pip install faster-whisper nvidia-cublas-cu12 nvidia-cudnn-cu12
```

Needs `ffmpeg` on PATH. The two `nvidia-*` packages are GPU-only; skip on a CPU-only box.

## Use

```bash
.venv/bin/python transcribe.py meeting.wav                    # autodetect device + language
.venv/bin/python transcribe.py -d cuda -c float16 meeting.wav # GPU, fastest
.venv/bin/python transcribe.py -l en *.mp4                    # force language, batch of files
```

Writes `meeting.wav.txt` next to the input: `[HH:MM:SS] text` paragraphs, one per ~30s chunk.
Same lines stream to stderr while it runs.

| flag | default | notes |
|---|---|---|
| `-m` | `large-v3-turbo` | also `tiny`, `base`, `small`, `medium`, `large-v3` |
| `-d` | `auto` | `cpu`, `cuda` |
| `-c` | `default` | `int8` (CPU), `float16` (GPU) |
| `-l` | autodetect | `en`, `ro`, ... |
| `-b` | `16` | batch size, lower if VRAM runs out |

Models download to `~/.cache/huggingface` on first use.

Speed: 30 min of audio in ~15s on an RTX 4070 Ti SUPER (`large-v3-turbo`, float16).

## GPU not used?

`RuntimeError: Library libcublas.so.12 is not found` means the `nvidia-*` wheels are missing.
Install them; `transcribe.py` preloads the libs out of `site-packages`, so no `LD_LIBRARY_PATH` needed.
