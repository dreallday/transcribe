#!/usr/bin/env bash
# Dictation: speak, and each finished sentence is typed into whatever textbox has focus.
# Ctrl-C to stop. Extra flags are passed through, e.g. ./dictate.sh -l ro -m small
cd "$(dirname "$0")" || exit 1
exec .venv/bin/python transcribe.py --mic --type -d cuda -c float16 "$@"
