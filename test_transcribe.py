"""Run: .venv/bin/python test_transcribe.py"""
from transcribe import quote

assert quote("don't") == "don''t"
assert quote("plain text 50%") == "plain text 50%"
assert quote("''") == "''''"
print("ok")
