"""Synthetic spoken test questions (Google TTS via gTTS) for the ASR evaluation.

Limitation (documented in the report): synthetic speech is cleaner than real farmer speech, so WER
here is a lower bound. Real recordings can be dropped into data/eval/audio_user/<case_id>.wav and are
picked up automatically by the ASR evaluation."""
from __future__ import annotations

from pathlib import Path

from .lang import GTTS_CODES


def synthesize_questions(cases: list[dict], out_dir: str | Path, langs=("hi", "kn", "ta")) -> dict[str, str]:
    from gtts import gTTS
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for c in cases:
        if c["lang"] not in langs:
            continue
        p = out_dir / f"{c['id']}.mp3"
        if not p.exists():
            gTTS(c["question"], lang=GTTS_CODES[c["lang"]]).save(str(p))
        paths[c["id"]] = str(p)
    return paths


def find_user_recordings(cases: list[dict], user_dir: str | Path) -> dict[str, str]:
    user_dir = Path(user_dir)
    out = {}
    if not user_dir.exists():
        return out
    for c in cases:
        for ext in (".wav", ".mp3", ".m4a", ".flac", ".ogg"):
            p = user_dir / f"{c['id']}{ext}"
            if p.exists():
                out[c["id"]] = str(p)
                break
    return out
