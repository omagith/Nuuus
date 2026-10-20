"""Persistent user settings in data/settings.json.

Flat dotted keys ("instant.stt_tier") in a flat JSON object, so the file
stays human-editable. Every ``set`` saves immediately via an atomic
replace — the app can be killed at any moment without corrupting it. A
corrupt file is moved aside to settings.json.bak and defaults are used.
"""
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Optional

from app.core.paths import SETTINGS_FILE

log = logging.getLogger(__name__)

DEFAULTS = {
    "audio.input_device": "",  # device name; "" = system default
    "ui.theme": "system",  # "system" | "light" | "dark"
    "ui.mica": False,
    "ui.geometry": "",  # base64 of QWidget.saveGeometry()
    "ui.sidebar_collapsed": False,
    "language": "ru",  # "ru" | "multi"
    "ollama.host": "http://localhost:11434",
    # «Поток». Tiers: STT "light" | "power"; LLM "light" | "power" | "ultra".
    # Device: "gpu" = CUDA when available (else CPU, silently); "cpu" also
    # makes Ollama keep the LLM off the GPU (num_gpu 0).
    "instant.stt_tier": "power",
    "instant.llm_tier": "power",
    "instant.device": "gpu",
    "instant.topic": "",
    "instant.only_questions": True,
    # Photos in «Поток»: "question" = attached to the next typed question,
    # "context" = read once and used by every following answer.
    "instant.photo_mode": "question",
    "instant.mic_meter": True,  # level meter visible (hidden = not computed/painted)
    # «Полная запись»
    "lecture.stt_tier": "power",
    "lecture.llm_tier": "power",
    "lecture.device": "gpu",
    "lecture.hf_token": "",  # used only when the HF_TOKEN env var is empty
    "lecture.denoise": False,
    "lecture.dereverb": False,
    "lecture.live_preview": True,  # live transcription panel while recording
    "lecture.mic_meter": True,
    # Photos in the notes: "timestamps" = after the notes of the part of the
    # lecture they were taken in; "end" = a «Фото с занятия» section at the end.
    "lecture.photo_placement": "timestamps",
    # Photo-reading model: "cpu" (default — leaves the 4 GB card to «Поток») | "gpu".
    "vision.device": "cpu",
    # Heavy work (with the embedded Ollama) may take at most this share of the
    # CPU and RAM, 50–100 %; 100 = no limits (app.core.resources).
    "system.load_limit": 95,
}

_MISSING = object()


class Settings:
    def __init__(self, path: Path = SETTINGS_FILE):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._values = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("settings root is not an object")
            self._values = data
        except (OSError, ValueError) as exc:
            log.warning("Settings file %s is unreadable (%s); using defaults", self.path, exc)
            try:
                os.replace(self.path, self.path.with_name(self.path.name + ".bak"))
            except OSError:
                pass
            self._values = {}

    def get(self, key: str, default: Any = _MISSING) -> Any:
        with self._lock:
            if key in self._values:
                return self._values[key]
        if default is not _MISSING:
            return default
        return DEFAULTS.get(key)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if self._values.get(key, _MISSING) == value:
                return
            self._values[key] = value
            self.save()

    def save(self) -> None:
        with self._lock:
            data = dict(self._values)
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        except OSError:
            log.exception("Could not save settings to %s", self.path)


_instance: Optional[Settings] = None
_instance_lock = threading.Lock()


def get_settings() -> Settings:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = Settings()
        return _instance
