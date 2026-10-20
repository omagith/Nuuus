"""Shared test setup: importable project root, headless Qt."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Never let a test (or a subprocess it spawns) touch the user's real settings.
os.environ["OTA_SETTINGS_FILE"] = str(Path(tempfile.mkdtemp(prefix="ota-settings-")) / "settings.json")


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    from app.core import settings as settings_module

    monkeypatch.setattr(settings_module, "_instance", settings_module.Settings(tmp_path / "settings.json"))
