"""Exercise real settings validation with isolated in-memory video defaults."""
from copy import deepcopy
from pathlib import Path
from video_settings import VideoSettingsStore


def store_with_defaults(entries):
    store = VideoSettingsStore.__new__(VideoSettingsStore)
    store.path = Path("unused-test-settings.json")
    store.defaults = deepcopy(entries)
    store.data = {"version": 2, "profiles": {}}
    return store
