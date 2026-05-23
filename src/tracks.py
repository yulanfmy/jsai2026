"""Track database management.

Loads the built-in track library from src/data/tracks.json and provides
lookup and filtering utilities.
"""

import json
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent / "data"
_TRACKS_FILE = _DATA_DIR / "tracks.json"

_tracks_cache: list[dict] | None = None


def load_tracks() -> list[dict]:
    """Load tracks from the JSON database (cached after first load)."""
    global _tracks_cache
    if _tracks_cache is not None:
        return _tracks_cache

    if not _TRACKS_FILE.exists():
        return []

    with open(_TRACKS_FILE) as f:
        _tracks_cache = json.load(f)
    return _tracks_cache


def save_tracks(tracks: list[dict]) -> None:
    """Persist tracks to the JSON database and invalidate cache."""
    global _tracks_cache
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_TRACKS_FILE, "w") as f:
        json.dump(tracks, f, indent=2, ensure_ascii=False)
    _tracks_cache = tracks


def invalidate_cache() -> None:
    global _tracks_cache
    _tracks_cache = None


def get_track_count() -> int:
    return len(load_tracks())


def tracks_have_features() -> bool:
    """Check whether the loaded tracks have LLM-estimated features."""
    tracks = load_tracks()
    if not tracks:
        return False
    return "energy" in tracks[0]


def get_all_vibes() -> list[str]:
    """Return sorted unique vibe labels across all tracks."""
    vibes = set()
    for t in load_tracks():
        v = t.get("vibe", "")
        if v:
            vibes.add(v.lower())
    return sorted(vibes)
