"""Track database management.

Provides per-user track storage so each Spotify user only sees their
own library.  Tracks are stored at ``src/data/users/{user_id}/tracks.json``.
"""

import json
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent / "data"
_USERS_DIR = _DATA_DIR / "users"

_tracks_cache: dict[str, list[dict]] = {}


def _user_file(user_id: str) -> Path:
    return _USERS_DIR / user_id / "tracks.json"


def load_tracks(user_id: str) -> list[dict]:
    """Load tracks for *user_id* (cached after first load)."""
    if user_id in _tracks_cache:
        return _tracks_cache[user_id]

    path = _user_file(user_id)
    if not path.exists():
        return []

    with open(path) as f:
        _tracks_cache[user_id] = json.load(f)
    return _tracks_cache[user_id]


def save_tracks(tracks: list[dict], user_id: str) -> None:
    """Persist tracks for *user_id* and invalidate that user's cache."""
    path = _user_file(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(tracks, f, indent=2, ensure_ascii=False)
    _tracks_cache[user_id] = tracks


def invalidate_cache(user_id: str) -> None:
    _tracks_cache.pop(user_id, None)


def get_track_count(user_id: str) -> int:
    return len(load_tracks(user_id))


def tracks_have_features(user_id: str) -> bool:
    """Check whether the user's tracks have LLM-estimated features."""
    tracks = load_tracks(user_id)
    if not tracks:
        return False
    return "energy" in tracks[0]


def get_all_vibes(user_id: str) -> list[str]:
    """Return sorted unique vibe labels across the user's tracks."""
    vibes = set()
    for t in load_tracks(user_id):
        v = t.get("vibe", "")
        if v:
            vibes.add(v.lower())
    return sorted(vibes)
