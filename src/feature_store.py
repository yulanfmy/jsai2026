"""Feature store — read-only access layer for the offline feature pipeline (§3.4).

Online code reads features ONLY through this module.
Never calls the LLM or the Zenodo join at request time.

The store is backed by a parquet file at cache/feature_store.parquet.
Features per track: track_id, V, E, T, arc_start_E, arc_start_T,
  arc_end_E, arc_end_T, key, mode, tempo, vibe, lyrics_present.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
_STORE_PATH = _CACHE_DIR / "feature_store.parquet"

STORE_COLUMNS = [
    "id", "title", "artist", "album",
    "V", "E", "T",
    "arc_start_E", "arc_start_T", "arc_end_E", "arc_end_T",
    "key", "mode", "tempo", "vibe", "lyrics_present",
]

# In-memory index for fast lookup
_INDEX: dict[str, dict] | None = None


def _ensure_loaded() -> dict[str, dict]:
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    _INDEX = _load_index()
    return _INDEX


def _load_index(path: Path | None = None) -> dict[str, dict]:
    if path is None:
        path = _STORE_PATH
    if not path.exists():
        return {}
    table = pq.read_table(path)
    rows = table.to_pylist()
    return {row["id"]: row for row in rows if "id" in row}


def get(track_id: str) -> dict | None:
    """Get a single track's features by ID. Returns None if not found."""
    idx = _ensure_loaded()
    return idx.get(track_id)


def get_all() -> list[dict]:
    """Get all tracks with features."""
    idx = _ensure_loaded()
    return list(idx.values())


def get_all_for_user(user_id: str) -> list[dict]:
    """Get all tracks for a specific user.

    Checks if a user-specific store exists, otherwise falls back to the
    global feature store.
    """
    user_store = _CACHE_DIR / f"feature_store_{user_id}.parquet"
    if user_store.exists():
        return list(_load_index(user_store).values())
    return get_all()


def track_count() -> int:
    """How many tracks are in the store."""
    idx = _ensure_loaded()
    return len(idx)


def reload():
    """Force reload from disk (e.g., after rebuilding the store)."""
    global _INDEX
    _INDEX = None
    _ensure_loaded()


def build_store(
    corrected_tracks: list[dict],
    path: Path | None = None,
    user_id: str | None = None,
) -> Path:
    """Build the feature store from corrected tracks.

    Filters to STORE_COLUMNS and writes to parquet.
    """
    import pyarrow as pa

    if path is None:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if user_id:
            path = _CACHE_DIR / f"feature_store_{user_id}.parquet"
        else:
            path = _STORE_PATH

    rows: list[dict] = []
    for t in corrected_tracks:
        row: dict = {}
        for col in STORE_COLUMNS:
            if col in t:
                row[col] = t[col]
        if "id" in row:
            rows.append(row)

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)

    # Reset the in-memory index
    global _INDEX
    _INDEX = None

    return path
