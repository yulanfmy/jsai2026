"""Zenodo dataset join for valence/energy ground truth (§3.2 of spec).

Joins user tracks to the 'Almost a million Spotify tracks' dataset
(https://zenodo.org/records/11453410) by Spotify track_id.

Scale conversion: Zenodo values [0,1] → system [-1,+1]: x_internal = 2*x - 1
Tempo is kept as-is (BPM).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "cache"
_ZENODO_CSV = _CACHE_DIR / "zenodo_tracks.csv"


def _to_internal(x: float) -> float:
    """Convert Spotify [0,1] scale to system [-1,+1]."""
    return 2.0 * x - 1.0


def load_zenodo_index(path: Path | None = None) -> dict[str, dict]:
    """Load the Zenodo CSV into a dict keyed by track_id.

    Returns {track_id: {zenodo_valence, zenodo_energy, zenodo_key, ...}}.
    Values are converted to internal scale where applicable.
    """
    if path is None:
        path = _ZENODO_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Zenodo dataset not found at {path}. "
            "Download from https://zenodo.org/records/11453410 and place at cache/zenodo_tracks.csv"
        )

    index: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row.get("track_id", "").strip()
            if not tid:
                continue
            try:
                entry: dict = {}
                # Audio features → internal scale [-1,+1]
                if row.get("valence"):
                    val_raw = float(row["valence"])
                    entry["zenodo_valence_raw"] = val_raw
                    entry["zenodo_valence"] = _to_internal(val_raw)
                if row.get("energy"):
                    en_raw = float(row["energy"])
                    entry["zenodo_energy_raw"] = en_raw
                    entry["zenodo_energy"] = _to_internal(en_raw)

                # Musical features (not scaled)
                if row.get("key"):
                    entry["zenodo_key"] = int(float(row["key"]))
                if row.get("mode"):
                    entry["zenodo_mode"] = int(float(row["mode"]))
                if row.get("tempo"):
                    entry["zenodo_tempo"] = float(row["tempo"])
                if row.get("danceability"):
                    entry["zenodo_danceability"] = float(row["danceability"])
                if row.get("acousticness"):
                    entry["zenodo_acousticness"] = float(row["acousticness"])
                if row.get("instrumentalness"):
                    entry["zenodo_instrumentalness"] = float(row["instrumentalness"])
                if row.get("speechiness"):
                    entry["zenodo_speechiness"] = float(row["speechiness"])
                if row.get("loudness"):
                    entry["zenodo_loudness"] = float(row["loudness"])

                index[tid] = entry
            except (ValueError, KeyError):
                continue

    return index


def join_tracks(
    user_tracks: list[dict],
    zenodo_index: dict[str, dict] | None = None,
) -> tuple[list[dict], int, int]:
    """Join user tracks to Zenodo ground truth.

    Returns (enriched_tracks, matched_count, total_count).
    Matched tracks get zenodo_* fields + has_zenodo=True.
    """
    if zenodo_index is None:
        zenodo_index = load_zenodo_index()

    matched = 0
    enriched: list[dict] = []

    for track in user_tracks:
        tid = track.get("id", "")
        if tid in zenodo_index:
            enriched.append({**track, **zenodo_index[tid], "has_zenodo": True})
            matched += 1
        else:
            enriched.append({**track, "has_zenodo": False})

    return enriched, matched, len(user_tracks)


def save_labeled_parquet(
    tracks: list[dict],
    path: Path | None = None,
    user_id: str | None = None,
) -> Path:
    """Save the matched ground-truth table to cache/labeled_tracks[_<user_id>].parquet."""
    if path is None:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fname = f"labeled_tracks_{user_id}.parquet" if user_id else "labeled_tracks.parquet"
        path = _CACHE_DIR / fname

    labeled = [t for t in tracks if t.get("has_zenodo")]
    table = pa.Table.from_pylist(labeled)
    pq.write_table(table, path)
    return path
