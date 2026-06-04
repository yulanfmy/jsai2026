"""Feature store — read-only access layer for the offline feature pipeline (§3.4).

Online code reads features ONLY through this module.
Never calls the LLM or the Zenodo join at request time.

The store is backed by a parquet file at cache/feature_store.parquet.
Features per track: track_id, V, E, T, zV, zE, zT (z-scored),
  arc_start_E, arc_start_T, arc_end_E, arc_end_T, key, mode, tempo,
  vibe, lyrics_present.

Per-axis μ and σ are persisted in cache/zscore_params[_<user_id>].json
and used by ``standardize()`` to map any (V,E,T) into the same z-space
that the library tracks live in.
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
    "zV", "zE", "zT",
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
    idx = {row["id"]: row for row in rows if "id" in row}

    # Backfill T from llm_raw.parquet if missing in feature store
    if idx and "T" not in next(iter(idx.values()), {}):
        llm_path = _CACHE_DIR / "llm_raw.parquet"
        if llm_path.exists():
            llm_table = pq.read_table(llm_path, columns=["id", "T_raw"])
            for llm_row in llm_table.to_pylist():
                tid = llm_row.get("id")
                if tid and tid in idx:
                    idx[tid]["T"] = float(llm_row.get("T_raw", 0.0))

    return idx


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


# ---------------------------------------------------------------------------
# Z-score standardization helpers (§3.5)
# ---------------------------------------------------------------------------

def _zscore_path(user_id: str | None = None) -> Path:
    if user_id:
        return _CACHE_DIR / f"zscore_params_{user_id}.json"
    return _CACHE_DIR / "zscore_params.json"


def get_zscore_params(user_id: str | None = None) -> dict:
    """Load persisted μ, σ per axis.

    Returns ``{"mu": {"V": …, "E": …, "T": …}, "sigma": {…}}``.
    Falls back to identity (μ=0, σ=1) if the file doesn't exist.
    """
    path = _zscore_path(user_id)
    if not path.exists():
        # Identity transform — raw values pass through unchanged
        return {
            "mu": {"V": 0.0, "E": 0.0, "T": 0.0},
            "sigma": {"V": 1.0, "E": 1.0, "T": 1.0},
        }
    with open(path) as f:
        return json.load(f)


def standardize(
    V: float, E: float, T: float,
    zscore_params: dict | None = None,
    user_id: str | None = None,
) -> tuple[float, float, float]:
    """Apply z-score: z[k] = (x[k] − μ[k]) / σ[k].

    Provide *either* ``zscore_params`` (already loaded dict) or
    ``user_id`` (will load from disk).  Caller is responsible for
    making sure the same μ/σ are used for tracks AND targets.
    """
    if zscore_params is None:
        zscore_params = get_zscore_params(user_id)
    mu = zscore_params["mu"]
    sig = zscore_params["sigma"]
    zV = (V - mu["V"]) / sig["V"] if sig["V"] > 1e-10 else 0.0
    zE = (E - mu["E"]) / sig["E"] if sig["E"] > 1e-10 else 0.0
    zT = (T - mu["T"]) / sig["T"] if sig["T"] > 1e-10 else 0.0
    return zV, zE, zT


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_store(
    corrected_tracks: list[dict],
    path: Path | None = None,
    user_id: str | None = None,
) -> Path:
    """Build the feature store from corrected tracks.

    Filters to STORE_COLUMNS, computes per-axis μ/σ, adds z-scored
    columns (zV, zE, zT), persists μ/σ as JSON sidecar, and writes
    the parquet file.
    """
    import statistics

    import pyarrow as pa

    if path is None:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if user_id:
            path = _CACHE_DIR / f"feature_store_{user_id}.parquet"
        else:
            path = _STORE_PATH

    # --- collect rows (without z-cols initially) ---
    raw_cols = [c for c in STORE_COLUMNS if c not in ("zV", "zE", "zT")]
    rows: list[dict] = []
    for t in corrected_tracks:
        row: dict = {}
        for col in raw_cols:
            if col in t:
                row[col] = t[col]
        if "id" in row:
            rows.append(row)

    # --- compute μ, σ over the library ---
    vs = [float(r["V"]) for r in rows if "V" in r]
    es = [float(r["E"]) for r in rows if "E" in r]
    ts = [float(r["T"]) for r in rows if "T" in r]

    mu = {
        "V": statistics.mean(vs) if vs else 0.0,
        "E": statistics.mean(es) if es else 0.0,
        "T": statistics.mean(ts) if ts else 0.0,
    }
    sigma = {
        "V": statistics.stdev(vs) if len(vs) > 1 else 1.0,
        "E": statistics.stdev(es) if len(es) > 1 else 1.0,
        "T": statistics.stdev(ts) if len(ts) > 1 else 1.0,
    }

    # Persist μ/σ
    zp = _zscore_path(user_id)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(zp, "w") as f:
        json.dump({"mu": mu, "sigma": sigma}, f, indent=2)

    # --- add z-scored columns ---
    zparams = {"mu": mu, "sigma": sigma}
    for r in rows:
        zV, zE, zT = standardize(
            float(r.get("V", 0)), float(r.get("E", 0)), float(r.get("T", 0)),
            zscore_params=zparams,
        )
        r["zV"] = zV
        r["zE"] = zE
        r["zT"] = zT

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)

    # Reset the in-memory index
    global _INDEX
    _INDEX = None

    return path
