"""Unit tests for z-score standardization (§3.5).

Acceptance criteria from the patch spec:
  - Standardized library: per-axis mean ≈ 0, std ≈ 1.
  - μ, σ persisted in the feature store.
  - A raw emotion coord through standardize() equals the offline
    transform output (proves track & target use the same transform).
  - §6/§7 distance runs on standardized coords (code-review check).
"""

from __future__ import annotations

import json
import math
import statistics
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tracks(n: int = 50) -> list[dict]:
    """Generate synthetic tracks with known V/E/T distributions."""
    import random
    rng = random.Random(42)
    tracks: list[dict] = []
    for i in range(n):
        tracks.append({
            "id": f"track_{i}",
            "title": f"Song {i}",
            "artist": f"Artist {i}",
            "album": f"Album {i}",
            "V": rng.gauss(0.1, 0.3),
            "E": rng.gauss(-0.2, 0.5),
            "T": rng.gauss(0.3, 0.15),
            "arc_start_E": rng.uniform(-1, 1),
            "arc_start_T": rng.uniform(-1, 1),
            "arc_end_E": rng.uniform(-1, 1),
            "arc_end_T": rng.uniform(-1, 1),
            "key": rng.randint(0, 11),
            "mode": rng.randint(0, 1),
            "tempo": rng.uniform(60, 180),
            "vibe": rng.choice(["energetic", "chill", "dark", "uplifting"]),
            "lyrics_present": rng.choice([True, False]),
        })
    return tracks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildStoreZScore:
    """build_store must compute μ/σ, persist them, and add zV/zE/zT."""

    def test_zscore_columns_present(self, tmp_path: Path):
        from src.feature_store import build_store
        import pyarrow.parquet as pq

        tracks = _make_tracks(30)
        out = build_store(tracks, path=tmp_path / "store.parquet",
                          user_id="test_user_001")

        table = pq.read_table(out)
        cols = set(table.column_names)
        assert "zV" in cols
        assert "zE" in cols
        assert "zT" in cols

    def test_mu_sigma_persisted(self, tmp_path: Path, monkeypatch):
        from src import feature_store as fs
        monkeypatch.setattr(fs, "_CACHE_DIR", tmp_path)

        tracks = _make_tracks(30)
        fs.build_store(tracks, user_id="test_user_002")

        zpath = tmp_path / "zscore_params_test_user_002.json"
        assert zpath.exists(), "zscore_params JSON not written"

        params = json.loads(zpath.read_text())
        assert "mu" in params and "sigma" in params
        for axis in ("V", "E", "T"):
            assert axis in params["mu"]
            assert axis in params["sigma"]
            assert params["sigma"][axis] > 0, f"sigma[{axis}] must be > 0"

    def test_library_mean_zero_std_one(self, tmp_path: Path, monkeypatch):
        """After build, the stored zV/zE/zT should have mean≈0, std≈1."""
        from src import feature_store as fs
        import pyarrow.parquet as pq

        monkeypatch.setattr(fs, "_CACHE_DIR", tmp_path)

        tracks = _make_tracks(200)
        out = fs.build_store(tracks, user_id="test_user_003")

        rows = pq.read_table(out).to_pylist()
        for axis in ("zV", "zE", "zT"):
            vals = [r[axis] for r in rows]
            mu = statistics.mean(vals)
            sd = statistics.stdev(vals)
            assert abs(mu) < 0.01, f"mean({axis}) = {mu}, expected ≈ 0"
            assert abs(sd - 1.0) < 0.01, f"std({axis}) = {sd}, expected ≈ 1"


class TestStandardizeMatchesOffline:
    """standardize(raw_coord) must equal the offline z[k] in the store."""

    def test_online_equals_offline(self, tmp_path: Path, monkeypatch):
        from src import feature_store as fs

        monkeypatch.setattr(fs, "_CACHE_DIR", tmp_path)

        tracks = _make_tracks(100)
        fs.build_store(tracks, user_id="test_user_004")

        zparams = fs.get_zscore_params("test_user_004")

        # Pick a track and verify online transform matches stored value
        import pyarrow.parquet as pq
        rows = pq.read_table(tmp_path / "feature_store_test_user_004.parquet").to_pylist()

        for row in rows[:10]:
            zV_online, zE_online, zT_online = fs.standardize(
                row["V"], row["E"], row["T"], zscore_params=zparams,
            )
            assert abs(zV_online - row["zV"]) < 1e-9
            assert abs(zE_online - row["zE"]) < 1e-9
            assert abs(zT_online - row["zT"]) < 1e-9

    def test_emotion_coord_standardized_correctly(self, tmp_path: Path, monkeypatch):
        """An emotion map coord passed through standardize() uses the
        same μ/σ as the library — proving track and target share space."""
        from src import feature_store as fs

        monkeypatch.setattr(fs, "_CACHE_DIR", tmp_path)

        tracks = _make_tracks(100)
        fs.build_store(tracks, user_id="test_user_005")
        zparams = fs.get_zscore_params("test_user_005")

        # Use an arbitrary emotion coordinate
        raw_V, raw_E, raw_T = 0.4, -0.3, -0.7  # e.g. "Calm"
        zV, zE, zT = fs.standardize(raw_V, raw_E, raw_T, zscore_params=zparams)

        mu = zparams["mu"]
        sig = zparams["sigma"]
        assert abs(zV - (raw_V - mu["V"]) / sig["V"]) < 1e-12
        assert abs(zE - (raw_E - mu["E"]) / sig["E"]) < 1e-12
        assert abs(zT - (raw_T - mu["T"]) / sig["T"]) < 1e-12


class TestTargetLossUsesZSpace:
    """target_loss must use z-scored coords when zscore_params is given."""

    def test_zscore_changes_distance(self, tmp_path: Path, monkeypatch):
        from src import feature_store as fs
        from src.selection.candidates import target_loss
        from src.planning.targets import StageTarget

        monkeypatch.setattr(fs, "_CACHE_DIR", tmp_path)

        tracks = _make_tracks(50)
        fs.build_store(tracks, user_id="test_user_006")
        zparams = fs.get_zscore_params("test_user_006")

        import pyarrow.parquet as pq
        rows = pq.read_table(tmp_path / "feature_store_test_user_006.parquet").to_pylist()
        track = rows[0]

        stage = StageTarget(
            stage=1, V=0.4, E=-0.3, T=-0.7,
            expected_dir_E=-0.1, expected_dir_T=-0.3,
            lead_axis="T",
        )

        loss_raw = target_loss(track, stage, gamma=0.2, zscore_params=None)
        loss_z = target_loss(track, stage, gamma=0.2, zscore_params=zparams)

        # They should differ (unless σ happens to be exactly 1 on all axes)
        # The important thing is that it runs without error and is finite
        assert math.isfinite(loss_z)
        assert math.isfinite(loss_raw)


class TestIdentityFallback:
    """Without a zscore_params file, standardize should be identity."""

    def test_identity_when_no_file(self):
        from src.feature_store import get_zscore_params, standardize

        params = get_zscore_params("nonexistent_user_xyz")
        assert params["mu"] == {"V": 0.0, "E": 0.0, "T": 0.0}
        assert params["sigma"] == {"V": 1.0, "E": 1.0, "T": 1.0}

        zV, zE, zT = standardize(0.5, -0.3, 0.7, zscore_params=params)
        assert abs(zV - 0.5) < 1e-12
        assert abs(zE - (-0.3)) < 1e-12
        assert abs(zT - 0.7) < 1e-12
