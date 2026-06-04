"""End-to-end offline feature pipeline.

Runs: bootstrap/LLM extraction → Zenodo join → valence correction → energy correction → feature store.
This script is meant to be run once (or when the track library changes).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(_ROOT))


def build(user_id: str | None = None, use_llm: bool = True) -> dict:
    """Run the full offline pipeline.

    Args:
        user_id: Spotify user ID (for per-user store). If None, uses the test set.
        use_llm: If True, use LLM extraction. If False, bootstrap from v1 features.

    Returns:
        Metrics dict with coverage, correction stats, and track count.
    """
    from src.feature_extraction.bootstrap_v1 import bootstrap_all
    from src.feature_extraction.zenodo_join import load_zenodo_index, join_tracks, save_labeled_parquet
    from src.feature_extraction.correct_valence import train_model as train_model_V, correct_valence, write_report as write_report_V
    from src.feature_extraction.correct_energy import train_model as train_model_E, correct_energy, write_report as write_report_E
    from src.feature_store import build_store

    # 1. Load tracks
    if user_id:
        user_path = _ROOT / "src" / "data" / "users" / user_id / "tracks.json"
        if not user_path.exists():
            raise FileNotFoundError(f"No tracks for user {user_id}")
        with open(user_path) as f:
            tracks = json.load(f)
    else:
        test_path = _ROOT / "test_data" / "playlist_1498.json"
        with open(test_path) as f:
            tracks = json.load(f)

    print(f"Loaded {len(tracks)} tracks")

    # 2. Feature extraction (bootstrap or LLM)
    if use_llm:
        from src.feature_extraction.llm_extract import extract_batch, save_raw_parquet
        extracted = extract_batch(tracks, batch_size=10)
        save_raw_parquet(extracted, user_id=user_id)
    else:
        extracted = bootstrap_all(tracks)
    print(f"Extracted features for {len(extracted)} tracks")

    # 3. Zenodo join
    print("Loading Zenodo index...")
    t0 = time.time()
    try:
        zenodo_index = load_zenodo_index()
        enriched, matched, total = join_tracks(extracted, zenodo_index)
        del zenodo_index
        print(f"Zenodo matched: {matched}/{total} ({matched/total*100:.1f}%)")
        save_labeled_parquet(enriched, user_id=user_id)
    except FileNotFoundError:
        print("Zenodo dataset not found — skipping join")
        enriched = extracted
        matched, total = 0, len(extracted)

    # 4. Train valence correction model (only if we have labeled data)
    metrics: dict = {"n_tracks": len(enriched), "matched": matched, "total": total}
    if matched >= 10:
        model_V, train_metrics_V = train_model_V(enriched, user_id=user_id)
        metrics.update(train_metrics_V)
        write_report_V(train_metrics_V)
        corrected = correct_valence(enriched, model_V)
        print(f"Valence correction: r_before={train_metrics_V['r_before']:.4f}, r_after={train_metrics_V['r_after']:.4f}")

        # 4b. Train energy correction model (same Scheme 1+6)
        model_E, train_metrics_E = train_model_E(corrected, user_id=user_id)
        metrics.update(train_metrics_E)
        write_report_E(train_metrics_E)
        corrected = correct_energy(corrected, model_E)
        print(f"Energy correction: r_before={train_metrics_E['r_before_E']:.4f}, r_after={train_metrics_E['r_after_E']:.4f}")
    else:
        corrected = enriched
        for t in corrected:
            if "V" not in t:
                t["V"] = t.get("V_raw", 0.0)
            if "E" not in t:
                t["E"] = t.get("E_raw", 0.0)
        print("Too few labeled tracks for correction — using raw features")

    # 4c. Map T_raw → T (no correction model for Tension — no ground truth)
    for tr in corrected:
        if "T" not in tr and "T_raw" in tr:
            tr["T"] = tr["T_raw"]

    # 5. Build feature store
    store_path = build_store(corrected, user_id=user_id)
    print(f"Feature store built: {store_path} ({len(corrected)} tracks)")

    return metrics


if __name__ == "__main__":
    user = sys.argv[1] if len(sys.argv) > 1 else None
    metrics = build(user_id=user)
    print(f"\nMetrics: {json.dumps(metrics, indent=2)}")
