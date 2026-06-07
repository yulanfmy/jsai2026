"""Build the v2 feature store for a user's tracks.

Runs the full pipeline: LLM extraction (Gemini) → Zenodo join →
correction (V/E) → feature store + z-scores.

Usage: python -m src.ingest <spotify_user_id>
"""

import sys


def main(user_id: str) -> None:
    from src.feature_extraction.build_pipeline import build

    print(f"Building feature store for user: {user_id}")
    metrics = build(user_id=user_id)

    print(f"\nDone.")
    print(f"  Tracks: {metrics['n_tracks']}")
    print(f"  Zenodo matched: {metrics.get('matched', 0)}/{metrics.get('total', 0)}")
    print(f"  Training pool: {metrics.get('pooled_labeled', 0)} labeled tracks")
    if "r_after" in metrics:
        print(f"  Valence correction: r = {metrics['r_after']:.4f}")
    if "r_after_E" in metrics:
        print(f"  Energy correction: r = {metrics['r_after_E']:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.ingest <spotify_user_id>")
        sys.exit(1)
    main(sys.argv[1])
