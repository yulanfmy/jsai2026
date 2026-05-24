"""Estimate audio features for a user's tracks using an LLM.

Run this once per user after loading their tracks to populate energy,
happiness, bpm, etc.

Usage: python -m src.ingest <spotify_user_id>
"""

import sys

from src.feature_estimator import estimate_batch
from src.tracks import load_tracks, save_tracks


def main(user_id: str) -> None:
    tracks = load_tracks(user_id)
    if not tracks:
        print(f"No tracks found for user '{user_id}'.")
        return

    already_done = sum(1 for t in tracks if "energy" in t)
    remaining = [t for t in tracks if "energy" not in t]

    print(f"User: {user_id}")
    print(f"Total tracks: {len(tracks)}")
    print(f"Already estimated: {already_done}")
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("All tracks already have features. Nothing to do.")
        return

    def progress(done: int, total: int) -> None:
        print(f"  Estimated {done}/{total} tracks...")

    estimated = estimate_batch(remaining, batch_size=10, progress_callback=progress)

    done_map = {t["id"]: t for t in estimated if "energy" in t}

    updated = []
    for t in tracks:
        if t["id"] in done_map:
            updated.append(done_map[t["id"]])
        else:
            updated.append(t)

    save_tracks(updated, user_id)
    newly_done = sum(1 for t in updated if "energy" in t) - already_done
    print(f"\nDone. Estimated features for {newly_done} new tracks.")
    print(f"Total tracks with features: {sum(1 for t in updated if 'energy' in t)}/{len(updated)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.ingest <spotify_user_id>")
        sys.exit(1)
    main(sys.argv[1])
