"""LLM-based audio feature estimation from track metadata.

Uses OpenAI to estimate audio features (energy, happiness, bpm, danceability,
acousticness, instrumentalness, vibe, best_listening_context) from track
title and artist name, as described in the paper.
"""

import json

from openai import OpenAI

from src.config import CHAT_MODEL, OPENAI_API_KEY

ESTIMATION_PROMPT = """\
You are a music analysis expert. Given a track's metadata, estimate its audio features.
Return ONLY a JSON object with these fields (no markdown, no explanation):

{
  "energy": <float 0-1, how energetic/intense the track is>,
  "happiness": <float 0-1, how positive/cheerful the track sounds (proxy for valence)>,
  "bpm": <int, estimated beats per minute>,
  "danceability": <float 0-1, how suitable for dancing>,
  "acousticness": <float 0-1, likelihood of being acoustic>,
  "instrumentalness": <float 0-1, likelihood of having no vocals>,
  "vibe": <string, one or two word mood descriptor e.g. "chill", "intense", "dreamy">,
  "best_listening_context": <string, e.g. "late night drive", "morning workout", "studying">
}

Base your estimates on your knowledge of the track's genre, artist style, and cultural context.\
"""

BATCH_PROMPT = """\
You are a music analysis expert. For each track below, estimate its audio features.
Return ONLY a JSON array where each element corresponds to a track (in order) with these fields:

{
  "energy": <float 0-1>,
  "happiness": <float 0-1, proxy for valence>,
  "bpm": <int>,
  "danceability": <float 0-1>,
  "acousticness": <float 0-1>,
  "instrumentalness": <float 0-1>,
  "vibe": <string, one or two word mood>,
  "best_listening_context": <string>
}

Return ONLY the JSON array, no markdown or explanation.\
"""


def _parse_json(text: str) -> dict | list:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def estimate_features(
    title: str,
    artist: str,
    album: str = "",
    client: OpenAI | None = None,
) -> dict:
    """Estimate audio features for a single track using an LLM."""
    if client is None:
        client = OpenAI(api_key=OPENAI_API_KEY)

    metadata = f"Title: {title}\nArtist: {artist}"
    if album:
        metadata += f"\nAlbum: {album}"

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": ESTIMATION_PROMPT},
            {"role": "user", "content": metadata},
        ],
        temperature=0.3,
        max_tokens=256,
    )

    text = response.choices[0].message.content or ""
    try:
        return _parse_json(text)
    except (json.JSONDecodeError, ValueError):
        return {}


def estimate_batch(
    tracks: list[dict],
    batch_size: int = 10,
    client: OpenAI | None = None,
    progress_callback: object = None,
) -> list[dict]:
    """Estimate features for a list of tracks in batches.

    Each track dict must have 'title' and 'artist' keys.
    Returns the same list with estimated features merged in.
    """
    if client is None:
        client = OpenAI(api_key=OPENAI_API_KEY)

    results = []
    total = len(tracks)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = tracks[start:end]

        track_list = ""
        for i, t in enumerate(batch, 1):
            track_list += f"{i}. Title: {t['title']} | Artist: {t['artist']}"
            if t.get("album"):
                track_list += f" | Album: {t['album']}"
            track_list += "\n"

        try:
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": BATCH_PROMPT},
                    {"role": "user", "content": track_list},
                ],
                temperature=0.3,
                max_tokens=2048,
            )
            text = response.choices[0].message.content or ""
            features_list = _parse_json(text)

            if isinstance(features_list, list):
                for j, track in enumerate(batch):
                    if j < len(features_list):
                        merged = {**track, **features_list[j]}
                    else:
                        merged = track
                    results.append(merged)
            else:
                results.extend(batch)
        except (json.JSONDecodeError, ValueError, KeyError):
            results.extend(batch)

        if progress_callback is not None:
            progress_callback(end, total)

    return results
