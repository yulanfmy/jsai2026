"""LLM-based feature extraction for MindTune v2 (§3.1 of spec).

Calls Gemini (preferred) or OpenAI (fallback) to extract per-track:
  - V_raw, E_raw, T_raw (3D emotion coordinates in [-1,+1])
  - arc_start / arc_end (Scheme B: start/end (E,T) vectors — numeric, not text)
  - sub_features: mode_major_conf, lyric_sentiment, vocal_brightness, chord_complexity
  - key (0-11), mode (0/1), tempo (BPM float), vibe (string), lyrics_present (bool)

Includes retries and JSON-parse tolerance.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.config import GEMINI_API_KEY, OPENAI_API_KEY

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "cache"

EXTRACTION_PROMPT = """\
You are a music analysis expert. Given a track's metadata, estimate its audio features.
Return ONLY a JSON object with these fields (no markdown, no explanation):

{
  "V_raw": <float -1 to 1, valence: unpleasant(-1) to pleasant(+1)>,
  "E_raw": <float -1 to 1, energy arousal: tired(-1) to awake/active(+1)>,
  "T_raw": <float -1 to 1, tension arousal: relaxed(-1) to tense/edgy(+1)>,
  "arc_start": {"E": <float -1 to 1>, "T": <float -1 to 1>},
  "arc_end": {"E": <float -1 to 1>, "T": <float -1 to 1>},
  "sub_features": {
    "mode_major_conf": <float 0 to 1, likelihood of major key>,
    "lyric_sentiment": <float -1 to 1, lyric sentiment>,
    "vocal_brightness": <float 0 to 1, vocal/timbre brightness>,
    "chord_complexity": <float 0 to 1, harmonic complexity>
  },
  "key": <int 0-11, C=0, C#=1, ..., B=11>,
  "mode": <int 0 or 1, 0=minor, 1=major>,
  "tempo": <float, estimated BPM>,
  "vibe": <string, one or two word mood descriptor>,
  "lyrics_present": <boolean, true if track has vocals/lyrics>
}

For arc_start/arc_end: estimate the emotional trajectory within the track.
arc_start is the (E,T) state at the beginning; arc_end is at the end.
Base your estimates on your knowledge of the track's genre, artist style, and song structure.\
"""

BATCH_PROMPT = """\
You are a music analysis expert. For each track below, estimate its audio features.
Return ONLY a JSON array where each element corresponds to a track (in order) with fields:
V_raw, E_raw, T_raw (floats -1 to 1),
arc_start (object with E and T floats),
arc_end (object with E and T floats),
sub_features (object with mode_major_conf 0-1, lyric_sentiment -1 to 1, vocal_brightness 0-1, chord_complexity 0-1),
key (int 0-11), mode (int 0 or 1), tempo (float BPM),
vibe (string), lyrics_present (boolean).

Return ONLY the JSON array, no markdown or explanation.\
"""


def _parse_json(text: str) -> dict | list:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def _flatten_features(raw: dict) -> dict:
    """Flatten LLM JSON output into the flat track feature schema."""
    result: dict = {}
    result["V_raw"] = float(raw.get("V_raw", 0.0))
    result["E_raw"] = float(raw.get("E_raw", 0.0))
    result["T_raw"] = float(raw.get("T_raw", 0.0))

    arc_start = raw.get("arc_start", {})
    arc_end = raw.get("arc_end", {})
    result["arc_start_E"] = float(arc_start.get("E", result["E_raw"]))
    result["arc_start_T"] = float(arc_start.get("T", result["T_raw"]))
    result["arc_end_E"] = float(arc_end.get("E", result["E_raw"]))
    result["arc_end_T"] = float(arc_end.get("T", result["T_raw"]))

    sub = raw.get("sub_features", {})
    result["mode_major_conf"] = float(sub.get("mode_major_conf", 0.5))
    result["lyric_sentiment"] = float(sub.get("lyric_sentiment", 0.0))
    result["vocal_brightness"] = float(sub.get("vocal_brightness", 0.5))
    result["chord_complexity"] = float(sub.get("chord_complexity", 0.5))

    result["key"] = int(raw.get("key", 0))
    result["mode"] = int(raw.get("mode", 0))
    result["tempo"] = float(raw.get("tempo", 120.0))
    result["vibe"] = str(raw.get("vibe", ""))
    result["lyrics_present"] = bool(raw.get("lyrics_present", True))

    return result


def _call_llm(prompt: str, user_content: str, max_retries: int = 3) -> str:
    """Call the best available LLM with retries."""
    last_err = None
    for attempt in range(max_retries):
        try:
            if GEMINI_API_KEY:
                return _call_gemini(prompt, user_content)
            if OPENAI_API_KEY:
                return _call_openai(prompt, user_content)
            raise RuntimeError("No LLM API key. Set GEMINI_API_KEY or OPENAI_API_KEY.")
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_err}")


def _call_gemini(prompt: str, user_content: str) -> str:
    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(
        [{"role": "user", "parts": [prompt + "\n\n" + user_content]}],
        generation_config={"temperature": 0.3, "max_output_tokens": 4096},
    )
    return response.text or ""


def _call_openai(prompt: str, user_content: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
        max_tokens=4096,
    )
    return response.choices[0].message.content or ""


def extract_single(title: str, artist: str, album: str = "") -> dict:
    """Extract features for a single track."""
    metadata = f"Title: {title}\nArtist: {artist}"
    if album:
        metadata += f"\nAlbum: {album}"
    text = _call_llm(EXTRACTION_PROMPT, metadata)
    raw = _parse_json(text)
    if isinstance(raw, dict):
        return _flatten_features(raw)
    return {}


def extract_batch(
    tracks: list[dict],
    batch_size: int = 10,
    progress_callback: object = None,
) -> list[dict]:
    """Extract features for a list of tracks in batches.

    Each track dict must have 'title' and 'artist' keys.
    Returns the same list with extracted features merged in.
    """
    results: list[dict] = []
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
            text = _call_llm(BATCH_PROMPT, track_list)
            features_list = _parse_json(text)

            if isinstance(features_list, list):
                for j, track in enumerate(batch):
                    if j < len(features_list) and isinstance(features_list[j], dict):
                        flat = _flatten_features(features_list[j])
                        merged = {**track, **flat}
                    else:
                        merged = track
                    results.append(merged)
            else:
                results.extend(batch)
        except Exception:
            results.extend(batch)

        if progress_callback is not None:
            progress_callback(end, total)

    return results


def save_raw_parquet(tracks: list[dict], path: Path | None = None) -> Path:
    """Save extracted features to cache/llm_raw.parquet."""
    if path is None:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _CACHE_DIR / "llm_raw.parquet"

    table = pa.Table.from_pylist(tracks)
    pq.write_table(table, path)
    return path


def load_raw_parquet(path: Path | None = None) -> list[dict]:
    """Load previously extracted features from parquet."""
    if path is None:
        path = _CACHE_DIR / "llm_raw.parquet"
    if not path.exists():
        return []
    table = pq.read_table(path)
    return table.to_pylist()
