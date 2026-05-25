# MindTune

Emotion-transition music recommendation system based on the **ISO principle** and **Russell's circumplex model**. MindTune generates playlists that guide users from their current emotional state to a desired target state through carefully ordered music.

## Overview

Unlike conventional recommendation systems that optimize for taste, MindTune optimizes for **emotional trajectory**. It uses music therapy's ISO principle to gradually transition listeners through curated playlists, with tracks selected using a multi-dimensional scoring algorithm.

Key features:
- **11 emotion states** mapped on Russell's Arousal-Valence 2D space
- **4 transition strategies**: Arousal First, Valence First, Linear, and Dynamic
- **LLM-based audio feature estimation** from track metadata (energy, happiness, BPM, etc.)
- **Weighted scoring algorithm** for track selection (energy:3, happiness:3, BPM:2, vibe:1, instrumentalness:2)
- **Interactive visualization** of the circumplex model and transition paths
- **Spotify integration** — track library sourced from user's Spotify liked songs
- **Multi-user support** — each user's library is stored and queried independently
- **Play on Spotify** via Spotify Connect — send playlists directly to your phone or desktop

## Architecture

```
Emotion Input (Current → Target)
         │
         ▼
┌──────────────────┐
│ Strategy Engine   │  ← Arousal First / Valence First / Linear / Dynamic
│ (ISO Principle)   │
└────────┬─────────┘
         │  3 Phases
         ▼
┌──────────────────┐     ┌──────────────────┐
│ Scoring Algorithm │────▶│  Track Library    │
│ (Weighted Match)  │     │ (per-user stored  │
└────────┬─────────┘     │  + LLM features)  │
         │               └──────────────────┘
         ▼
┌──────────────────┐
│ Streamlit UI      │  ← Circumplex visualization + playlist
└──────────────────┘
```

## Setup

### 1. Install dependencies

```bash
pip install -e .
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

### 3. Log in and import your library

Open the app and either:
- **Connect with Spotify** (recommended) — automatically identifies you and lets you import your liked songs
- **Enter your Spotify User ID** — if your library was previously imported

### 4. Estimate audio features

After importing your library, estimate audio features using the in-app button, or from the command line:

```bash
python -m src.ingest <your_spotify_user_id>
```

### 5. Run the app

```bash
streamlit run src/app.py
```

## Transition Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| **Arousal First** | Adjusts energy first, then mood | Anxiety, anger (high-arousal negative) |
| **Valence First** | Shifts mood first, then energy | Sadness, fatigue (low-arousal negative) |
| **Linear** | Changes both dimensions equally | Moderate transitions |
| **Dynamic** | Auto-selects based on larger gap | General use (recommended) |

## Project Structure

```
src/
├── app.py               # Streamlit UI
├── config.py            # Environment settings
├── emotions.py          # Russell's circumplex model (11 emotions)
├── strategies.py        # 4 transition strategies
├── scoring.py           # Weighted track scoring algorithm
├── playlist.py          # Playlist generation orchestrator
├── feature_estimator.py # LLM-based audio feature estimation
├── tracks.py            # Per-user track database management
├── ingest.py            # Batch feature estimation script
├── spotify.py           # Spotify API (auth, library import, playback)
└── data/
    └── users/           # Per-user track libraries (gitignored)
        └── <user_id>/
            └── tracks.json
```

## References

- Altshuler, I. M. (1948). *A psychiatrist's experience with music as a therapeutic agent*
- Russell, J. A. (1980). *A circumplex model of affect*. Journal of Personality and Social Psychology, 39(6), 1161-1178
- Posner, J., Russell, J. A., & Peterson, B. S. (2005). *The circumplex model of affect: An integrative approach to affective neuroscience*
