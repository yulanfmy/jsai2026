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
│ (Weighted Match)  │     │ (1,498 tracks +  │
└────────┬─────────┘     │  LLM features)   │
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

### 3. Estimate audio features

The track library ships with raw Spotify metadata. Run the ingestion step to estimate audio features (energy, happiness, BPM, etc.) using an LLM:

```bash
python -m src.ingest
```

Or use the in-app estimation button in the Streamlit UI.

### 4. Run the app

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
├── tracks.py            # Track database management
├── ingest.py            # Batch feature estimation script
└── data/
    └── tracks.json      # Track library (Spotify liked songs)
```

## References

- Altshuler, I. M. (1948). *A psychiatrist's experience with music as a therapeutic agent*
- Russell, J. A. (1980). *A circumplex model of affect*. Journal of Personality and Social Psychology, 39(6), 1161-1178
- Posner, J., Russell, J. A., & Peterson, B. S. (2005). *The circumplex model of affect: An integrative approach to affective neuroscience*
