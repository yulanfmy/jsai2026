"""Configuration for MindTune v2.

Environment-level config (API keys, Spotify creds) lives here.
Algorithm parameters come from config/params.yaml via config_loader.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# --- LLM ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# --- Spotify ---
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8501")

# --- v2 fixed parameters — re-exported from config_loader for convenience ---
from src.config_loader import PARAMS  # noqa: E402

N_STAGES = PARAMS.N
K_DEFAULT = PARAMS.K_default
ALPHA = PARAMS.alpha
LAMBDA = PARAMS.lam
GAMMA = PARAMS.gamma
W1 = PARAMS.w1
W2 = PARAMS.w2
W3 = PARAMS.w3

# Backward compat
TRACKS_PER_PHASE = 3
