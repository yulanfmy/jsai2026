"""Configuration for MindTune."""

import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
TRACKS_PER_PHASE = int(os.getenv("TRACKS_PER_PHASE", "3"))
