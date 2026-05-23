"""Playlist generation using transition strategies and track scoring."""

from dataclasses import dataclass

from src.config import TRACKS_PER_PHASE
from src.emotions import Emotion
from src.scoring import select_tracks
from src.strategies import Phase, get_strategy
from src.tracks import load_tracks


@dataclass
class PlaylistPhase:
    phase: Phase
    tracks: list[dict]


@dataclass
class Playlist:
    strategy_name: str
    current_emotion: Emotion
    target_emotion: Emotion
    phases: list[PlaylistPhase]

    @property
    def all_tracks(self) -> list[dict]:
        result = []
        for pp in self.phases:
            result.extend(pp.tracks)
        return result


def generate_playlist(
    current: Emotion,
    target: Emotion,
    strategy_name: str = "Dynamic",
    tracks_per_phase: int | None = None,
    vibe_keywords: list[str] | None = None,
    prefer_vocals: bool | None = None,
) -> Playlist:
    """Generate a transition playlist from current to target emotion."""
    if tracks_per_phase is None:
        tracks_per_phase = TRACKS_PER_PHASE

    strategy_fn = get_strategy(strategy_name)
    phases = strategy_fn(current, target)

    all_tracks = load_tracks()
    used_ids: set[str] = set()
    playlist_phases: list[PlaylistPhase] = []

    for phase in phases:
        selected = select_tracks(
            all_tracks,
            phase,
            count=tracks_per_phase,
            exclude_ids=used_ids,
            vibe_keywords=vibe_keywords,
            prefer_vocals=prefer_vocals,
        )
        for t in selected:
            used_ids.add(t.get("id", t.get("title", "")))
        playlist_phases.append(PlaylistPhase(phase=phase, tracks=selected))

    return Playlist(
        strategy_name=strategy_name,
        current_emotion=current,
        target_emotion=target,
        phases=playlist_phases,
    )
