"""MindTune - Emotion-Transition Music Recommendation System.

Streamlit web interface implementing the ISO principle and Russell's
circumplex model for emotionally adaptive playlist generation.
"""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import plotly.graph_objects as go
import streamlit as st

from src.config import OPENAI_API_KEY
from src.emotions import Emotion, get_emotion, list_emotions
from src.feature_estimator import estimate_batch
from src.playlist import Playlist, generate_playlist
from src.strategies import get_strategy_description
from src.tracks import (
    get_all_vibes,
    get_track_count,
    invalidate_cache,
    load_tracks,
    save_tracks,
    tracks_have_features,
)

st.set_page_config(page_title="MindTune", page_icon="\U0001f3b5", layout="wide")


def render_circumplex(
    current: Emotion | None = None,
    target: Emotion | None = None,
    playlist: Playlist | None = None,
) -> go.Figure:
    """Draw the Arousal-Valence circumplex with emotions and transition path."""
    fig = go.Figure()

    emotions = list_emotions()
    fig.add_trace(
        go.Scatter(
            x=[e.valence for e in emotions],
            y=[e.arousal for e in emotions],
            mode="markers+text",
            text=[e.name for e in emotions],
            textposition="top center",
            marker=dict(size=10, color="#888"),
            name="Emotions",
            hovertemplate="%{text}<br>Valence: %{x:.1f}<br>Arousal: %{y:.1f}<extra></extra>",
        )
    )

    if current:
        fig.add_trace(
            go.Scatter(
                x=[current.valence],
                y=[current.arousal],
                mode="markers+text",
                text=[f"NOW: {current.name}"],
                textposition="bottom center",
                marker=dict(size=16, color="#FF6B6B", symbol="star"),
                name="Current",
            )
        )
    if target:
        fig.add_trace(
            go.Scatter(
                x=[target.valence],
                y=[target.arousal],
                mode="markers+text",
                text=[f"GOAL: {target.name}"],
                textposition="bottom center",
                marker=dict(size=16, color="#4ECDC4", symbol="star"),
                name="Target",
            )
        )

    if playlist and current and target:
        path_v = [current.valence]
        path_a = [current.arousal]
        labels = ["Start"]
        for pp in playlist.phases:
            path_v.append(pp.phase.valence)
            path_a.append(pp.phase.arousal)
            labels.append(pp.phase.label)

        fig.add_trace(
            go.Scatter(
                x=path_v,
                y=path_a,
                mode="lines+markers",
                line=dict(dash="dot", width=2, color="#FFD93D"),
                marker=dict(size=8, color="#FFD93D"),
                text=labels,
                name="Transition Path",
                hovertemplate="%{text}<br>Valence: %{x:.2f}<br>Arousal: %{y:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Russell's Circumplex Model of Affect",
        xaxis=dict(
            title="Valence (Negative \u2190 \u2192 Positive)",
            range=[-1.2, 1.2],
            zeroline=True,
            zerolinecolor="#555",
        ),
        yaxis=dict(
            title="Arousal (Low \u2190 \u2192 High)",
            range=[-1.2, 1.2],
            zeroline=True,
            zerolinecolor="#555",
        ),
        height=500,
        showlegend=True,
        template="plotly_dark",
    )
    return fig


def render_feature_estimation_ui() -> None:
    """Show UI for running LLM feature estimation on tracks."""
    st.warning(
        f"Your track library has {get_track_count()} tracks but audio features "
        "have not been estimated yet. Run the feature estimation to enable "
        "playlist generation."
    )

    if not OPENAI_API_KEY:
        st.error(
            "Set `OPENAI_API_KEY` in your `.env` file to enable LLM feature estimation."
        )
        return

    st.info(
        "This will use OpenAI to estimate energy, happiness, BPM, and other "
        "features for each track based on its metadata. Tracks are processed "
        "in batches of 10."
    )

    if st.button("Estimate Features", type="primary"):
        tracks = load_tracks()
        remaining = [t for t in tracks if "energy" not in t]

        progress_bar = st.progress(0, text="Estimating features...")
        status = st.empty()

        def on_progress(done: int, total: int) -> None:
            progress_bar.progress(done / total, text=f"Estimated {done}/{total} tracks...")
            status.text(f"Processing batch... ({done}/{total})")

        estimated = estimate_batch(remaining, batch_size=10, progress_callback=on_progress)

        done_map = {t["id"]: t for t in estimated if "energy" in t}
        updated = []
        for t in tracks:
            if t["id"] in done_map:
                updated.append(done_map[t["id"]])
            else:
                updated.append(t)

        save_tracks(updated)
        invalidate_cache()

        newly_done = sum(1 for t in updated if "energy" in t)
        progress_bar.progress(1.0, text="Done!")
        status.success(f"Estimated features for {newly_done}/{len(updated)} tracks.")
        st.rerun()


def render_playlist(playlist: Playlist) -> None:
    """Display the generated playlist with phase breakdown."""
    for i, pp in enumerate(playlist.phases):
        phase = pp.phase
        st.subheader(f"Phase {i + 1}: {phase.label}")
        st.caption(
            f"Target \u2014 Energy: {(phase.arousal + 1) / 2:.2f} | "
            f"Happiness: {(phase.valence + 1) / 2:.2f} | "
            f"BPM: {phase.bpm_low}\u2013{phase.bpm_high}"
        )

        if not pp.tracks:
            st.write("*No matching tracks for this phase*")
            continue

        for j, track in enumerate(pp.tracks, 1):
            energy = track.get("energy", "?")
            happiness = track.get("happiness", "?")
            bpm = track.get("bpm", "?")
            vibe = track.get("vibe", "")
            context = track.get("best_listening_context", "")

            energy_str = f"{energy:.2f}" if isinstance(energy, float) else str(energy)
            happiness_str = (
                f"{happiness:.2f}" if isinstance(happiness, float) else str(happiness)
            )

            col1, col2 = st.columns([3, 2])
            with col1:
                spotify_url = f"https://open.spotify.com/track/{track.get('id', '')}"
                st.markdown(
                    f"**{j}. [{track['title']}]({spotify_url})** \u2014 {track['artist']}"
                )
            with col2:
                st.caption(
                    f"E:{energy_str} | H:{happiness_str} | BPM:{bpm}"
                    + (f" | {vibe}" if vibe else "")
                    + (f" | {context}" if context else "")
                )
        st.divider()


def main() -> None:
    st.title("\U0001f3b5 MindTune")
    st.caption(
        "Emotion-transition music recommendation based on the ISO principle "
        "and Russell's circumplex model"
    )

    track_count = get_track_count()
    if track_count == 0:
        st.error("No tracks loaded. Add tracks to `src/data/tracks.json`.")
        st.stop()

    has_features = tracks_have_features()

    if not has_features:
        render_feature_estimation_ui()
        st.divider()
        st.info("You can still explore the emotion model below while features are being set up.")

    st.sidebar.header("Emotion Settings")

    emotion_names = [e.name for e in list_emotions()]

    current_name = st.sidebar.selectbox(
        "How are you feeling now?",
        emotion_names,
        index=emotion_names.index("Anxious"),
    )
    target_name = st.sidebar.selectbox(
        "How do you want to feel?",
        emotion_names,
        index=emotion_names.index("Calm"),
    )

    current = get_emotion(current_name)
    target = get_emotion(target_name)

    st.sidebar.header("Transition Strategy")
    strategy_names = ["Dynamic", "Arousal First", "Valence First", "Linear"]
    strategy = st.sidebar.selectbox("Strategy", strategy_names, index=0)
    st.sidebar.caption(get_strategy_description(strategy))

    st.sidebar.header("Preferences")
    tracks_per_phase = st.sidebar.slider("Tracks per phase", 1, 5, 3)

    vocals_option = st.sidebar.radio(
        "Vocals preference",
        ["No preference", "Prefer vocals", "Prefer instrumental"],
        index=0,
    )
    prefer_vocals: bool | None = None
    if vocals_option == "Prefer vocals":
        prefer_vocals = True
    elif vocals_option == "Prefer instrumental":
        prefer_vocals = False

    vibes = get_all_vibes()
    selected_vibes: list[str] = []
    if vibes:
        selected_vibes = st.sidebar.multiselect("Vibe filter", vibes)

    tab1, tab2, tab3 = st.tabs(
        ["\U0001f3b6 Generate Playlist", "\U0001f4ca Circumplex Model", "\U0001f4c0 Track Library"]
    )

    with tab1:
        if current_name == target_name:
            st.info("Select different emotions for current and target to generate a playlist.")
        elif not has_features:
            st.info(
                "Run feature estimation first (see above) to generate playlists."
            )
        else:
            if st.button("Generate Playlist", type="primary"):
                playlist = generate_playlist(
                    current=current,
                    target=target,
                    strategy_name=strategy,
                    tracks_per_phase=tracks_per_phase,
                    vibe_keywords=selected_vibes if selected_vibes else None,
                    prefer_vocals=prefer_vocals,
                )
                st.session_state["playlist"] = playlist

            if "playlist" in st.session_state:
                playlist = st.session_state["playlist"]
                st.success(
                    f"Playlist: {playlist.current_emotion.name} \u2192 "
                    f"{playlist.target_emotion.name} "
                    f"({playlist.strategy_name} strategy)"
                )
                render_playlist(playlist)

                st.plotly_chart(
                    render_circumplex(current, target, playlist),
                    use_container_width=True,
                )

    with tab2:
        playlist_to_show = st.session_state.get("playlist")
        fig = render_circumplex(current, target, playlist_to_show)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Emotion Details")
        cols = st.columns(3)
        for i, em in enumerate(list_emotions()):
            with cols[i % 3]:
                st.markdown(
                    f"**{em.name}** \u2014 A:{em.arousal:.1f}, V:{em.valence:.1f}  \n"
                    f"BPM: {em.bpm_low}\u2013{em.bpm_high}  \n"
                    f"_{em.description}_"
                )

    with tab3:
        st.subheader(f"Track Library ({track_count} tracks)")
        tracks = load_tracks()

        search = st.text_input("Search tracks", "")
        if search:
            search_lower = search.lower()
            tracks = [
                t
                for t in tracks
                if search_lower in t.get("title", "").lower()
                or search_lower in t.get("artist", "").lower()
            ]
            st.caption(f"Showing {len(tracks)} matching tracks")

        page_size = 50
        total_pages = max(1, (len(tracks) + page_size - 1) // page_size)
        page = st.number_input("Page", 1, total_pages, 1)
        page_tracks = tracks[(page - 1) * page_size : page * page_size]

        for t in page_tracks:
            energy = t.get("energy")
            happiness = t.get("happiness")
            bpm = t.get("bpm")
            vibe = t.get("vibe", "")

            features = ""
            if energy is not None:
                features = (
                    f" | E:{energy:.2f} H:{happiness:.2f} BPM:{bpm}"
                    + (f" [{vibe}]" if vibe else "")
                )

            spotify_url = f"https://open.spotify.com/track/{t.get('id', '')}"
            st.markdown(f"[{t['title']}]({spotify_url}) \u2014 {t['artist']}{features}")


if __name__ == "__main__":
    main()
