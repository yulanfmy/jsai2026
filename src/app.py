"""MindTune v2 — Emotion-Transition Music Recommendation System.

3D emotion model (V/E/T) with dynamic path planning,
Viterbi DP track selection, and Spotify Connect playback.

Multi-user: each Spotify user's track library is stored and queried
independently so recommendations only use the current user's music.
"""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import plotly.graph_objects as go
import streamlit as st

from src.config import (
    OPENAI_API_KEY,
    SPOTIFY_CLIENT_ID,
    SPOTIFY_CLIENT_SECRET,
    SPOTIFY_REDIRECT_URI,
)
from src.config_loader import PARAMS
from src.emotions import Emotion, get_emotion, list_emotions
from src.spotify import (
    SpotifyAPIError,
    exchange_code_for_token,
    fetch_liked_songs,
    get_auth_url,
    get_current_user,
    get_devices,
    refresh_access_token,
    start_playback,
)
from src.tracks import (
    get_all_vibes,
    get_track_count,
    invalidate_cache,
    load_tracks,
    save_tracks,
    tracks_have_features,
)

st.set_page_config(page_title="MindTune", page_icon="\U0001f3b5", layout="wide")


# ---------------------------------------------------------------------------
# Session / auth helpers
# ---------------------------------------------------------------------------

def handle_spotify_callback() -> None:
    params = st.query_params
    code = params.get("code")
    if not code:
        return
    if "spotify_token" in st.session_state:
        st.query_params.clear()
        return
    try:
        token_data = exchange_code_for_token(code, SPOTIFY_REDIRECT_URI)
        st.session_state["spotify_token"] = token_data["access_token"]
        st.session_state["spotify_refresh"] = token_data.get("refresh_token", "")
        user = get_current_user(token_data["access_token"])
        st.session_state["spotify_user_id"] = user["id"]
        st.session_state["spotify_user_name"] = user.get("display_name", user["id"])
        if "user_id" not in st.session_state:
            st.session_state["user_id"] = user["id"]
    except Exception as exc:
        st.session_state.pop("spotify_token", None)
        st.session_state["_auth_error"] = str(exc)
    st.query_params.clear()


def ensure_spotify_token() -> str | None:
    token = st.session_state.get("spotify_token")
    if not token:
        return None
    try:
        get_current_user(token)
        return token
    except Exception:
        pass
    refresh = st.session_state.get("spotify_refresh")
    if not refresh:
        st.session_state.pop("spotify_token", None)
        return None
    try:
        token_data = refresh_access_token(refresh)
        st.session_state["spotify_token"] = token_data["access_token"]
        if "refresh_token" in token_data:
            st.session_state["spotify_refresh"] = token_data["refresh_token"]
        return token_data["access_token"]
    except Exception:
        st.session_state.pop("spotify_token", None)
        return None


def get_user_id() -> str | None:
    return st.session_state.get("user_id")


# ---------------------------------------------------------------------------
# Login / welcome screen
# ---------------------------------------------------------------------------

def render_login() -> None:
    st.title("\U0001f3b5 MindTune")
    st.caption(
        "Emotion-transition music recommendation using 3D emotion model "
        "(V/E/T) with dynamic path planning and Viterbi DP"
    )

    auth_err = st.session_state.pop("_auth_error", None)
    if auth_err:
        st.error(
            f"Spotify login failed: {auth_err}\n\n"
            f"**Tip:** Make sure you open this app at the same URL as the "
            f"redirect URI: `{SPOTIFY_REDIRECT_URI}`"
        )

    st.markdown("---")
    st.subheader("Welcome! Please log in to get started.")
    st.write(
        "MindTune generates personalised emotion-transition playlists from "
        "**your own** Spotify library. Each user's music is kept private."
    )

    has_creds = SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Option A: Enter your Spotify User ID")
        st.caption("Use this if your library has already been imported.")
        with st.form("login_form"):
            uid = st.text_input(
                "Spotify User ID",
                placeholder="e.g. 31vyyplvkasqpb7cksrua3jep2q4",
                key="login_user_id_input",
            )
            submitted = st.form_submit_button("Log in")
            if submitted:
                if uid.strip():
                    st.session_state["user_id"] = uid.strip()
                    st.rerun()
                else:
                    st.error("Please enter your Spotify User ID.")

    with col2:
        st.markdown("#### Option B: Connect with Spotify")
        st.caption(
            "Log in with your Spotify account to automatically identify "
            "yourself and import your library."
        )
        if has_creds:
            auth_url = get_auth_url(SPOTIFY_REDIRECT_URI)
            st.markdown(
                f'<a href="{auth_url}" target="_self" style="'
                'display:inline-block;padding:0.6em 1.2em;background:#1DB954;'
                'color:white;border-radius:24px;text-decoration:none;'
                'font-weight:bold;margin-top:0.5em;">'
                '\U0001f3a7 Connect with Spotify</a>',
                unsafe_allow_html=True,
            )
        else:
            st.info(
                "Set `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` in "
                "`.env` to enable Spotify login."
            )


# ---------------------------------------------------------------------------
# Library import
# ---------------------------------------------------------------------------

def render_library_import(user_id: str) -> None:
    st.info(
        f"No track library found for user **{user_id}**. "
        "Import your Spotify liked songs to get started."
    )
    token = ensure_spotify_token()
    if not token:
        has_creds = SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET
        if has_creds:
            auth_url = get_auth_url(SPOTIFY_REDIRECT_URI)
            st.markdown(
                f'<a href="{auth_url}" target="_self" style="'
                'display:inline-block;padding:0.5em 1em;background:#1DB954;'
                'color:white;border-radius:24px;text-decoration:none;'
                'font-weight:bold;">'
                '\U0001f3a7 Connect to Spotify to import your library</a>',
                unsafe_allow_html=True,
            )
        else:
            st.warning(
                "Set Spotify credentials in `.env` to enable automatic import."
            )
        return

    user_name = st.session_state.get("spotify_user_name", user_id)
    st.caption(f"Authenticated as **{user_name}**")

    if st.button("Import my Spotify liked songs", type="primary", key="import_lib"):
        progress = st.progress(0, text="Fetching liked songs from Spotify...")
        status = st.empty()

        def on_progress(done: int, total: int) -> None:
            pct = done / total if total else 0
            progress.progress(pct, text=f"Fetched {done}/{total} songs...")
            status.text(f"Page {done // 50 + 1}...")

        try:
            tracks = fetch_liked_songs(token, progress_callback=on_progress)
            save_tracks(tracks, user_id)
            invalidate_cache(user_id)
            progress.progress(1.0, text="Done!")
            status.success(f"Imported {len(tracks)} tracks into your library.")
            st.rerun()
        except SpotifyAPIError as exc:
            st.error(f"Spotify API error: {exc}")
        except Exception as exc:
            st.error(f"Import failed: {exc}")


# ---------------------------------------------------------------------------
# 3D Circumplex visualisation
# ---------------------------------------------------------------------------

def render_circumplex_3d(
    current: Emotion | None = None,
    target: Emotion | None = None,
    v2_result: dict | None = None,
) -> go.Figure:
    """Draw the 3D V/E/T emotion space with transition path."""
    fig = go.Figure()

    emotions = list_emotions()
    fig.add_trace(
        go.Scatter3d(
            x=[e.V for e in emotions],
            y=[e.E for e in emotions],
            z=[e.T for e in emotions],
            mode="markers+text",
            text=[e.name for e in emotions],
            textposition="top center",
            marker=dict(size=5, color="#888"),
            name="Emotions",
            hovertemplate="%{text}<br>V: %{x:.1f}<br>E: %{y:.1f}<br>T: %{z:.1f}<extra></extra>",
        )
    )

    if current:
        fig.add_trace(
            go.Scatter3d(
                x=[current.V], y=[current.E], z=[current.T],
                mode="markers+text",
                text=[f"NOW: {current.name}"],
                textposition="top center",
                marker=dict(size=10, color="#FF6B6B", symbol="diamond"),
                name="Current",
            )
        )
    if target:
        fig.add_trace(
            go.Scatter3d(
                x=[target.V], y=[target.E], z=[target.T],
                mode="markers+text",
                text=[f"GOAL: {target.name}"],
                textposition="top center",
                marker=dict(size=10, color="#4ECDC4", symbol="diamond"),
                name="Target",
            )
        )

    if v2_result and current and target:
        stages = v2_result.get("stages", [])
        if stages:
            path_v = [current.V] + [s["V"] for s in stages]
            path_e = [current.E] + [s["E"] for s in stages]
            path_t = [current.T] + [s["T"] for s in stages]
            labels = ["Start"] + [f"Stage {s['stage']} [{s['lead_axis']}]" for s in stages]

            fig.add_trace(
                go.Scatter3d(
                    x=path_v, y=path_e, z=path_t,
                    mode="lines+markers",
                    line=dict(width=4, color="#FFD93D"),
                    marker=dict(size=4, color="#FFD93D"),
                    text=labels,
                    name="Transition Path",
                    hovertemplate="%{text}<br>V: %{x:.2f}<br>E: %{y:.2f}<br>T: %{z:.2f}<extra></extra>",
                )
            )

        # Show selected tracks on the 3D plot
        track_items = v2_result.get("tracks", [])
        if track_items:
            tv = [item["track"].get("V", 0) for item in track_items]
            te = [item["track"].get("E", 0) for item in track_items]
            tt = [item["track"].get("T", 0) for item in track_items]
            t_labels = [f"S{item['stage']}: {item['track'].get('title', '?')[:25]}" for item in track_items]

            fig.add_trace(
                go.Scatter3d(
                    x=tv, y=te, z=tt,
                    mode="markers",
                    marker=dict(size=6, color="#4ECDC4", symbol="circle", opacity=0.8),
                    text=t_labels,
                    name="Selected Tracks",
                    hovertemplate="%{text}<br>V: %{x:.2f}<br>E: %{y:.2f}<br>T: %{z:.2f}<extra></extra>",
                )
            )

    fig.update_layout(
        title="3D Emotion Space (V/E/T)",
        scene=dict(
            xaxis=dict(title="Valence (V)", range=[-1.2, 1.2]),
            yaxis=dict(title="Energy Arousal (E)", range=[-1.2, 1.2]),
            zaxis=dict(title="Tension Arousal (T)", range=[-1.2, 1.2]),
        ),
        height=600,
        showlegend=True,
        template="plotly_dark",
    )
    return fig


# ---------------------------------------------------------------------------
# Feature store check
# ---------------------------------------------------------------------------

def _check_feature_store(user_id: str) -> bool:
    """Check if the feature store exists for this user."""
    store_path = Path(__file__).resolve().parent.parent / "cache" / f"feature_store_{user_id}.parquet"
    global_store = Path(__file__).resolve().parent.parent / "cache" / "feature_store.parquet"
    return store_path.exists() or global_store.exists()


def _feature_store_count(user_id: str) -> int:
    """Count tracks in the feature store."""
    try:
        from src.feature_store import get_all_for_user
        return len(get_all_for_user(user_id))
    except Exception:
        return 0


def render_feature_store_build(user_id: str) -> None:
    """Show UI for building the feature store from v1 tracks."""
    st.warning(
        "The v2 feature store has not been built for this user yet. "
        "Build it to enable v2 playlist generation."
    )
    st.info(
        "This will bootstrap v2 features (V/E/T + arc vectors) from the "
        "existing v1 track features and build the feature store. This takes "
        "about 30 seconds."
    )

    if st.button("Build Feature Store", type="primary", key="build_fs"):
        with st.spinner("Building feature store..."):
            try:
                from src.feature_extraction.build_pipeline import build
                metrics = build(user_id=user_id)
                st.success(
                    f"Feature store built with {metrics['n_tracks']} tracks! "
                    f"Zenodo coverage: {metrics.get('matched', 0)}/{metrics.get('total', 0)}"
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Build failed: {exc}")


# ---------------------------------------------------------------------------
# Feature estimation (v1 fallback)
# ---------------------------------------------------------------------------

def render_feature_estimation_ui(user_id: str) -> None:
    st.warning(
        f"Your track library has {get_track_count(user_id)} tracks but audio features "
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
        "features for each track based on its metadata."
    )

    if st.button("Estimate Features", type="primary"):
        from src.feature_estimator import estimate_batch
        tracks = load_tracks(user_id)
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

        save_tracks(updated, user_id)
        invalidate_cache(user_id)

        newly_done = sum(1 for t in updated if "energy" in t)
        progress_bar.progress(1.0, text="Done!")
        status.success(f"Estimated features for {newly_done}/{len(updated)} tracks.")
        st.rerun()


# ---------------------------------------------------------------------------
# Spotify Connect playback
# ---------------------------------------------------------------------------

def render_play_on_spotify(track_ids: list[str]) -> None:
    has_creds = SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET
    if not has_creds:
        st.info("Set Spotify credentials in `.env` to enable playback.")
        return

    token = ensure_spotify_token()
    if not token:
        auth_url = get_auth_url(SPOTIFY_REDIRECT_URI)
        st.markdown(
            f'<a href="{auth_url}" target="_self" style="'
            'display:inline-block;padding:0.5em 1em;background:#1DB954;'
            'color:white;border-radius:24px;text-decoration:none;'
            'font-weight:bold;">'
            '\U0001f3a7 Connect to Spotify</a>',
            unsafe_allow_html=True,
        )
        return

    if not track_ids:
        st.warning("No tracks with Spotify IDs found.")
        return

    user_name = st.session_state.get("spotify_user_name", "")
    st.caption(f"Connected to Spotify as **{user_name}**")

    try:
        devices = get_devices(token)
    except SpotifyAPIError as exc:
        if exc.status_code == 401:
            for key in ("spotify_token", "spotify_refresh", "spotify_user_id", "spotify_user_name"):
                st.session_state.pop(key, None)
            st.warning("Spotify session expired. Please reconnect.")
            st.rerun()
        st.error(f"Could not fetch devices: {exc}")
        return

    if not devices:
        st.warning(
            "No active Spotify devices found. "
            "Open the Spotify app on your phone or computer first."
        )
        if st.button("\U0001f504 Refresh devices", key="refresh_devices"):
            st.rerun()
        return

    device_labels = [f"{d['name']} ({d['type']})" for d in devices]
    selected_idx = st.selectbox(
        "Select device",
        range(len(devices)),
        format_func=lambda i: device_labels[i],
        key="spotify_device",
    )
    selected_device = devices[selected_idx]

    if st.button("\u25b6 Play on Spotify", type="primary", key="play_spotify"):
        try:
            start_playback(token, track_ids, device_id=selected_device["id"])
            st.success(
                f"Now playing on **{selected_device['name']}** \u2014 "
                f"{len(track_ids)} tracks queued!"
            )
            st.balloons()
        except SpotifyAPIError as exc:
            if exc.status_code == 403:
                st.error("Playback failed (403). Spotify Premium required.")
            elif exc.status_code == 404:
                st.error("Device not found. Reopen the Spotify app and try again.")
            else:
                st.error(f"Playback failed: {exc}")
        except Exception as exc:
            st.error(f"Playback failed: {exc}")


# ---------------------------------------------------------------------------
# v2 Playlist display
# ---------------------------------------------------------------------------

def render_v2_playlist(result: dict) -> None:
    """Display the v2 generated playlist with stage breakdown."""
    for item in result.get("tracks", []):
        track = item["track"]
        stage = item["stage"]
        lead = item["lead_axis"]

        st.subheader(f"Stage {stage} \u2014 Lead axis: {lead}")
        st.caption(
            f"Target \u2014 V: {item['target_V']:.2f} | "
            f"E: {item['target_E']:.2f} | T: {item['target_T']:.2f}"
        )

        col1, col2 = st.columns([3, 2])
        with col1:
            spotify_url = f"https://open.spotify.com/track/{track.get('id', '')}"
            st.markdown(
                f"**[{track.get('title', '?')}]({spotify_url})** \u2014 {track.get('artist', '?')}"
            )
        with col2:
            v = track.get("V", 0)
            e = track.get("E", 0)
            t = track.get("T", 0)
            st.caption(
                f"V:{v:.2f} | E:{e:.2f} | T:{t:.2f}"
                + (f" | {track.get('vibe', '')}" if track.get("vibe") else "")
            )
        st.divider()


# ---------------------------------------------------------------------------
# 5-star rating widget (Phase 2 placeholder)
# ---------------------------------------------------------------------------

def render_rating_widget(result: dict) -> None:
    """Show a 5-star rating widget for the generated playlist.

    Phase 2 placeholder — captures the rating but does NOT update the model yet.
    """
    st.subheader("Rate this playlist")
    st.caption("Your feedback will help improve future recommendations (Phase 2).")

    col1, col2 = st.columns([2, 3])
    with col1:
        rating = st.slider(
            "How well does this playlist match your emotional transition?",
            1, 5, 3,
            key="playlist_rating",
        )
    with col2:
        stars = "\u2b50" * rating + "\u2606" * (5 - rating)
        st.markdown(f"### {stars}")

    if st.button("Submit Rating", key="submit_rating"):
        st.session_state["last_rating"] = {
            "source": result.get("source", ""),
            "target": result.get("target", ""),
            "rating": rating,
            "track_ids": result.get("all_track_ids", []),
        }
        st.success(f"Rating saved ({rating}/5). Thank you!")
        st.caption("Note: Phase 2 bandit learning is not yet active.")


# ---------------------------------------------------------------------------
# Main app (post-login)
# ---------------------------------------------------------------------------

def render_app(user_id: str) -> None:
    # Sidebar
    st.sidebar.markdown(f"**User:** `{user_id}`")
    if st.sidebar.button("Log out", key="logout_btn"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.title("\U0001f3b5 MindTune v2")
    st.caption(
        "3D emotion model (V/E/T) with dynamic path planning, "
        "Viterbi DP track selection, and Spotify Connect playback"
    )

    track_count = get_track_count(user_id)
    if track_count == 0:
        render_library_import(user_id)
        st.stop()

    has_features = tracks_have_features(user_id)
    if not has_features:
        render_feature_estimation_ui(user_id)
        st.divider()
        st.info("You can still explore the emotion model below while features are being set up.")

    # Check feature store
    has_store = _check_feature_store(user_id)
    fs_count = _feature_store_count(user_id) if has_store else 0

    # Sidebar: Emotion Settings
    st.sidebar.header("Emotion Settings")
    emotion_names = [e.name for e in list_emotions()]

    current_name = st.sidebar.selectbox(
        "How are you feeling now?",
        emotion_names,
        index=emotion_names.index("Anxious") if "Anxious" in emotion_names else 0,
    )
    target_name = st.sidebar.selectbox(
        "How do you want to feel?",
        emotion_names,
        index=emotion_names.index("Calm") if "Calm" in emotion_names else 0,
    )

    current = get_emotion(current_name)
    target = get_emotion(target_name)

    # Sidebar: Algorithm params
    st.sidebar.header("Algorithm Settings")
    st.sidebar.caption("Dynamic path planning (v2)")
    N = st.sidebar.slider(
        "Number of stages (N)", 3, 12, PARAMS.N,
        help="How many stages in the emotion transition path"
    )
    K = st.sidebar.slider(
        "Candidates per stage (K)", 3, 20, PARAMS.K_default,
        help="Higher K = more candidate tracks considered per stage"
    )

    # Collapsible v1 Linear strategy (debug only)
    with st.sidebar.expander("v1 Linear strategy (debug)"):
        st.caption("Legacy 2D strategy from v1. Use for comparison only.")
        v1_enabled = st.checkbox("Enable v1 mode", value=False, key="v1_mode")

    # Sidebar: Library management
    st.sidebar.markdown("---")
    st.sidebar.header("Library")
    st.sidebar.caption(f"Track library: {track_count} tracks")
    st.sidebar.caption(
        f"Feature store: {fs_count} tracks" if has_store
        else "Feature store: not built"
    )
    token = ensure_spotify_token()
    if token:
        if st.sidebar.button("\U0001f504 Refresh Library from Spotify", key="refresh_lib"):
            with st.sidebar:
                with st.spinner("Fetching liked songs..."):
                    try:
                        tracks = fetch_liked_songs(token)
                        save_tracks(tracks, user_id)
                        invalidate_cache(user_id)
                        st.sidebar.success(f"Updated! {len(tracks)} tracks imported.")
                        # Rebuild feature store automatically
                        from src.feature_extraction.build_pipeline import build
                        build(user_id=user_id)
                        st.rerun()
                    except Exception as exc:
                        st.sidebar.error(f"Refresh failed: {exc}")
    else:
        has_creds = SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET
        if has_creds:
            auth_url = get_auth_url(SPOTIFY_REDIRECT_URI)
            st.sidebar.markdown(
                f'<a href="{auth_url}" target="_self" style="'
                'display:inline-block;padding:0.4em 0.8em;background:#1DB954;'
                'color:white;border-radius:16px;text-decoration:none;'
                'font-size:0.85em;margin-top:0.3em;">'
                '\U0001f3a7 Connect Spotify to refresh</a>',
                unsafe_allow_html=True,
            )

    # Main tabs
    tab1, tab2, tab3 = st.tabs(
        ["\U0001f3b6 Generate Playlist", "\U0001f4ca 3D Circumplex", "\U0001f4c0 Track Library"]
    )

    with tab1:
        if current_name == target_name:
            st.info("Select different emotions for current and target.")
        elif not has_features and not has_store:
            st.info("Run feature estimation or build the feature store first.")
        elif not has_store:
            render_feature_store_build(user_id)
        else:
            v1_mode = st.session_state.get("v1_mode", False)

            if v1_mode:
                # Legacy v1 mode
                if st.button("Generate Playlist (v1 Linear)", type="secondary"):
                    from src.playlist import generate_playlist
                    from src.strategies import get_strategy_description
                    playlist = generate_playlist(
                        current=current,
                        target=target,
                        user_id=user_id,
                        strategy_name="Linear",
                        tracks_per_phase=3,
                    )
                    st.session_state["v1_playlist"] = playlist

                if "v1_playlist" in st.session_state:
                    playlist = st.session_state["v1_playlist"]
                    st.warning("v1 Linear mode (for comparison only)")
                    for i, pp in enumerate(playlist.phases):
                        phase = pp.phase
                        st.subheader(f"Phase {i + 1}: {phase.label}")
                        for j, track in enumerate(pp.tracks, 1):
                            spotify_url = f"https://open.spotify.com/track/{track.get('id', '')}"
                            st.markdown(f"**{j}. [{track['title']}]({spotify_url})** \u2014 {track['artist']}")
                        st.divider()
            else:
                # v2 Dynamic mode
                if st.button("Generate Playlist", type="primary"):
                    with st.spinner("Running dynamic path planning + Viterbi DP..."):
                        from src.selection.assemble import recommend_v2
                        result = recommend_v2(
                            source_label=current_name,
                            target_label=target_name,
                            user_id=user_id,
                            K=K,
                            N=N,
                        )
                        st.session_state["v2_result"] = result

                if "v2_result" in st.session_state:
                    result = st.session_state["v2_result"]
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success(
                            f"{result['source']} \u2192 {result['target']} | "
                            f"Axis order: {' \u2192 '.join(result.get('axis_order', []))} | "
                            f"Stages: {result.get('stage_alloc', {})}"
                        )
                        render_v2_playlist(result)

                        # Play on Spotify
                        track_ids = [
                            item["track"].get("id", "")
                            for item in result.get("tracks", [])
                            if item["track"].get("id")
                        ]
                        render_play_on_spotify(track_ids)

                        # Rating widget
                        render_rating_widget(result)

                        # Show progress matrix
                        with st.expander("Progress Matrix"):
                            pm = result.get("progress_matrix", [])
                            if pm:
                                import pandas as pd
                                axes = list(pm[0].keys()) if pm else []
                                data = [[row.get(a, 0) for a in axes] for row in pm]
                                df = pd.DataFrame(data, columns=axes, index=[f"Stage {i+1}" for i in range(len(pm))])
                                st.dataframe(df.style.format("{:.4f}"), use_container_width=True)

                        # 3D chart inline
                        st.plotly_chart(
                            render_circumplex_3d(current, target, result),
                            use_container_width=True,
                            key="playlist_circumplex",
                        )

    with tab2:
        v2_result = st.session_state.get("v2_result")
        fig = render_circumplex_3d(current, target, v2_result)
        st.plotly_chart(fig, use_container_width=True, key="tab_circumplex")

        st.subheader("Emotion Details (V/E/T)")
        cols = st.columns(3)
        for i, em in enumerate(list_emotions()):
            with cols[i % 3]:
                st.markdown(
                    f"**{em.name}** \u2014 V:{em.V:.1f}, E:{em.E:.1f}, T:{em.T:.1f}"
                )

    with tab3:
        st.subheader(f"Track Library ({track_count} tracks)")
        tracks = load_tracks(user_id)

        search = st.text_input("Search tracks", "")
        if search:
            search_lower = search.lower()
            tracks = [
                t for t in tracks
                if search_lower in t.get("title", "").lower()
                or search_lower in t.get("artist", "").lower()
            ]
            st.caption(f"Showing {len(tracks)} matching tracks")

        page_size = 50
        total_pages = max(1, (len(tracks) + page_size - 1) // page_size)
        page = st.number_input("Page", 1, total_pages, 1)
        page_tracks = tracks[(page - 1) * page_size : page * page_size]

        for t in page_tracks:
            features = ""
            v = t.get("V", t.get("energy"))
            e = t.get("E")
            tval = t.get("T")
            if v is not None and e is not None:
                features = f" | V:{v:.2f} E:{e:.2f}"
                if tval is not None:
                    features += f" T:{tval:.2f}"
            elif t.get("energy") is not None:
                features = (
                    f" | E:{t['energy']:.2f} H:{t.get('happiness', 0):.2f}"
                    f" BPM:{t.get('bpm', '?')}"
                )
            vibe = t.get("vibe", "")
            if vibe:
                features += f" [{vibe}]"

            spotify_url = f"https://open.spotify.com/track/{t.get('id', '')}"
            st.markdown(f"[{t['title']}]({spotify_url}) \u2014 {t['artist']}{features}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    handle_spotify_callback()
    user_id = get_user_id()
    if user_id:
        render_app(user_id)
    else:
        render_login()


if __name__ == "__main__":
    main()
