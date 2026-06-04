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
    GEMINI_API_KEY,
    SPOTIFY_CLIENT_ID,
    SPOTIFY_CLIENT_SECRET,
    SPOTIFY_REDIRECT_URI,
)
from src.config_loader import PARAMS
from src.dashboard import can_view_dashboard, render_dashboard
from src.emotions import Emotion, get_emotion, list_emotions
from src.i18n import t, emotion_name
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
)

st.set_page_config(page_title="MindTune", page_icon="\U0001f3b5", layout="wide")


def _lang() -> str:
    """Return the current UI language code ('en' or 'ja')."""
    return st.session_state.get("lang", "en")


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
    L = _lang()

    # Language toggle on login page too
    lang_options = {"English": "en", "日本語": "ja"}
    lang_label = [k for k, v in lang_options.items() if v == L][0]
    selected = st.radio(
        t("language", L), list(lang_options.keys()),
        index=list(lang_options.keys()).index(lang_label),
        horizontal=True, key="login_lang_radio",
    )
    if lang_options[selected] != L:
        st.session_state["lang"] = lang_options[selected]
        st.rerun()

    st.title(t("app_title", L))
    st.caption(t("app_subtitle", L))

    # Dashboard link on login page (gated by can_view_dashboard)
    if can_view_dashboard():
        dash_label = t("dashboard_link", L)
        st.markdown(
            f'<a href="?page=dashboard" target="_self" style="'
            f'font-size:0.9em;color:#1DB954;text-decoration:none;'
            f'font-weight:600;">'
            f'\U0001f4ca {dash_label}</a>',
            unsafe_allow_html=True,
        )

    auth_err = st.session_state.pop("_auth_error", None)
    if auth_err:
        st.error(
            t("login_failed", L, error=auth_err) + "\n\n"
            + t("login_tip", L, uri=SPOTIFY_REDIRECT_URI)
        )

    st.markdown("---")
    st.subheader(t("welcome", L))
    st.write(t("welcome_desc", L))

    has_creds = SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(t("option_a_title", L))
        st.caption(t("option_a_caption", L))
        with st.form("login_form"):
            uid = st.text_input(
                t("spotify_user_id", L),
                placeholder="e.g. 31vyyplvkasqpb7cksrua3jep2q4",
                key="login_user_id_input",
            )
            submitted = st.form_submit_button(t("log_in", L))
            if submitted:
                if uid.strip():
                    st.session_state["user_id"] = uid.strip()
                    st.rerun()
                else:
                    st.error(t("enter_user_id_error", L))

    with col2:
        st.markdown(t("option_b_title", L))
        st.caption(t("option_b_caption", L))
        if has_creds:
            auth_url = get_auth_url(SPOTIFY_REDIRECT_URI)
            btn_text = t("connect_spotify", L)
            st.markdown(
                f'<a href="{auth_url}" target="_self" style="'
                'display:inline-block;padding:0.6em 1.2em;background:#1DB954;'
                'color:white;border-radius:24px;text-decoration:none;'
                f'font-weight:bold;margin-top:0.5em;">'
                f'{btn_text}</a>',
                unsafe_allow_html=True,
            )
        else:
            st.info(t("set_spotify_creds", L))


# ---------------------------------------------------------------------------
# Library import
# ---------------------------------------------------------------------------

def render_library_import(user_id: str) -> None:
    L = _lang()
    st.info(t("no_library", L, user_id=user_id))
    token = ensure_spotify_token()
    if not token:
        has_creds = SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET
        if has_creds:
            auth_url = get_auth_url(SPOTIFY_REDIRECT_URI)
            btn_text = t("connect_to_import", L)
            st.markdown(
                f'<a href="{auth_url}" target="_self" style="'
                'display:inline-block;padding:0.5em 1em;background:#1DB954;'
                'color:white;border-radius:24px;text-decoration:none;'
                f'font-weight:bold;">'
                f'{btn_text}</a>',
                unsafe_allow_html=True,
            )
        else:
            st.warning(t("set_creds_import", L))
        return

    user_name = st.session_state.get("spotify_user_name", user_id)
    st.caption(t("authenticated_as", L, name=user_name))

    if st.button(t("import_liked_songs", L), type="primary", key="import_lib"):
        progress = st.progress(0, text=t("fetching_liked_songs", L))
        status = st.empty()

        def on_progress(done: int, total: int) -> None:
            pct = done / total if total else 0
            progress.progress(pct, text=t("fetched_songs", L, done=done, total=total))
            status.text(f"Page {done // 50 + 1}...")

        try:
            tracks = fetch_liked_songs(token, progress_callback=on_progress)
            save_tracks(tracks, user_id)
            invalidate_cache(user_id)
            progress.progress(1.0, text=t("done", L))
            status.success(t("imported_tracks", L, count=len(tracks)))
            st.rerun()
        except SpotifyAPIError as exc:
            st.error(t("spotify_api_error", L, error=exc))
        except Exception as exc:
            st.error(t("import_failed", L, error=exc))


# ---------------------------------------------------------------------------
# 3D Circumplex visualisation
# ---------------------------------------------------------------------------

def render_circumplex_3d(
    current: Emotion | None = None,
    target: Emotion | None = None,
    v2_result: dict | None = None,
) -> go.Figure:
    """Draw the 3D V/E/T emotion space with transition path."""
    L = _lang()
    fig = go.Figure()

    emotions = list_emotions()

    # Per-emotion text positions to avoid overlapping labels
    _label_pos: dict[str, str] = {
        "Angry": "top left",
        "Fear": "middle left",
        "Anxious": "bottom left",
        "Restless": "top right",
        "Sad": "bottom left",
        "Melancholy": "bottom right",
        "Tired": "bottom center",
        "Calm": "top right",
        "Peaceful": "bottom right",
        "Focused": "top center",
        "Confident": "middle right",
        "Excited": "top right",
    }
    positions = [_label_pos.get(e.name, "top center") for e in emotions]

    fig.add_trace(
        go.Scatter3d(
            x=[e.V for e in emotions],
            y=[e.E for e in emotions],
            z=[e.T for e in emotions],
            mode="markers+text",
            text=[emotion_name(e.name, L) for e in emotions],
            textposition=positions,
            textfont=dict(size=12),
            marker=dict(size=7, color="#888"),
            name=t("legend_emotions", L),
            hovertemplate="%{text}<br>V: %{x:.1f}<br>E: %{y:.1f}<br>T: %{z:.1f}<extra></extra>",
        )
    )

    if current:
        fig.add_trace(
            go.Scatter3d(
                x=[current.V], y=[current.E], z=[current.T],
                mode="markers+text",
                text=[t("now_label", L, name=emotion_name(current.name, L))],
                textposition="top center",
                textfont=dict(size=13, color="#FF6B6B"),
                marker=dict(size=12, color="#FF6B6B", symbol="diamond"),
                name=t("legend_current", L),
            )
        )
    if target:
        fig.add_trace(
            go.Scatter3d(
                x=[target.V], y=[target.E], z=[target.T],
                mode="markers+text",
                text=[t("goal_label", L, name=emotion_name(target.name, L))],
                textposition="top center",
                textfont=dict(size=13, color="#4ECDC4"),
                marker=dict(size=12, color="#4ECDC4", symbol="diamond"),
                name=t("legend_target", L),
            )
        )

    if v2_result and current and target:
        stages = v2_result.get("stages", [])
        if stages:
            path_v = [current.V] + [s["V"] for s in stages]
            path_e = [current.E] + [s["E"] for s in stages]
            path_t = [current.T] + [s["T"] for s in stages]
            labels = [t("start", L)] + [
                t("stage_label", L, n=s["stage"], axis=s["lead_axis"]) for s in stages
            ]

            fig.add_trace(
                go.Scatter3d(
                    x=path_v, y=path_e, z=path_t,
                    mode="lines+markers",
                    line=dict(width=4, color="#FFD93D"),
                    marker=dict(size=4, color="#FFD93D"),
                    text=labels,
                    name=t("legend_path", L),
                    hovertemplate="%{text}<br>V: %{x:.2f}<br>E: %{y:.2f}<br>T: %{z:.2f}<extra></extra>",
                )
            )

        # Show selected tracks on the 3D plot (one trace per stage for clarity)
        track_items = v2_result.get("tracks", [])
        _stage_colors = [
            "#FF6B6B", "#FFD93D", "#6BCB77", "#4D96FF",
            "#9B59B6", "#E67E22", "#1ABC9C", "#E74C3C",
        ]
        for i, item in enumerate(track_items):
            tr = item["track"]
            stg = item["stage"]
            clr = _stage_colors[i % len(_stage_colors)]
            label = f"S{stg}: {tr.get('title', '?')[:25]}"
            fig.add_trace(
                go.Scatter3d(
                    x=[float(tr.get("V") or 0)],
                    y=[float(tr.get("E") or 0)],
                    z=[float(tr.get("T") or 0)],
                    mode="markers+text",
                    marker=dict(size=8, color=clr, symbol="circle",
                                opacity=0.9, line=dict(width=1, color="white")),
                    text=[label],
                    textposition="top center",
                    textfont=dict(size=9, color=clr),
                    name=label,
                    legendgroup="tracks",
                    showlegend=True,
                    hovertemplate=f"{label}<br>V: %{{x:.2f}}<br>E: %{{y:.2f}}<br>T: %{{z:.2f}}<extra></extra>",
                )
            )

    fig.update_layout(
        title=dict(text=t("chart_title", L), font=dict(size=18)),
        scene=dict(
            xaxis=dict(title=t("axis_valence", L), range=[-1.3, 1.3]),
            yaxis=dict(title=t("axis_energy", L), range=[-1.3, 1.3]),
            zaxis=dict(title=t("axis_tension", L), range=[-1.3, 1.3]),
            aspectmode="cube",
        ),
        height=800,
        showlegend=True,
        template="plotly_dark",
        margin=dict(l=0, r=0, t=40, b=0),
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
    """Show UI for building the v2 feature store (LLM extraction + correction)."""
    L = _lang()
    count = get_track_count(user_id)
    st.warning(t("features_not_estimated", L, count=count))

    if not GEMINI_API_KEY:
        st.error(t("set_llm_key", L))
        return

    st.info(t("estimation_info", L))

    if st.button(t("estimate_features", L), type="primary", key="build_fs"):
        with st.spinner(t("building_fs", L)):
            try:
                from src.feature_extraction.build_pipeline import build
                metrics = build(user_id=user_id)
                st.success(
                    t("fs_built", L,
                      count=metrics["n_tracks"],
                      matched=metrics.get("matched", 0),
                      total=metrics.get("total", 0))
                )
                st.rerun()
            except Exception as exc:
                st.error(t("build_failed", L, error=exc))


# ---------------------------------------------------------------------------
# Spotify Connect playback
# ---------------------------------------------------------------------------

def render_play_on_spotify(track_ids: list[str]) -> None:
    L = _lang()
    has_creds = SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET
    if not has_creds:
        st.info(t("set_creds_playback", L))
        return

    token = ensure_spotify_token()
    if not token:
        auth_url = get_auth_url(SPOTIFY_REDIRECT_URI)
        btn_text = t("connect_spotify_short", L)
        st.markdown(
            f'<a href="{auth_url}" target="_self" style="'
            'display:inline-block;padding:0.5em 1em;background:#1DB954;'
            'color:white;border-radius:24px;text-decoration:none;'
            f'font-weight:bold;">'
            f'{btn_text}</a>',
            unsafe_allow_html=True,
        )
        return

    if not track_ids:
        st.warning(t("no_track_ids", L))
        return

    user_name = st.session_state.get("spotify_user_name", "")
    st.caption(t("connected_as", L, name=user_name))

    try:
        devices = get_devices(token)
    except SpotifyAPIError as exc:
        if exc.status_code == 401:
            for key in ("spotify_token", "spotify_refresh", "spotify_user_id", "spotify_user_name"):
                st.session_state.pop(key, None)
            st.warning(t("session_expired", L))
            st.rerun()
        st.error(t("fetch_devices_error", L, error=exc))
        return

    if not devices:
        st.warning(t("no_devices", L))
        if st.button(t("refresh_devices", L), key="refresh_devices"):
            st.rerun()
        return

    device_labels = [f"{d['name']} ({d['type']})" for d in devices]
    selected_idx = st.selectbox(
        t("select_device", L),
        range(len(devices)),
        format_func=lambda i: device_labels[i],
        key="spotify_device",
    )
    selected_device = devices[selected_idx]

    if st.button(t("play_on_spotify", L), type="primary", key="play_spotify"):
        try:
            start_playback(token, track_ids, device_id=selected_device["id"])
            st.success(t("now_playing", L, device=selected_device["name"], count=len(track_ids)))
            st.balloons()
        except SpotifyAPIError as exc:
            if exc.status_code == 403:
                st.error(t("playback_403", L))
            elif exc.status_code == 404:
                st.error(t("playback_404", L))
            else:
                st.error(t("playback_failed", L, error=exc))
        except Exception as exc:
            st.error(t("playback_failed", L, error=exc))


# ---------------------------------------------------------------------------
# v2 Playlist display
# ---------------------------------------------------------------------------

def render_v2_playlist(result: dict) -> None:
    """Display the v2 generated playlist with stage breakdown."""
    L = _lang()
    for item in result.get("tracks", []):
        track = item["track"]
        stage = item["stage"]
        lead = item["lead_axis"]

        st.subheader(t("stage_heading", L, stage=stage, lead=lead))
        st.caption(
            t("stage_target_caption", L,
              v=item["target_V"], e=item["target_E"], tval=item["target_T"])
        )

        col1, col2 = st.columns([3, 2])
        with col1:
            spotify_url = f"https://open.spotify.com/track/{track.get('id', '')}"
            st.markdown(
                f"**[{track.get('title', '?')}]({spotify_url})** \u2014 {track.get('artist', '?')}"
            )
        with col2:
            tv = track.get("V", 0)
            te = track.get("E", 0)
            tt = track.get("T", 0)
            st.caption(
                f"V:{tv:.2f} | E:{te:.2f} | T:{tt:.2f}"
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
    L = _lang()
    st.subheader(t("rate_playlist", L))
    st.caption(t("rate_caption", L))

    col1, col2 = st.columns([2, 3])
    with col1:
        rating = st.slider(
            t("rate_slider", L),
            1, 5, 3,
            key="playlist_rating",
        )
    with col2:
        stars = "\u2b50" * rating + "\u2606" * (5 - rating)
        st.markdown(f"### {stars}")

    if st.button(t("submit_rating", L), key="submit_rating"):
        st.session_state["last_rating"] = {
            "source": result.get("source", ""),
            "target": result.get("target", ""),
            "rating": rating,
            "track_ids": result.get("all_track_ids", []),
        }
        st.success(t("rating_saved", L, rating=rating))
        st.caption(t("phase2_note", L))


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Evaluation mode: blind A/B/C comparison
# ---------------------------------------------------------------------------

def _render_eval_mode(
    L: str,
    current: "Emotion",
    target: "Emotion",
    current_name: str,
    target_name: str,
    user_id: str,
    K: int,
    N: int,
) -> None:
    """Render the blind A/B/C evaluation comparison."""
    token = ensure_spotify_token()

    if st.button(t("generate_eval", L), type="primary"):
        with st.spinner(t("running_eval", L)):
            results: dict[str, dict] = {}

            # 1. Dynamic (v2)
            from src.selection.assemble import recommend_v2
            results["dynamic"] = recommend_v2(
                source_label=current_name,
                target_label=target_name,
                user_id=user_id,
                K=K,
                N=N,
            )

            # 2. Linear baseline (linear interpolation + greedy nearest-neighbor)
            from src.evaluation import linear_baseline
            results["linear"] = linear_baseline(
                source_label=current_name,
                target_label=target_name,
                user_id=user_id,
                N=N,
            )

            # 3. Spotify Autoplay baseline
            if token:
                from src.evaluation import spotify_autoplay_baseline
                results["autoplay"] = spotify_autoplay_baseline(token, N)
            else:
                results["autoplay"] = {
                    "source": "autoplay", "target": "autoplay",
                    "method": "spotify_autoplay",
                    "error": t("eval_need_spotify", L),
                    "tracks": [],
                }

            # Assign blind labels
            from src.evaluation import assign_blind_labels, log_eval_session
            mapping = assign_blind_labels(["dynamic", "linear", "autoplay"])

            # Log
            log_eval_session(user_id, mapping, results)

            st.session_state["eval_results"] = results
            st.session_state["eval_mapping"] = mapping

    # Display results
    if "eval_results" in st.session_state and "eval_mapping" in st.session_state:
        results = st.session_state["eval_results"]
        mapping = st.session_state["eval_mapping"]

        st.info(t("eval_blind_info", L))

        for label in sorted(mapping.keys()):
            method = mapping[label]
            result = results.get(method, {})

            st.subheader(f"Method {label}")

            if "error" in result:
                st.warning(result["error"])
                continue

            tracks = result.get("tracks", [])
            if not tracks:
                st.warning(t("eval_no_tracks", L))
                continue

            for item in tracks:
                track = item["track"]
                spotify_url = f"https://open.spotify.com/track/{track.get('id', '')}"
                st.markdown(
                    f"**{item['stage']}. [{track.get('title', '?')}]({spotify_url})** "
                    f"\u2014 {track.get('artist', '?')}"
                )

            # Play button per method
            method_track_ids = [
                item["track"].get("id", "")
                for item in tracks if item["track"].get("id")
            ]
            if method_track_ids and token:
                if st.button(
                    t("play_method", L, label=label),
                    key=f"play_eval_{label}",
                ):
                    try:
                        devices = get_devices(token)
                        if devices:
                            start_playback(token, method_track_ids, devices[0]["id"])
                            st.success(t("playing_method", L, label=label))
                        else:
                            st.warning(t("no_device", L))
                    except SpotifyAPIError as exc:
                        st.error(str(exc))

            st.divider()

        # Toggle to reveal method mapping (hidden by default for blinding)
        if st.checkbox(t("show_mapping_toggle", L), value=False, key="show_mapping"):
            st.markdown("---")
            st.subheader(t("mapping_header", L))
            for label in sorted(mapping.keys()):
                method = mapping[label]
                st.markdown(f"- **Method {label}** = {method}")


# ---------------------------------------------------------------------------
# Main app (post-login)
# ---------------------------------------------------------------------------

def render_app(user_id: str) -> None:
    L = _lang()

    # Sidebar: language toggle (top)
    lang_options = {"English": "en", "日本語": "ja"}
    lang_label = [k for k, v in lang_options.items() if v == L][0]
    selected_lang = st.sidebar.radio(
        t("language", L), list(lang_options.keys()),
        index=list(lang_options.keys()).index(lang_label),
        horizontal=True, key="app_lang_radio",
    )
    if lang_options[selected_lang] != L:
        st.session_state["lang"] = lang_options[selected_lang]
        st.rerun()

    # Sidebar
    st.sidebar.markdown(t("user_label", L, user_id=user_id))
    if st.sidebar.button(t("log_out", L), key="logout_btn"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.title(t("main_title", L))
    st.caption(t("main_subtitle", L))

    track_count = get_track_count(user_id)
    if track_count == 0:
        render_library_import(user_id)
        st.stop()

    # Check feature store (v2 pipeline — all features come from parquet)
    has_store = _check_feature_store(user_id)
    fs_count = _feature_store_count(user_id) if has_store else 0

    if not has_store:
        render_feature_store_build(user_id)
        st.stop()

    # Sidebar: Emotion Settings
    st.sidebar.header(t("emotion_settings", L))
    emotions_list = list_emotions()
    emotion_names = [e.name for e in emotions_list]
    # Display names (translated) with mapping back to internal names
    display_names = [emotion_name(n, L) for n in emotion_names]

    current_idx = emotion_names.index("Anxious") if "Anxious" in emotion_names else 0
    target_idx = emotion_names.index("Calm") if "Calm" in emotion_names else 0

    current_sel = st.sidebar.selectbox(
        t("how_feeling", L), display_names, index=current_idx,
    )
    target_sel = st.sidebar.selectbox(
        t("how_want_feel", L), display_names, index=target_idx,
    )

    # Map display name back to internal name
    current_name = emotion_names[display_names.index(current_sel)]
    target_name = emotion_names[display_names.index(target_sel)]

    current = get_emotion(current_name)
    target = get_emotion(target_name)

    # Sidebar: Algorithm params
    st.sidebar.header(t("algo_settings", L))
    st.sidebar.caption(t("algo_caption", L))
    N = st.sidebar.slider(
        t("n_stages", L), 3, 12, PARAMS.N,
        help=t("n_stages_help", L),
    )
    K = st.sidebar.slider(
        t("k_candidates", L), 3, 20, PARAMS.K_default,
        help=t("k_candidates_help", L),
    )

    # Sidebar: Library management
    st.sidebar.markdown("---")
    st.sidebar.header(t("library_header", L))
    st.sidebar.caption(t("track_library_count", L, count=track_count))
    st.sidebar.caption(
        t("feature_store_count", L, count=fs_count) if has_store
        else t("feature_store_not_built", L)
    )
    token = ensure_spotify_token()
    if token:
        # Combined Refresh & Rebuild button
        if st.sidebar.button(t("refresh_and_rebuild", L), key="refresh_rebuild"):
            with st.sidebar:
                try:
                    with st.spinner(t("fetching_liked", L)):
                        tracks = fetch_liked_songs(token)
                        save_tracks(tracks, user_id)
                        invalidate_cache(user_id)
                    st.sidebar.info(t("updated_tracks", L, count=len(tracks)))

                    # Rebuild feature store (v2 pipeline)
                    with st.spinner(t("building_fs", L)):
                        from src.feature_extraction.build_pipeline import build
                        metrics = build(user_id=user_id)
                    st.sidebar.success(
                        t("fs_built", L,
                          count=metrics["n_tracks"],
                          matched=metrics.get("matched", 0),
                          total=metrics.get("total", 0))
                    )
                    st.rerun()
                except Exception as exc:
                    st.sidebar.error(t("refresh_failed", L, error=exc))
    else:
        has_creds = SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET
        if has_creds:
            auth_url = get_auth_url(SPOTIFY_REDIRECT_URI)
            btn_text = t("connect_spotify_refresh", L)
            st.sidebar.markdown(
                f'<a href="{auth_url}" target="_self" style="'
                'display:inline-block;padding:0.4em 0.8em;background:#1DB954;'
                'color:white;border-radius:16px;text-decoration:none;'
                f'font-size:0.85em;margin-top:0.3em;">'
                f'{btn_text}</a>',
                unsafe_allow_html=True,
            )

    # Clean Rebuild: wipe all cache then re-download + rebuild from scratch
    if st.sidebar.button(t("clean_rebuild", L), key="clean_rebuild"):
        with st.sidebar:
            try:
                from pathlib import Path as _P
                _cache = _P(__file__).resolve().parent.parent / "cache"
                # Delete user-specific cache files
                # Keep llm_raw (expensive LLM extraction) — only delete
                # derived files that are cheap to regenerate
                for pattern in [
                    f"feature_store_{user_id}.parquet",
                    f"zscore_params_{user_id}.json",
                    f"labeled_tracks_{user_id}.parquet",
                    "labeled_tracks.parquet",  # legacy
                ]:
                    p = _cache / pattern
                    if p.exists():
                        p.unlink()
                # Clear tracks.json so features are re-estimated
                _user_tracks = _P(__file__).resolve().parent / "data" / "users" / user_id / "tracks.json"
                if _user_tracks.exists():
                    _user_tracks.unlink()
                invalidate_cache(user_id)
                st.sidebar.info(t("cache_cleared", L))

                # Re-download from Spotify if connected
                _token = ensure_spotify_token()
                if _token:
                    with st.spinner(t("fetching_liked", L)):
                        tracks = fetch_liked_songs(_token)
                        save_tracks(tracks, user_id)
                        invalidate_cache(user_id)
                    st.sidebar.info(t("updated_tracks", L, count=len(tracks)))

                    # Rebuild feature store (v2 pipeline)
                    with st.spinner(t("building_fs", L)):
                        from src.feature_extraction.build_pipeline import build
                        metrics = build(user_id=user_id)
                    st.sidebar.success(
                        t("fs_built", L,
                          count=metrics["n_tracks"],
                          matched=metrics.get("matched", 0),
                          total=metrics.get("total", 0))
                    )
                else:
                    st.sidebar.warning(t("clean_rebuild_no_spotify", L))
                st.rerun()
            except Exception as exc:
                st.sidebar.error(t("clean_rebuild_failed", L, error=exc))

    # Reset Ratings button for testing
    if st.sidebar.button(t("reset_ratings", L), key="reset_ratings"):
        st.session_state.pop("last_rating", None)
        st.session_state.pop("v2_result", None)
        st.session_state.pop("eval_results", None)
        st.session_state.pop("eval_mapping", None)
        st.sidebar.success(t("ratings_reset", L))
        st.rerun()

    # Sidebar: Evaluation section
    st.sidebar.markdown("---")
    st.sidebar.header(t("eval_header", L))
    st.sidebar.caption(t("eval_caption", L))

    # Eval mode toggle (blind A/B/C comparison)
    eval_mode = st.sidebar.checkbox(
        t("eval_mode_toggle", L), value=False, key="eval_mode",
        help=t("eval_mode_help", L),
    )

    # Main tabs
    tab1, tab2, tab3 = st.tabs(
        [t("tab_generate", L), t("tab_circumplex", L), t("tab_library", L)]
    )

    with tab1:
        if current_name == target_name:
            st.info(t("same_emotion", L))
        elif eval_mode:
            # --- Evaluation mode: blind A/B/C comparison ---
            _render_eval_mode(
                L, current, target, current_name, target_name,
                user_id, K, N,
            )
        else:
            # v2 Dynamic mode
            if st.button(t("generate_v2", L), type="primary"):
                with st.spinner(t("running_v2", L)):
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
                    src_disp = emotion_name(result["source"], L)
                    tgt_disp = emotion_name(result["target"], L)
                    order_str = " \u2192 ".join(result.get("axis_order", []))
                    st.success(
                        t("result_summary", L,
                          source=src_disp, target=tgt_disp,
                          order=order_str, alloc=result.get("stage_alloc", {}))
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
                    with st.expander(t("progress_matrix", L)):
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

        st.subheader(t("emotion_details", L))
        cols = st.columns(3)
        for i, em in enumerate(list_emotions()):
            with cols[i % 3]:
                em_disp = emotion_name(em.name, L)
                st.markdown(
                    f"**{em_disp}** \u2014 V:{em.V:.1f}, E:{em.E:.1f}, T:{em.T:.1f}"
                )

    with tab3:
        st.subheader(t("track_library_title", L, count=track_count))
        tracks = load_tracks(user_id)

        search = st.text_input(t("search_tracks", L), "")
        if search:
            search_lower = search.lower()
            tracks = [
                tr for tr in tracks
                if search_lower in tr.get("title", "").lower()
                or search_lower in tr.get("artist", "").lower()
            ]
            st.caption(t("showing_matches", L, count=len(tracks)))

        page_size = 50
        total_pages = max(1, (len(tracks) + page_size - 1) // page_size)
        page = st.number_input(t("page", L), 1, total_pages, 1)
        page_tracks = tracks[(page - 1) * page_size : page * page_size]

        for tr in page_tracks:
            features = ""
            v = tr.get("V")
            e = tr.get("E")
            tval = tr.get("T")
            if v is not None and e is not None:
                features = f" | V:{v:.2f} E:{e:.2f}"
                if tval is not None:
                    features += f" T:{tval:.2f}"
            vibe = tr.get("vibe", "")
            if vibe:
                features += f" [{vibe}]"

            spotify_url = f"https://open.spotify.com/track/{tr.get('id', '')}"
            st.markdown(f"[{tr['title']}]({spotify_url}) \u2014 {tr['artist']}{features}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    handle_spotify_callback()
    user_id = get_user_id()

    # Route: dashboard page (accessible without login)
    if st.query_params.get("page") == "dashboard":
        if can_view_dashboard(user_id):
            render_dashboard()
        else:
            st.error("Access denied.")
        return

    if user_id:
        render_app(user_id)
    else:
        render_login()


if __name__ == "__main__":
    main()
