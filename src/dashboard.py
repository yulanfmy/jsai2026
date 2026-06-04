"""Usage Dashboard — read-only system-health / management view.

Shows per-user library stats, Zenodo coverage, data quality, and
usage metrics. Currently visible to all logged-in users (family use).

To make admin-only later:
    Change ``can_view_dashboard()`` to check ``user.is_admin`` or
    a server-side allowlist.  No other code changes needed.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

from src.i18n import t

_DATA_DIR = Path(__file__).resolve().parent / "data" / "users"
_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
_EVAL_LOG_DIR = Path("cache/eval_logs")

# Sub-features required by the correction model
_REQUIRED_SUB_FEATURES = [
    "mode_major_conf", "lyric_sentiment", "vocal_brightness", "chord_complexity",
]

LOW_COVERAGE_THRESHOLD = 0.50  # flag users below 50%


def can_view_dashboard(user_id: str | None = None) -> bool:
    """Gate access to the dashboard.

    Currently returns True for all users (family use).
    To make admin-only later, change to:
        return user_id in ADMIN_ALLOWLIST
    or:
        return user.is_admin
    No other code changes needed — both the route and the link use this.
    """
    return True


def _discover_users() -> list[str]:
    """Return list of user IDs that have used the app (have a data directory)."""
    if not _DATA_DIR.exists():
        return []
    return sorted(
        d.name for d in _DATA_DIR.iterdir()
        if d.is_dir() and (d / "tracks.json").exists()
    )


def _load_user_tracks(user_id: str) -> list[dict]:
    """Load a user's track library from tracks.json."""
    path = _DATA_DIR / user_id / "tracks.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _load_feature_store(user_id: str) -> pd.DataFrame | None:
    """Load a user's feature store parquet if it exists."""
    path = _CACHE_DIR / f"feature_store_{user_id}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _count_eval_logs(user_id: str) -> int:
    """Count evaluation log files (playlists generated) for a user."""
    if not _EVAL_LOG_DIR.exists():
        return 0
    return sum(1 for f in _EVAL_LOG_DIR.glob(f"eval_{user_id}_*.json"))


def _safe_display_name(user_id: str) -> str:
    """Create a privacy-safe display name from user ID.

    Shows first 6 + last 4 chars with ellipsis in between.
    """
    if len(user_id) <= 12:
        return user_id
    return f"{user_id[:6]}...{user_id[-4:]}"


def _load_zenodo_track_ids() -> set[str]:
    """Load the set of track IDs that have Zenodo ground truth."""
    labeled_path = _CACHE_DIR / "labeled_tracks.parquet"
    if not labeled_path.exists():
        return set()
    df = pd.read_parquet(labeled_path, columns=["id"])
    return set(df["id"].dropna().tolist())


def _compute_dashboard_data() -> dict:
    """Compute all dashboard metrics in a single pass.

    Returns dict with 'users' (list of per-user dicts) and 'system' (aggregate).
    """
    user_ids = _discover_users()
    zenodo_ids = _load_zenodo_track_ids()

    # Load LLM cache once for data-quality checks
    llm_path = _CACHE_DIR / "llm_raw.parquet"
    llm_df = pd.read_parquet(llm_path) if llm_path.exists() else None

    users_data = []
    all_track_ids: set[str] = set()
    total_zenodo = 0
    total_tracks = 0
    total_playlists = 0
    total_missing_subfeatures = 0
    total_fallback_raw = 0

    for uid in user_ids:
        tracks = _load_user_tracks(uid)
        lib_size = len(tracks)
        track_ids = {tr.get("id", "") for tr in tracks if tr.get("id")}
        all_track_ids |= track_ids

        # Zenodo coverage: count how many of this user's tracks are in the labeled set
        zenodo_count = len(track_ids & zenodo_ids)

        # Data quality: check LLM cache for missing sub-features
        missing_subfeatures = 0
        fallback_raw = 0
        if llm_df is not None:
            user_llm = llm_df[llm_df["id"].isin(track_ids)]
            for col in _REQUIRED_SUB_FEATURES:
                if col in user_llm.columns:
                    missing_subfeatures += int(user_llm[col].isna().sum())
                else:
                    missing_subfeatures += len(user_llm)

        coverage_pct = (zenodo_count / lib_size * 100) if lib_size > 0 else 0.0
        playlists = _count_eval_logs(uid)

        total_zenodo += zenodo_count
        total_tracks += lib_size
        total_playlists += playlists
        total_missing_subfeatures += missing_subfeatures
        total_fallback_raw += fallback_raw

        users_data.append({
            "user_id": uid,
            "display_name": _safe_display_name(uid),
            "library_size": lib_size,
            "zenodo_covered": zenodo_count,
            "coverage_pct": coverage_pct,
            "playlists_generated": playlists,
            "low_coverage": coverage_pct < LOW_COVERAGE_THRESHOLD * 100,
            "missing_subfeatures": missing_subfeatures,
            "fallback_raw": fallback_raw,
        })

    overall_coverage = (total_zenodo / total_tracks * 100) if total_tracks > 0 else 0.0

    # Coverage distribution buckets
    buckets = {"0–25%": 0, "25–50%": 0, "50–75%": 0, "75–100%": 0}
    for u in users_data:
        pct = u["coverage_pct"]
        if pct < 25:
            buckets["0–25%"] += 1
        elif pct < 50:
            buckets["25–50%"] += 1
        elif pct < 75:
            buckets["50–75%"] += 1
        else:
            buckets["75–100%"] += 1

    return {
        "users": users_data,
        "system": {
            "total_users": len(user_ids),
            "total_unique_tracks": len(all_track_ids),
            "total_tracks": total_tracks,
            "overall_coverage_pct": overall_coverage,
            "total_playlists": total_playlists,
            "total_missing_subfeatures": total_missing_subfeatures,
            "total_fallback_raw": total_fallback_raw,
            "coverage_buckets": buckets,
        },
    }


def render_dashboard() -> None:
    """Render the full dashboard page."""
    L = st.session_state.get("lang", "en")

    st.title(t("dashboard_title", L))
    st.caption(t("dashboard_subtitle", L))

    data = _compute_dashboard_data()
    sys = data["system"]
    users = data["users"]

    # ── Summary cards ──
    st.header(t("dashboard_summary", L))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("dashboard_total_users", L), sys["total_users"])
    c2.metric(t("dashboard_total_tracks", L), sys["total_unique_tracks"])
    c3.metric(t("dashboard_overall_coverage", L), f"{sys['overall_coverage_pct']:.1f}%")
    c4.metric(t("dashboard_total_playlists", L), sys["total_playlists"])

    st.markdown("---")

    # ── Per-user table ──
    st.header(t("dashboard_per_user", L))

    if not users:
        st.info(t("dashboard_no_users", L))
    else:
        # Build DataFrame for sortable display
        table_data = []
        for u in users:
            row = {
                t("dashboard_col_user", L): u["display_name"],
                t("dashboard_col_library", L): u["library_size"],
                t("dashboard_col_zenodo", L): u["zenodo_covered"],
                t("dashboard_col_coverage", L): f"{u['coverage_pct']:.1f}%",
                t("dashboard_col_playlists", L): u["playlists_generated"],
                t("dashboard_col_status", L): (
                    t("dashboard_low_coverage_flag", L)
                    if u["low_coverage"]
                    else t("dashboard_ok", L)
                ),
            }
            table_data.append(row)

        df = pd.DataFrame(table_data)

        # Highlight low-coverage rows
        def _highlight_low_coverage(row: pd.Series) -> list[str]:
            status_col = t("dashboard_col_status", L)
            flag_text = t("dashboard_low_coverage_flag", L)
            if row.get(status_col) == flag_text:
                return ["background-color: #fff3cd"] * len(row)
            return [""] * len(row)

        styled = df.style.apply(_highlight_low_coverage, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Data quality panel ──
    st.header(t("dashboard_data_quality", L))

    dq1, dq2 = st.columns(2)
    with dq1:
        st.metric(
            t("dashboard_missing_subfeatures", L),
            sys["total_missing_subfeatures"],
            help=t("dashboard_missing_subfeatures_help", L),
        )
    with dq2:
        st.metric(
            t("dashboard_fallback_raw", L),
            sys["total_fallback_raw"],
            help=t("dashboard_fallback_raw_help", L),
        )

    # Coverage distribution
    st.subheader(t("dashboard_coverage_dist", L))
    buckets = sys["coverage_buckets"]
    if any(v > 0 for v in buckets.values()):
        import plotly.express as px
        fig = px.bar(
            x=list(buckets.keys()),
            y=list(buckets.values()),
            labels={"x": t("dashboard_coverage_range", L),
                    "y": t("dashboard_user_count", L)},
        )
        fig.update_layout(
            height=250,
            margin=dict(t=10, b=30, l=40, r=10),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(t("dashboard_no_data", L))

    # Back link
    if st.button(t("dashboard_back", L)):
        st.query_params.clear()
        st.rerun()
