"""Internationalisation — EN / JP UI strings for MindTune."""

from __future__ import annotations

TRANSLATIONS: dict[str, dict[str, str]] = {
    # ── Login page ──────────────────────────────────────────────
    "app_title": {
        "en": "\U0001f3b5 MindTune",
        "ja": "\U0001f3b5 MindTune",
    },
    "app_subtitle": {
        "en": (
            "Emotion-transition music recommendation using 3D emotion model "
            "(V/E/T) with dynamic path planning and Viterbi DP"
        ),
        "ja": (
            "3D感情モデル（V/E/T）とダイナミック経路計画・Viterbi DPによる"
            "感情遷移音楽レコメンデーション"
        ),
    },
    "welcome": {
        "en": "Welcome! Please log in to get started.",
        "ja": "ようこそ！ログインして始めましょう。",
    },
    "welcome_desc": {
        "en": (
            "MindTune generates personalised emotion-transition playlists from "
            "**your own** Spotify library. Each user's music is kept private."
        ),
        "ja": (
            "MindTuneはあなた自身のSpotifyライブラリから、パーソナライズされた"
            "感情遷移プレイリストを生成します。各ユーザーの音楽データは非公開です。"
        ),
    },
    "option_a_title": {
        "en": "#### Option A: Enter your Spotify User ID",
        "ja": "#### オプションA：Spotify ユーザーIDを入力",
    },
    "option_a_caption": {
        "en": "Use this if your library has already been imported.",
        "ja": "既にライブラリをインポート済みの場合はこちらを使用してください。",
    },
    "spotify_user_id": {
        "en": "Spotify User ID",
        "ja": "Spotify ユーザーID",
    },
    "log_in": {
        "en": "Log in",
        "ja": "ログイン",
    },
    "enter_user_id_error": {
        "en": "Please enter your Spotify User ID.",
        "ja": "Spotify ユーザーIDを入力してください。",
    },
    "option_b_title": {
        "en": "#### Option B: Connect with Spotify",
        "ja": "#### オプションB：Spotifyで接続",
    },
    "option_b_caption": {
        "en": (
            "Log in with your Spotify account to automatically identify "
            "yourself and import your library."
        ),
        "ja": (
            "Spotifyアカウントでログインし、自動的に本人確認と"
            "ライブラリのインポートを行います。"
        ),
    },
    "connect_spotify": {
        "en": "\U0001f3a7 Connect with Spotify",
        "ja": "\U0001f3a7 Spotifyで接続",
    },
    "set_spotify_creds": {
        "en": (
            "Set `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` in "
            "`.env` to enable Spotify login."
        ),
        "ja": (
            "`.env` に `SPOTIFY_CLIENT_ID` と `SPOTIFY_CLIENT_SECRET` を"
            "設定してSpotifyログインを有効にしてください。"
        ),
    },
    "login_failed": {
        "en": "Spotify login failed: {error}",
        "ja": "Spotifyログインに失敗しました: {error}",
    },
    "login_tip": {
        "en": "**Tip:** Make sure you open this app at the same URL as the redirect URI: `{uri}`",
        "ja": "**ヒント:** リダイレクトURIと同じURLでアプリを開いてください: `{uri}`",
    },

    # ── Library import ──────────────────────────────────────────
    "no_library": {
        "en": "No track library found for user **{user_id}**. Import your Spotify liked songs to get started.",
        "ja": "ユーザー **{user_id}** のトラックライブラリが見つかりません。Spotifyのお気に入り曲をインポートしてください。",
    },
    "authenticated_as": {
        "en": "Authenticated as **{name}**",
        "ja": "**{name}** として認証済み",
    },
    "import_liked_songs": {
        "en": "Import my Spotify liked songs",
        "ja": "Spotifyのお気に入り曲をインポート",
    },
    "fetching_liked_songs": {
        "en": "Fetching liked songs from Spotify...",
        "ja": "Spotifyからお気に入り曲を取得中...",
    },
    "fetched_songs": {
        "en": "Fetched {done}/{total} songs...",
        "ja": "{done}/{total} 曲を取得しました...",
    },
    "done": {
        "en": "Done!",
        "ja": "完了！",
    },
    "imported_tracks": {
        "en": "Imported {count} tracks into your library.",
        "ja": "{count} 曲をライブラリにインポートしました。",
    },
    "spotify_api_error": {
        "en": "Spotify API error: {error}",
        "ja": "Spotify APIエラー: {error}",
    },
    "import_failed": {
        "en": "Import failed: {error}",
        "ja": "インポートに失敗しました: {error}",
    },
    "connect_to_import": {
        "en": "\U0001f3a7 Connect to Spotify to import your library",
        "ja": "\U0001f3a7 Spotifyに接続してライブラリをインポート",
    },
    "set_creds_import": {
        "en": "Set Spotify credentials in `.env` to enable automatic import.",
        "ja": "`.env` にSpotify認証情報を設定してください。",
    },

    # ── 3D Circumplex ───────────────────────────────────────────
    "now_label": {
        "en": "NOW: {name}",
        "ja": "現在: {name}",
    },
    "goal_label": {
        "en": "GOAL: {name}",
        "ja": "目標: {name}",
    },
    "start": {
        "en": "Start",
        "ja": "開始",
    },
    "stage_label": {
        "en": "Stage {n} [{axis}]",
        "ja": "ステージ {n} [{axis}]",
    },
    "chart_title": {
        "en": "3D Emotion Space (V/E/T)",
        "ja": "3D感情空間 (V/E/T)",
    },
    "axis_valence": {
        "en": "Valence (V)",
        "ja": "感情価 (V)",
    },
    "axis_energy": {
        "en": "Energy Arousal (E)",
        "ja": "エネルギー覚醒度 (E)",
    },
    "axis_tension": {
        "en": "Tension Arousal (T)",
        "ja": "緊張覚醒度 (T)",
    },
    "legend_emotions": {
        "en": "Emotions",
        "ja": "感情",
    },
    "legend_current": {
        "en": "Current",
        "ja": "現在",
    },
    "legend_target": {
        "en": "Target",
        "ja": "目標",
    },
    "legend_path": {
        "en": "Transition Path",
        "ja": "遷移パス",
    },
    "legend_tracks": {
        "en": "Selected Tracks",
        "ja": "選択された曲",
    },

    # ── Feature store build ─────────────────────────────────────
    "fs_not_built": {
        "en": (
            "The v2 feature store has not been built for this user yet. "
            "Build it to enable v2 playlist generation."
        ),
        "ja": (
            "このユーザーのv2特徴量ストアはまだ構築されていません。"
            "v2プレイリスト生成を有効にするには構築してください。"
        ),
    },
    "fs_build_info": {
        "en": (
            "This will bootstrap v2 features (V/E/T + arc vectors) from the "
            "existing v1 track features and build the feature store. This takes "
            "about 30 seconds."
        ),
        "ja": (
            "既存のv1トラック特徴量からv2特徴量（V/E/T ＋ アークベクトル）を"
            "ブートストラップし、特徴量ストアを構築します。約30秒かかります。"
        ),
    },
    "build_feature_store": {
        "en": "Build Feature Store",
        "ja": "特徴量ストアを構築",
    },
    "building_fs": {
        "en": "Building feature store...",
        "ja": "特徴量ストアを構築中...",
    },
    "fs_built": {
        "en": "Feature store built with {count} tracks! Zenodo coverage: {matched}/{total}",
        "ja": "特徴量ストアを{count}曲で構築しました！ Zenodoカバレッジ: {matched}/{total}",
    },
    "build_failed": {
        "en": "Build failed: {error}",
        "ja": "構築に失敗しました: {error}",
    },

    # ── Feature estimation ──────────────────────────────────────
    "features_not_estimated": {
        "en": (
            "Your track library has {count} tracks but audio features "
            "have not been estimated yet. Run the feature estimation to enable "
            "playlist generation."
        ),
        "ja": (
            "トラックライブラリに{count}曲ありますが、音声特徴量がまだ推定されていません。"
            "プレイリスト生成を有効にするには特徴量推定を実行してください。"
        ),
    },
    "set_openai_key": {
        "en": "Set `OPENAI_API_KEY` in your `.env` file to enable LLM feature estimation.",
        "ja": "`.env` に `OPENAI_API_KEY` を設定してLLM特徴量推定を有効にしてください。",
    },
    "estimation_info": {
        "en": (
            "This will use OpenAI to estimate energy, happiness, BPM, and other "
            "features for each track based on its metadata."
        ),
        "ja": (
            "OpenAIを使用して各トラックのメタデータに基づき、エネルギー、"
            "幸福度、BPMなどの特徴量を推定します。"
        ),
    },
    "estimate_features": {
        "en": "Estimate Features",
        "ja": "特徴量を推定",
    },
    "estimating_features": {
        "en": "Estimating features...",
        "ja": "特徴量を推定中...",
    },
    "estimated_tracks": {
        "en": "Estimated {done}/{total} tracks...",
        "ja": "{done}/{total} 曲を推定しました...",
    },
    "processing_batch": {
        "en": "Processing batch... ({done}/{total})",
        "ja": "バッチ処理中... ({done}/{total})",
    },
    "estimation_done": {
        "en": "Estimated features for {done}/{total} tracks.",
        "ja": "{done}/{total} 曲の特徴量を推定しました。",
    },

    # ── Spotify Connect playback ────────────────────────────────
    "set_creds_playback": {
        "en": "Set Spotify credentials in `.env` to enable playback.",
        "ja": "`.env` にSpotify認証情報を設定して再生を有効にしてください。",
    },
    "connect_spotify_short": {
        "en": "\U0001f3a7 Connect to Spotify",
        "ja": "\U0001f3a7 Spotifyに接続",
    },
    "no_track_ids": {
        "en": "No tracks with Spotify IDs found.",
        "ja": "Spotify IDのあるトラックが見つかりません。",
    },
    "connected_as": {
        "en": "Connected to Spotify as **{name}**",
        "ja": "**{name}** としてSpotifyに接続中",
    },
    "fetch_devices_error": {
        "en": "Could not fetch devices: {error}",
        "ja": "デバイスの取得に失敗しました: {error}",
    },
    "no_devices": {
        "en": (
            "No active Spotify devices found. "
            "Open the Spotify app on your phone or computer first."
        ),
        "ja": (
            "アクティブなSpotifyデバイスが見つかりません。"
            "まずスマートフォンやパソコンでSpotifyアプリを開いてください。"
        ),
    },
    "refresh_devices": {
        "en": "\U0001f504 Refresh devices",
        "ja": "\U0001f504 デバイスを更新",
    },
    "select_device": {
        "en": "Select device",
        "ja": "デバイスを選択",
    },
    "play_on_spotify": {
        "en": "\u25b6 Play on Spotify",
        "ja": "\u25b6 Spotifyで再生",
    },
    "now_playing": {
        "en": "Now playing on **{device}** \u2014 {count} tracks queued!",
        "ja": "**{device}** で再生中 \u2014 {count} 曲をキューに追加！",
    },
    "playback_403": {
        "en": "Playback failed (403). Spotify Premium required.",
        "ja": "再生に失敗しました (403)。Spotify Premiumが必要です。",
    },
    "playback_404": {
        "en": "Device not found. Reopen the Spotify app and try again.",
        "ja": "デバイスが見つかりません。Spotifyアプリを再起動してください。",
    },
    "playback_failed": {
        "en": "Playback failed: {error}",
        "ja": "再生に失敗しました: {error}",
    },
    "session_expired": {
        "en": "Spotify session expired. Please reconnect.",
        "ja": "Spotifyセッションが期限切れです。再接続してください。",
    },

    # ── v2 Playlist display ─────────────────────────────────────
    "stage_heading": {
        "en": "Stage {stage} \u2014 Lead axis: {lead}",
        "ja": "ステージ {stage} \u2014 主軸: {lead}",
    },
    "stage_target_caption": {
        "en": "Target \u2014 V: {v:.2f} | E: {e:.2f} | T: {tval:.2f}",
        "ja": "目標 \u2014 V: {v:.2f} | E: {e:.2f} | T: {tval:.2f}",
    },

    # ── Rating widget ───────────────────────────────────────────
    "rate_playlist": {
        "en": "Rate this playlist",
        "ja": "このプレイリストを評価",
    },
    "rate_caption": {
        "en": "Your feedback will help improve future recommendations (Phase 2).",
        "ja": "あなたのフィードバックは今後のレコメンデーション改善に役立ちます（Phase 2）。",
    },
    "rate_slider": {
        "en": "How well does this playlist match your emotional transition?",
        "ja": "このプレイリストは感情遷移にどの程度マッチしていますか？",
    },
    "submit_rating": {
        "en": "Submit Rating",
        "ja": "評価を送信",
    },
    "rating_saved": {
        "en": "Rating saved ({rating}/5). Thank you!",
        "ja": "評価を保存しました（{rating}/5）。ありがとうございます！",
    },
    "phase2_note": {
        "en": "Note: Phase 2 bandit learning is not yet active.",
        "ja": "注: Phase 2のバンディット学習はまだ有効ではありません。",
    },

    # ── Main app / sidebar ──────────────────────────────────────
    "user_label": {
        "en": "**User:** `{user_id}`",
        "ja": "**ユーザー:** `{user_id}`",
    },
    "log_out": {
        "en": "Log out",
        "ja": "ログアウト",
    },
    "main_title": {
        "en": "\U0001f3b5 MindTune v2",
        "ja": "\U0001f3b5 MindTune v2",
    },
    "main_subtitle": {
        "en": (
            "3D emotion model (V/E/T) with dynamic path planning, "
            "Viterbi DP track selection, and Spotify Connect playback"
        ),
        "ja": (
            "3D感情モデル（V/E/T）＋ダイナミック経路計画、"
            "Viterbi DPトラック選択、Spotify Connect再生"
        ),
    },
    "emotion_settings": {
        "en": "Emotion Settings",
        "ja": "感情設定",
    },
    "how_feeling": {
        "en": "How are you feeling now?",
        "ja": "今の気分は？",
    },
    "how_want_feel": {
        "en": "How do you want to feel?",
        "ja": "どんな気分になりたい？",
    },
    "algo_settings": {
        "en": "Algorithm Settings",
        "ja": "アルゴリズム設定",
    },
    "algo_caption": {
        "en": "Dynamic path planning (v2)",
        "ja": "ダイナミック経路計画 (v2)",
    },
    "n_stages": {
        "en": "Number of stages (N)",
        "ja": "ステージ数 (N)",
    },
    "n_stages_help": {
        "en": "How many stages in the emotion transition path",
        "ja": "感情遷移パスのステージ数",
    },
    "k_candidates": {
        "en": "Candidates per stage (K)",
        "ja": "ステージあたりの候補数 (K)",
    },
    "k_candidates_help": {
        "en": "Higher K = more candidate tracks considered per stage",
        "ja": "Kが大きいほど各ステージで考慮される候補曲が増えます",
    },
    "alpha_label": {
        "en": "α (lead-axis concentration)",
        "ja": "α（主軸集中度）",
    },
    "alpha_help": {
        "en": "0.33 = linear (equal pace), 0.6 = dynamic (default), 1.0 = fully sequential",
        "ja": "0.33 = 線形（均等）、0.6 = 動的（デフォルト）、1.0 = 完全逐次",
    },
    "v1_debug_title": {
        "en": "v1 Linear strategy (debug)",
        "ja": "v1 線形戦略（デバッグ）",
    },
    "v1_debug_caption": {
        "en": "Legacy 2D strategy from v1. Use for comparison only.",
        "ja": "v1のレガシー2D戦略。比較用のみ。",
    },
    "enable_v1": {
        "en": "Enable v1 mode",
        "ja": "v1モードを有効化",
    },
    "library_header": {
        "en": "Library",
        "ja": "ライブラリ",
    },
    "track_library_count": {
        "en": "Track library: {count} tracks",
        "ja": "トラックライブラリ: {count} 曲",
    },
    "feature_store_count": {
        "en": "Feature store: {count} tracks",
        "ja": "特徴量ストア: {count} 曲",
    },
    "feature_store_not_built": {
        "en": "Feature store: not built",
        "ja": "特徴量ストア: 未構築",
    },
    "refresh_library": {
        "en": "\U0001f504 Refresh Library from Spotify",
        "ja": "\U0001f504 Spotifyからライブラリを更新",
    },
    "fetching_liked": {
        "en": "Fetching liked songs...",
        "ja": "お気に入り曲を取得中...",
    },
    "updated_tracks": {
        "en": "Updated! {count} tracks imported.",
        "ja": "更新しました！ {count} 曲をインポートしました。",
    },
    "refresh_failed": {
        "en": "Refresh failed: {error}",
        "ja": "更新に失敗しました: {error}",
    },
    "connect_spotify_refresh": {
        "en": "\U0001f3a7 Connect Spotify to refresh",
        "ja": "\U0001f3a7 Spotifyに接続して更新",
    },
    "refresh_and_rebuild": {
        "en": "\U0001f504 Refresh & Rebuild",
        "ja": "\U0001f504 更新＆再構築",
    },
    "rebuild_feature_store": {
        "en": "\U0001f504 Rebuild Feature Store",
        "ja": "\U0001f504 特徴量ストアを再構築",
    },
    "clean_rebuild": {
        "en": "\U0001f5d1 Clean Rebuild (delete all & re-import)",
        "ja": "\U0001f5d1 クリーン再構築（全削除＆再インポート）",
    },
    "cache_cleared": {
        "en": "All cached data cleared.",
        "ja": "キャッシュデータをすべて削除しました。",
    },
    "clean_rebuild_no_spotify": {
        "en": "Cache cleared but Spotify not connected — connect and click again to re-import.",
        "ja": "キャッシュを削除しましたがSpotifyが未接続です。接続後に再度クリックしてください。",
    },
    "clean_rebuild_failed": {
        "en": "Clean rebuild failed: {error}",
        "ja": "クリーン再構築に失敗しました: {error}",
    },
    "reset_ratings": {
        "en": "\U0001f9f9 Reset Ratings",
        "ja": "\U0001f9f9 評価をリセット",
    },
    "ratings_reset": {
        "en": "Ratings and playlist results cleared.",
        "ja": "評価とプレイリスト結果をクリアしました。",
    },

    # ── Evaluation section ────────────────────────────────────────
    "eval_header": {
        "en": "Evaluation",
        "ja": "評価実験",
    },
    "eval_caption": {
        "en": "Parameters and modes for evaluation only.",
        "ja": "評価実験用のパラメータとモード。",
    },
    "eval_mode_toggle": {
        "en": "Blind A/B/C comparison",
        "ja": "ブラインドA/B/C比較",
    },
    "eval_mode_help": {
        "en": "Generates playlists from Dynamic, Linear, and Spotify Autoplay — presented as unlabeled Method A/B/C.",
        "ja": "Dynamic、Linear、Spotify Autoplayの3手法でプレイリストを生成し、ラベルなしのMethod A/B/Cとして表示します。",
    },
    "generate_eval": {
        "en": "Generate A/B/C Comparison",
        "ja": "A/B/C比較を生成",
    },
    "running_eval": {
        "en": "Generating playlists for all three methods...",
        "ja": "3手法のプレイリストを生成中...",
    },
    "eval_blind_info": {
        "en": "Three methods are shown below as Method A, B, and C (randomly assigned). Listen to each and rate them.",
        "ja": "以下に3つの手法をMethod A、B、Cとして表示しています（ランダムに割り当て）。それぞれを聴いて評価してください。",
    },
    "eval_need_spotify": {
        "en": "Spotify Autoplay baseline requires Spotify connection. Connect via sidebar.",
        "ja": "Spotify Autoplayベースラインにはスポティファイ接続が必要です。サイドバーから接続してください。",
    },
    "eval_no_tracks": {
        "en": "No tracks generated for this method.",
        "ja": "この手法ではトラックが生成されませんでした。",
    },
    "play_method": {
        "en": "▶ Play Method {label}",
        "ja": "▶ Method {label} を再生",
    },
    "playing_method": {
        "en": "Playing Method {label}...",
        "ja": "Method {label} を再生中...",
    },
    "show_mapping_toggle": {
        "en": "Show method mapping (de-blind)",
        "ja": "手法の割り当てを表示（ブラインド解除）",
    },
    "mapping_header": {
        "en": "Method Mapping",
        "ja": "手法の割り当て",
    },

    # ── Tabs ────────────────────────────────────────────────────
    "tab_generate": {
        "en": "\U0001f3b6 Generate Playlist",
        "ja": "\U0001f3b6 プレイリスト生成",
    },
    "tab_circumplex": {
        "en": "\U0001f4ca 3D Circumplex",
        "ja": "\U0001f4ca 3D サーカンプレックス",
    },
    "tab_library": {
        "en": "\U0001f4c0 Track Library",
        "ja": "\U0001f4c0 トラックライブラリ",
    },

    # ── Generate tab ────────────────────────────────────────────
    "same_emotion": {
        "en": "Select different emotions for current and target.",
        "ja": "現在と目標に異なる感情を選択してください。",
    },
    "need_features_or_store": {
        "en": "Run feature estimation or build the feature store first.",
        "ja": "まず特徴量推定の実行または特徴量ストアの構築を行ってください。",
    },
    "features_setup_note": {
        "en": "You can still explore the emotion model below while features are being set up.",
        "ja": "特徴量のセットアップ中でも、下の感情モデルを探索できます。",
    },
    "generate_v1": {
        "en": "Generate Playlist (v1 Linear)",
        "ja": "プレイリスト生成（v1 線形）",
    },
    "v1_mode_warning": {
        "en": "v1 Linear mode (for comparison only)",
        "ja": "v1 線形モード（比較用のみ）",
    },
    "phase_heading": {
        "en": "Phase {n}: {label}",
        "ja": "フェーズ {n}: {label}",
    },
    "generate_v2": {
        "en": "Generate Playlist",
        "ja": "プレイリスト生成",
    },
    "running_v2": {
        "en": "Running dynamic path planning + Viterbi DP...",
        "ja": "ダイナミック経路計画 + Viterbi DPを実行中...",
    },
    "result_summary": {
        "en": "{source} \u2192 {target} | Axis order: {order} | Stages: {alloc}",
        "ja": "{source} \u2192 {target} | 軸順序: {order} | ステージ: {alloc}",
    },
    "progress_matrix": {
        "en": "Progress Matrix",
        "ja": "進行行列",
    },
    "emotion_details": {
        "en": "Emotion Details (V/E/T)",
        "ja": "感情の詳細 (V/E/T)",
    },
    "track_library_title": {
        "en": "Track Library ({count} tracks)",
        "ja": "トラックライブラリ ({count} 曲)",
    },
    "search_tracks": {
        "en": "Search tracks",
        "ja": "トラックを検索",
    },
    "showing_matches": {
        "en": "Showing {count} matching tracks",
        "ja": "{count} 件のトラックが見つかりました",
    },
    "page": {
        "en": "Page",
        "ja": "ページ",
    },

    # ── Language toggle ─────────────────────────────────────────
    "language": {
        "en": "Language / 言語",
        "ja": "Language / 言語",
    },
}

# ── Emotion name translations ───────────────────────────────────
EMOTION_NAMES: dict[str, dict[str, str]] = {
    "Angry":      {"en": "Angry",      "ja": "怒り"},
    "Anxious":    {"en": "Anxious",    "ja": "不安"},
    "Fear":       {"en": "Fear",       "ja": "恐怖"},
    "Restless":   {"en": "Restless",   "ja": "落ち着かない"},
    "Sad":        {"en": "Sad",        "ja": "悲しみ"},
    "Melancholy": {"en": "Melancholy", "ja": "憂鬱"},
    "Tired":      {"en": "Tired",      "ja": "疲労"},
    "Calm":       {"en": "Calm",       "ja": "穏やか"},
    "Peaceful":   {"en": "Peaceful",   "ja": "平和"},
    "Focused":    {"en": "Focused",    "ja": "集中"},
    "Confident":  {"en": "Confident",  "ja": "自信"},
    "Excited":    {"en": "Excited",    "ja": "興奮"},
}


def t(key: str, lang: str = "en", **kwargs: object) -> str:
    """Look up a translated string, with optional format kwargs."""
    entry = TRANSLATIONS.get(key, {})
    text = entry.get(lang, entry.get("en", key))
    if kwargs:
        text = text.format(**kwargs)
    return text


def emotion_name(name: str, lang: str = "en") -> str:
    """Translate an emotion label.  Falls back to the English name."""
    entry = EMOTION_NAMES.get(name, {})
    return entry.get(lang, name)
