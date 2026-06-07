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

---

# MindTune（日本語）

**ISO原理**と**Russellの円環モデル**に基づく感情遷移型音楽推薦システム。MindTuneは、ユーザーの現在の感情状態から目標の感情状態へと導くプレイリストを、音楽の順序を慎重に選択して生成します。

## 概要

従来の推薦システムが嗜好の最適化を目指すのに対し、MindTuneは**感情の軌道**を最適化します。音楽療法のISO原理を用いて、リスナーを段階的に遷移させるプレイリストを生成し、多次元スコアリングアルゴリズムによってトラックを選択します。

主な特徴：
- Russellの覚醒度-感情価2D空間に**11の感情状態**をマッピング
- **4つの遷移戦略**: 覚醒度優先、感情価優先、線形、動的
- トラックメタデータからの**LLMベースのオーディオ特徴量推定**（エネルギー、幸福度、BPMなど）
- **重み付きスコアリングアルゴリズム**によるトラック選択（energy:3, happiness:3, BPM:2, vibe:1, instrumentalness:2）
- 円環モデルと遷移パスの**インタラクティブな可視化**
- **Spotify連携** — ユーザーのSpotifyお気に入り曲からトラックライブラリを取得
- **マルチユーザー対応** — 各ユーザーのライブラリを独立して保存・検索
- **Spotify Connectで再生** — プレイリストをスマートフォンやデスクトップに直接送信

## アーキテクチャ

```
感情入力（現在 → 目標）
         │
         ▼
┌──────────────────┐
│ 戦略エンジン       │  ← 覚醒度優先 / 感情価優先 / 線形 / 動的
│ （ISO原理）        │
└────────┬─────────┘
         │  3フェーズ
         ▼
┌──────────────────┐     ┌──────────────────┐
│ スコアリング       │────▶│  トラックライブラリ │
│ アルゴリズム       │     │ （ユーザーごとに保存│
│ （重み付きマッチ）  │     │  + LLM特徴量）    │
└────────┬─────────┘     └──────────────────┘
         │
         ▼
┌──────────────────┐
│ Streamlit UI      │  ← 円環モデル可視化 + プレイリスト
└──────────────────┘
```

## セットアップ

### 1. 依存関係のインストール

```bash
pip install -e .
```

### 2. 環境設定

```bash
cp .env.example .env
# .envにOpenAI APIキーを設定してください
```

### 3. ログインとライブラリのインポート

アプリを開いて、以下のいずれかを選択してください：
- **Spotifyと連携**（推奨） — 自動的にユーザーを識別し、お気に入り曲をインポート
- **Spotify User IDを入力** — 以前にライブラリをインポート済みの場合

### 4. オーディオ特徴量の推定

ライブラリをインポートした後、アプリ内のボタンでオーディオ特徴量を推定するか、コマンドラインから実行してください：

```bash
python -m src.ingest <your_spotify_user_id>
```

### 5. アプリの起動

```bash
streamlit run src/app.py
```

## 遷移戦略

| 戦略 | 説明 | 適した場面 |
|------|------|-----------|
| **覚醒度優先** | まずエネルギーを調整し、次に気分を変化 | 不安、怒り（高覚醒・ネガティブ） |
| **感情価優先** | まず気分を変化させ、次にエネルギーを調整 | 悲しみ、疲労（低覚醒・ネガティブ） |
| **線形** | 両次元を均等に変化 | 穏やかな遷移 |
| **動的** | 差が大きい次元を自動選択 | 一般的な使用（推奨） |

## プロジェクト構成

```
src/
├── app.py               # Streamlit UI
├── config.py            # 環境設定
├── emotions.py          # Russellの円環モデル（11の感情）
├── strategies.py        # 4つの遷移戦略
├── scoring.py           # 重み付きトラックスコアリングアルゴリズム
├── playlist.py          # プレイリスト生成オーケストレーター
├── feature_estimator.py # LLMベースのオーディオ特徴量推定
├── tracks.py            # ユーザーごとのトラックデータベース管理
├── ingest.py            # バッチ特徴量推定スクリプト
├── spotify.py           # Spotify API（認証、ライブラリインポート、再生）
└── data/
    └── users/           # ユーザーごとのトラックライブラリ（gitignore対象）
        └── <user_id>/
            └── tracks.json
```

## 参考文献

- Altshuler, I. M. (1948). *A psychiatrist's experience with music as a therapeutic agent*
- Russell, J. A. (1980). *A circumplex model of affect*. Journal of Personality and Social Psychology, 39(6), 1161-1178
- Posner, J., Russell, J. A., & Peterson, B. S. (2005). *The circumplex model of affect: An integrative approach to affective neuroscience*
