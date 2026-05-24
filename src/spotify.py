"""Spotify API integration for playlist creation."""

import urllib.parse

import requests

from src.config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET

SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE = "https://api.spotify.com/v1"
SCOPES = "playlist-modify-public playlist-modify-private"


def get_auth_url(redirect_uri: str) -> str:
    """Generate the Spotify OAuth authorization URL."""
    params = {
        "client_id": SPOTIFY_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": SCOPES,
        "show_dialog": "true",
    }
    return SPOTIFY_AUTH_URL + "?" + urllib.parse.urlencode(params)


def exchange_code_for_token(code: str, redirect_uri: str) -> dict:
    """Exchange an authorization code for access and refresh tokens."""
    resp = requests.post(
        SPOTIFY_TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": SPOTIFY_CLIENT_ID,
            "client_secret": SPOTIFY_CLIENT_SECRET,
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def refresh_access_token(refresh_token: str) -> dict:
    """Use a refresh token to obtain a new access token."""
    resp = requests.post(
        SPOTIFY_TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": SPOTIFY_CLIENT_ID,
            "client_secret": SPOTIFY_CLIENT_SECRET,
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def get_current_user(access_token: str) -> dict:
    """Fetch the current user's Spotify profile."""
    resp = requests.get(
        f"{SPOTIFY_API_BASE}/me",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def create_playlist(
    access_token: str,
    name: str,
    description: str = "",
) -> dict:
    """Create a new playlist in the current user's Spotify account."""
    resp = requests.post(
        f"{SPOTIFY_API_BASE}/me/playlists",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json={
            "name": name,
            "description": description,
            "public": False,
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def add_tracks(access_token: str, playlist_id: str, track_ids: list[str]) -> dict:
    """Add tracks to a Spotify playlist. Accepts Spotify track IDs."""
    uris = [f"spotify:track:{tid}" for tid in track_ids]
    resp = requests.post(
        f"{SPOTIFY_API_BASE}/playlists/{playlist_id}/tracks",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json={"uris": uris},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def save_playlist(
    access_token: str,
    name: str,
    description: str,
    track_ids: list[str],
) -> dict:
    """Create a playlist and add tracks in one call. Returns the playlist object."""
    playlist = create_playlist(access_token, name, description)
    if track_ids:
        add_tracks(access_token, playlist["id"], track_ids)
    return playlist
