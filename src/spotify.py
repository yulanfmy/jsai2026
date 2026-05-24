"""Spotify API integration for playback via Spotify Connect."""

import json as _json
import subprocess
import urllib.parse

import requests

from src.config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET

SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE = "https://api.spotify.com/v1"
SCOPES = "user-read-playback-state user-modify-playback-state"


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


class SpotifyAPIError(Exception):
    """Raised when a Spotify API call returns a non-success status."""

    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"Spotify API error {status_code}: {body}")


def _curl(
    method: str, url: str, access_token: str, payload: dict | None = None,
) -> tuple[int, str]:
    """Execute an HTTP request via subprocess curl.

    Both the ``requests`` library and ``urllib.request`` return 403 when
    called from inside a Streamlit server process, while the same token
    and URL succeed with curl.  Using subprocess curl bypasses whatever
    network interception causes the issue.
    """
    cmd = [
        "curl", "-s", "-w", "\n%{http_code}",
        "-X", method,
        "-H", f"Authorization: Bearer {access_token}",
    ]
    if payload is not None:
        cmd += ["-H", "Content-Type: application/json", "-d", _json.dumps(payload)]
    cmd.append(url)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    lines = result.stdout.strip().rsplit("\n", 1)
    body = lines[0] if len(lines) > 1 else ""
    status = int(lines[-1]) if lines and lines[-1].isdigit() else 0
    return status, body


def get_devices(access_token: str) -> list[dict]:
    """Return the list of the user's available Spotify Connect devices."""
    status, body = _curl("GET", f"{SPOTIFY_API_BASE}/me/player/devices", access_token)
    if status < 200 or status >= 300:
        raise SpotifyAPIError(status, body)
    data = _json.loads(body) if body else {}
    return data.get("devices", [])


def start_playback(
    access_token: str,
    track_ids: list[str],
    device_id: str | None = None,
) -> None:
    """Start playback of the given tracks on a Spotify Connect device.

    If *device_id* is ``None`` the user's currently active device is used.
    """
    uris = [f"spotify:track:{tid}" for tid in track_ids]
    url = f"{SPOTIFY_API_BASE}/me/player/play"
    if device_id:
        url += f"?device_id={device_id}"
    status, body = _curl("PUT", url, access_token, {"uris": uris})
    if status not in (200, 202, 204):
        raise SpotifyAPIError(status, body)
