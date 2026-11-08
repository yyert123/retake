"""Minimal Miragea API client (https://miragea.ai/v1), standard library only.

The API key is read from the MIRAGEA_API_KEY environment variable and is never
logged, cached or written to disk.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from . import __version__

DEFAULT_BASE = "https://miragea.ai/v1"
# Miragea sits behind Cloudflare, which rejects default library user agents.
USER_AGENT = f"retake-vo/{__version__} (+https://github.com/yyert123/retake)"


class MirageaError(RuntimeError):
    """An API or network error with a human-readable fix hint."""

    def __init__(self, message: str, hint: str = "", status: int | None = None, code: str = ""):
        self.hint = hint
        self.status = status
        self.code = code
        super().__init__(message + (f"\n  How to fix: {hint}" if hint else ""))


_HINTS = {
    401: "Your MIRAGEA_API_KEY was rejected. Create a key at https://miragea.ai and export MIRAGEA_API_KEY=...",
    402: "Your Miragea balance is too low for this job. Top up at https://miragea.ai.",
    403: "This key has no API access. API access is enabled per account at https://miragea.ai.",
    413: "The file is too large for one request (limit: 1024 MB / 90 min). Trim it first.",
    429: "Rate limited or out of quota. Wait a minute and run the same command again (cached steps are not re-billed).",
}


class Client:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, timeout: float = 60):
        self.key = api_key or os.environ.get("MIRAGEA_API_KEY", "")
        if not self.key:
            raise MirageaError(
                "MIRAGEA_API_KEY is not set.",
                "Get a key at https://miragea.ai (API section), then run: export MIRAGEA_API_KEY=your_key",
                code="missing_api_key",
            )
        self.base = (base_url or os.environ.get("MIRAGEA_BASE_URL") or DEFAULT_BASE).rstrip("/")
        self.timeout = timeout

    # -- low level -------------------------------------------------------
    def _request(self, method: str, path: str, body: bytes | None = None, content_type: str | None = None,
                 idem_key: str | None = None) -> dict:
        headers = {"Authorization": "Bearer " + self.key, "Accept": "application/json", "User-Agent": USER_AGENT}
        if content_type:
            headers["Content-Type"] = content_type
        if idem_key:
            headers["Idempotency-Key"] = idem_key
        req = urllib.request.Request(self.base + path, data=body, headers=headers, method=method)
        last: Exception | None = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.load(resp)
            except urllib.error.HTTPError as exc:
                try:
                    err = json.load(exc).get("error", {})
                except Exception:  # noqa: BLE001
                    err = {}
                retryable = err.get("retryable") or exc.code in (502, 503, 504)
                if retryable and attempt < 3:
                    time.sleep(2 * (attempt + 1))
                    continue
                msg = err.get("message") or f"Miragea API returned HTTP {exc.code}"
                raise MirageaError(msg, _HINTS.get(exc.code, ""), exc.code, err.get("code", "")) from None
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
                if attempt < 3:
                    time.sleep(2 * (attempt + 1))
                    continue
        raise MirageaError(f"Could not reach {self.base}: {last}",
                           "Check your internet connection or MIRAGEA_BASE_URL, then re-run; the same request is retried safely.")

    def wait(self, task: dict, timeout: float = 900) -> dict:
        deadline = time.time() + timeout
        while task.get("status") not in ("succeeded", "failed"):
            if time.time() > deadline:
                raise MirageaError(f"Task {task.get('task_id')} still {task.get('status')} after {timeout:.0f}s.",
                                   "Re-run the same command later; finished results are cached.")
            time.sleep(max(1.0, float(task.get("poll_after_seconds") or 3)))
            task = self._request("GET", "/tasks/" + urllib.parse.quote(task["task_id"], safe=""))
        if task["status"] == "failed":
            err = task.get("error") or {}
            raise MirageaError(f"Miragea task {task.get('task_id')} failed: {err.get('message', 'unknown error')}",
                               "Points are refunded automatically for failed audio tasks. Check the input and retry.")
        return task

    # -- endpoints -------------------------------------------------------
    def capabilities(self) -> dict:
        return self._request("GET", "/capabilities")

    def transcribe(self, path: str | Path) -> dict:
        """Upload a local audio/video file; returns {text, language, segments[{start,end,text}], srt}."""
        path = Path(path)
        rate = self.capabilities().get("subtitles", {}).get("points_per_second", 3)
        boundary = "retake-" + uuid.uuid4().hex
        head = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"expected_rate\"\r\n\r\n{rate}\r\n"
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
                f"Content-Type: application/octet-stream\r\n\r\n").encode()
        body = head + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        task = self._request("POST", "/subtitles/recognitions", body, "multipart/form-data; boundary=" + boundary,
                             str(uuid.uuid4()))
        return self.wait(task)["result"]

    def speak(self, text: str, out_path: str | Path, *, actor_id: str | None = None,
              reference_audio_url: str | None = None, reference_text: str | None = None,
              model: str = "expressive", fmt: str = "wav", speed: float | None = None) -> Path:
        """Generate speech in a saved voice (actor_id) or a cloned reference voice; saves to out_path."""
        if not (actor_id or reference_audio_url):
            raise MirageaError("No voice given.", "Pass a Miragea voice id (--voice) or a public reference URL (--ref-url).")
        payload: dict = {"text": text, "model": model, "format": fmt}
        if actor_id:
            payload["actor_id"] = actor_id
        else:
            payload["reference_audio_url"] = reference_audio_url
            if reference_text:
                payload["reference_text"] = reference_text
        if speed:
            payload["speed"] = speed
        task = self._request("POST", "/audio/generations", json.dumps(payload).encode(), "application/json",
                             str(uuid.uuid4()))
        url = self.wait(task)["result"]["audio_url"]
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        out_path = Path(out_path)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            out_path.write_bytes(resp.read())
        return out_path
