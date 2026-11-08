"""Audio helpers: decode/encode via ffmpeg, pause snapping, loudness + EQ match,
room tone and equal-power crossfades. Mono float32 internally; the output keeps
the source sample rate and channel count."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np


class AudioError(RuntimeError):
    pass


def require_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise AudioError("ffmpeg was not found on PATH.\n  How to fix: install it (macOS: brew install ffmpeg; "
                         "Ubuntu: sudo apt install ffmpeg; Windows: winget install ffmpeg) and re-run.")
    return exe


def probe(path: str | Path) -> tuple[int, int]:
    """Return (sample_rate, channels) of the first audio stream."""
    out = subprocess.run([shutil.which("ffprobe") or "ffprobe", "-v", "error", "-select_streams", "a:0",
                          "-show_entries", "stream=sample_rate,channels", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True)
    try:
        sr, ch = out.stdout.strip().splitlines()[0].split(",")[:2]
        return int(sr), int(ch)
    except Exception:  # noqa: BLE001
        raise AudioError(f"Could not read audio from {path}. Is it an audio or video file?") from None


def load(path: str | Path, sr: int | None = None, channels: int | None = None) -> tuple[np.ndarray, int]:
    """Decode any ffmpeg-readable file to float32 array shaped (n, channels)."""
    require_ffmpeg()
    src_sr, src_ch = probe(path)
    sr = sr or src_sr
    channels = channels or src_ch
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-f", "f32le", "-ac", str(channels), "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, channels).copy(), sr


def save(path: str | Path, audio: np.ndarray, sr: int) -> Path:
    path = Path(path)
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    ch = 1 if audio.ndim == 1 else audio.shape[1]
    codec = ["-c:a", "pcm_s24le"] if path.suffix.lower() == ".wav" else []
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ac", str(ch), "-ar", str(sr), "-i", "-",
                    *codec, str(path)], input=audio.tobytes(), check=True)
    return path


def mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=1) if audio.ndim == 2 else audio


def frame_rms(x: np.ndarray, sr: int, hop_s: float = 0.01, win_s: float = 0.03) -> np.ndarray:
    hop, win = max(1, int(sr * hop_s)), max(1, int(sr * win_s))
    if len(x) < win:
        return np.array([float(np.sqrt(np.mean(x ** 2) + 1e-12))])
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return np.sqrt(np.mean(x[idx] ** 2, axis=1) + 1e-12)


def snap_to_pause(x: np.ndarray, sr: int, t: float, lo: float, hi: float) -> float:
    """Move time ``t`` to the quietest 30 ms point inside [lo, hi] (seconds)."""
    lo, hi = max(0.0, lo), min(len(x) / sr, hi)
    if hi - lo < 0.04:
        return t
    a, b = int(lo * sr), int(hi * sr)
    rms = frame_rms(x[a:b], sr)
    # Prefer the centre of the longest quiet run, not just the single minimum.
    thresh = np.min(rms) * 1.6 + 1e-5
    quiet = rms <= thresh
    best, best_len, run_start = None, 0, None
    for i, q in enumerate(list(quiet) + [False]):
        if q and run_start is None:
            run_start = i
        elif not q and run_start is not None:
            if i - run_start > best_len:
                best, best_len = (run_start + i) // 2, i - run_start
            run_start = None
    i = best if best is not None else int(np.argmin(rms))
    return lo + i * 0.01 + 0.015


def trim_silence(x: np.ndarray, sr: int, pad_s: float = 0.03, rel_db: float = -40.0) -> np.ndarray:
    m = mono(x)
    rms = frame_rms(m, sr)
    level = 20 * np.log10(rms + 1e-12)
    voiced = np.where(level > level.max() + rel_db)[0]
    if len(voiced) == 0:
        return x
    a = max(0, int((voiced[0] * 0.01 - pad_s) * sr))
    b = min(len(x), int((voiced[-1] * 0.01 + 0.03 + pad_s) * sr))
    return x[a:b]


def speech_rms(x: np.ndarray, sr: int) -> float:
    """RMS over the louder (speech) frames only, so pauses do not skew the match."""
    rms = frame_rms(mono(x), sr)
    if len(rms) < 4:
        return float(np.mean(rms))
    top = rms[rms >= np.percentile(rms, 40)]
    return float(np.sqrt(np.mean(top ** 2)))


def match_loudness(patch: np.ndarray, ref: np.ndarray, sr: int, max_gain_db: float = 18.0) -> tuple[np.ndarray, float]:
    g = speech_rms(ref, sr) / max(speech_rms(patch, sr), 1e-9)
    g_db = float(np.clip(20 * np.log10(g), -max_gain_db, max_gain_db))
    return patch * (10 ** (g_db / 20)), g_db


def _avg_spectrum(x: np.ndarray, n_fft: int) -> np.ndarray:
    m = mono(x)
    if len(m) < n_fft:
        m = np.pad(m, (0, n_fft - len(m)))
    hop = n_fft // 2
    frames = [m[i:i + n_fft] * np.hanning(n_fft) for i in range(0, len(m) - n_fft + 1, hop)]
    return np.sqrt(np.mean(np.abs(np.fft.rfft(np.array(frames), axis=1)) ** 2, axis=0) + 1e-12)


def match_eq(patch: np.ndarray, ref: np.ndarray, sr: int, max_db: float = 9.0, n_fft: int = 2048) -> np.ndarray:
    """Shape the patch's long-term spectrum toward the reference (smoothed, clamped)."""
    curve = _avg_spectrum(ref, n_fft) / _avg_spectrum(patch, n_fft)
    curve_db = 20 * np.log10(curve)
    # Smooth over ~1/3 octave so we match tone colour, not individual harmonics.
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    smooth = np.empty_like(curve_db)
    for i, f in enumerate(freqs):
        lo, hi = f / 1.12, f * 1.12 + sr / n_fft
        sel = (freqs >= lo) & (freqs <= hi)
        smooth[i] = np.mean(curve_db[sel])
    smooth = np.clip(smooth - np.median(smooth[(freqs > 200) & (freqs < 4000)]), -max_db, max_db)
    gains = 10 ** (smooth / 20)
    p = patch if patch.ndim == 2 else patch[:, None]
    n = len(p)
    size = 1 << int(np.ceil(np.log2(n + n_fft)))
    g_full = np.interp(np.fft.rfftfreq(size, 1 / sr), freqs, gains)
    out = np.fft.irfft(np.fft.rfft(p, size, axis=0) * g_full[:, None], size, axis=0)[:n]
    return (out if patch.ndim == 2 else out[:, 0]).astype(np.float32)


def room_tone(src: np.ndarray, sr: int, length: int) -> np.ndarray:
    """Loop the quietest 400 ms of the take to make background noise of ``length`` samples."""
    m = mono(src)
    win = int(0.4 * sr)
    if len(m) <= win or length <= 0:
        return np.zeros((length, src.shape[1]), dtype=np.float32)
    rms = frame_rms(m, sr, hop_s=0.05, win_s=0.4)
    i = int(np.argmin(rms)) * int(0.05 * sr)
    piece = src[i:i + win]
    fade = int(0.05 * sr)
    reps = int(np.ceil(length / (win - fade))) + 1
    out = np.zeros((reps * (win - fade) + fade, src.shape[1]), dtype=np.float32)
    ramp = np.linspace(0, 1, fade)[:, None]
    for r in range(reps):
        seg = piece.copy()
        seg[:fade] *= np.sqrt(ramp)
        seg[-fade:] *= np.sqrt(1 - ramp)
        s = r * (win - fade)
        out[s:s + win] += seg
    return out[:length]


def crossfade_join(a: np.ndarray, b: np.ndarray, sr: int, ms: float = 20.0) -> np.ndarray:
    n = min(int(sr * ms / 1000), len(a), len(b))
    if n <= 1:
        return np.concatenate([a, b])
    t = np.linspace(0, np.pi / 2, n)[:, None]
    mid = a[-n:] * np.cos(t) + b[:n] * np.sin(t)
    return np.concatenate([a[:-n], mid, b[n:]])
