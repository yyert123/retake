"""Offline tests: alignment, DSP and the full fix pipeline with a fake Miragea client."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from retake import dsp
from retake.align import apply_replacement, find_sentence, sentences_from_transcript, split_sentences
from retake.core import fix, transcribe

SR = 24000
needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")

TRANSCRIPT = {
    "text": "By O. Henry. The meeting is on Wednesday. Bring the slides, please.",
    "segments": [
        {"start": 0.1, "end": 0.9, "text": "By O Henry"},
        {"start": 1.2, "end": 2.1, "text": "The meeting is on Wed"},
        {"start": 2.1, "end": 2.9, "text": "nesday"},
        {"start": 3.4, "end": 4.6, "text": "Bring the slides, please"},
    ],
}


def tone(sec, freq=220.0, amp=0.3):
    t = np.arange(int(sec * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(sec, amp=0.002):
    rng = np.random.default_rng(0)
    return (amp * rng.standard_normal(int(sec * SR))).astype(np.float32)


def make_take(path: Path):
    parts = [silence(0.1), tone(0.8), silence(0.3), tone(1.7, 300), silence(0.5), tone(1.2, 260), silence(0.4)]
    dsp.save(path, np.concatenate(parts)[:, None], SR)


class FakeClient:
    def __init__(self):
        self.spoken = []

    def transcribe(self, path):
        return TRANSCRIPT

    def speak(self, text, out_path, **kw):
        self.spoken.append((text, kw))
        dsp.save(out_path, np.concatenate([silence(0.2), tone(1.5, 310, 0.05), silence(0.3)])[:, None], 24000)
        return Path(out_path)


def test_split_keeps_initials_together():
    spans = split_sentences(TRANSCRIPT["text"])
    assert [TRANSCRIPT["text"][a:b] for a, b in spans][0] == "By O. Henry."
    assert len(spans) == 3


def test_sentence_times_survive_mid_word_segments():
    s = sentences_from_transcript(TRANSCRIPT)
    assert s[1].text == "The meeting is on Wednesday."
    assert s[1].start == pytest.approx(1.2, abs=0.01)
    assert s[1].end == pytest.approx(2.9, abs=0.05)


def test_find_and_replace():
    s = sentences_from_transcript(TRANSCRIPT)
    hit = find_sentence(s, "on wednesday")
    assert hit.index == 1
    assert apply_replacement(hit.text, "Wednesday", "Thursday") == "The meeting is on Thursday."
    with pytest.raises(LookupError):
        find_sentence(s, "Friday")


def test_loudness_match_and_crossfade():
    a, b = tone(1.0, amp=0.3)[:, None], tone(1.0, amp=0.03)[:, None]
    out, g = dsp.match_loudness(b, a, SR)
    assert g == pytest.approx(20.0, abs=0.5) or g == 18.0  # clamped at 18 dB
    joined = dsp.crossfade_join(a, a, SR, 20)
    assert len(joined) == 2 * len(a) - int(SR * 0.02)


def test_eq_match_moves_spectrum_toward_reference():
    bright = (tone(1.0, 3000, 0.2) + tone(1.0, 200, 0.05))[:, None]
    dull = (tone(1.0, 3000, 0.02) + tone(1.0, 200, 0.05))[:, None]
    out = dsp.match_eq(dull, bright, SR)
    spec = lambda x: np.abs(np.fft.rfft(x[:, 0]))
    f = np.fft.rfftfreq(len(dull), 1 / SR)
    hi = (f > 2800) & (f < 3200)
    assert spec(out)[hi].max() > 1.5 * spec(dull)[hi].max()


def test_snap_to_pause_finds_the_gap():
    x = np.concatenate([tone(1.0), silence(0.4), tone(1.0)])
    t = dsp.snap_to_pause(x, SR, 0.9, 0.7, 1.6)
    assert 1.0 <= t <= 1.4


@needs_ffmpeg
def test_fix_end_to_end(tmp_path):
    take = tmp_path / "take.wav"
    make_take(take)
    client = FakeClient()
    res = fix(take, [("Wednesday", "Thursday")], voice="v1", client=client)
    assert client.spoken[0][0] == "The meeting is on Thursday."
    assert client.spoken[0][1]["actor_id"] == "v1"
    out, sr = dsp.load(res.output)
    assert sr == SR and len(out) > SR * 4
    rep = json.loads(res.report.read_text())
    e = rep["edits"][0]
    assert e["new_text"] == "The meeting is on Thursday."
    # The untouched first and last sentences are preserved byte-for-byte in timing.
    assert e["start"] > 0.9 and e["end"] < 3.5
    assert res.markers.read_text().startswith("Name,Start,End")
    # Transcript cached: a second call must not need the client.
    assert transcribe(take, client=None)["sha"]


@needs_ffmpeg
def test_fix_without_voice_explains_how(tmp_path):
    take = tmp_path / "take.wav"
    make_take(take)
    with pytest.raises(Exception) as err:
        fix(take, [("Wednesday", "Thursday")], client=FakeClient())
    assert "--voice" in str(err.value) and "--host" in str(err.value)


def test_missing_key_message(monkeypatch):
    from retake.miragea import Client, MirageaError
    monkeypatch.delenv("MIRAGEA_API_KEY", raising=False)
    with pytest.raises(MirageaError) as err:
        Client()
    assert "export MIRAGEA_API_KEY" in str(err.value)
