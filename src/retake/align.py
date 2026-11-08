"""Turn Miragea's segment-level transcript into sentence-level timings.

Miragea returns the full transcript plus short segments whose boundaries can
fall mid-word ("cen" | "ts"). We locate every segment inside the full text,
interpolate a time for every character, then split the text into sentences.
Sentence boundaries are later snapped to real pauses in the audio (see dsp).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

_SENT_RE = re.compile(r"[^.!?…]+(?:[.!?…]+[\"'”’)\]]*|$)", re.S)


@dataclass
class Sentence:
    index: int
    text: str
    start: float
    end: float

    def to_dict(self) -> dict:
        return asdict(self)


def _norm(ch: str) -> str:
    return ch.lower() if ch.isalnum() else ""


def char_times(text: str, segments: list[dict]) -> list[float | None]:
    """Return an estimated time (seconds) for each character of ``text``."""
    times: list[float | None] = [None] * (len(text) + 1)
    # Build an index of alphanumeric characters of the full text.
    alnum_pos = [i for i, c in enumerate(text) if c.isalnum()]
    alnum_str = "".join(text[i].lower() for i in alnum_pos)
    cursor = 0
    for seg in segments:
        s_txt = "".join(_norm(c) for c in seg.get("text", ""))
        if not s_txt:
            continue
        found = alnum_str.find(s_txt, cursor)
        if found < 0:  # tolerate small differences: anchor on the first 4 chars
            found = alnum_str.find(s_txt[:4], cursor)
            if found < 0:
                continue
        a, b = found, min(found + len(s_txt), len(alnum_pos)) - 1
        n = max(1, b - a)
        for k in range(a, b + 1):
            frac = (k - a) / n
            times[alnum_pos[k]] = float(seg["start"]) + frac * (float(seg["end"]) - float(seg["start"]))
        cursor = b + 1
    # Fill gaps by linear interpolation between known points.
    known = [i for i, t in enumerate(times) if t is not None]
    if not known:
        return times
    for i in range(len(times)):
        if times[i] is None:
            left = max((k for k in known if k < i), default=None)
            right = min((k for k in known if k > i), default=None)
            if left is None:
                times[i] = times[right]
            elif right is None:
                times[i] = times[left]
            else:
                f = (i - left) / (right - left)
                times[i] = times[left] + f * (times[right] - times[left])
    return times


_ABBREV = re.compile(r"(?:\b[A-Z]|\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|No|Inc|Ltd|U\.S))\.$")


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Return (start, end) character spans of sentences, whitespace-trimmed.

    Initials and common abbreviations ("O. Henry", "Dr. Smith") do not end a sentence."""
    spans: list[tuple[int, int]] = []
    for m in _SENT_RE.finditer(text):
        a, b = m.start(), m.end()
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if a < b and any(c.isalnum() for c in text[a:b]):
            if spans and _ABBREV.search(text[spans[-1][0]:spans[-1][1]]):
                spans[-1] = (spans[-1][0], b)
            else:
                spans.append((a, b))
    return spans


def sentences_from_transcript(transcript: dict, duration: float | None = None) -> list[Sentence]:
    text = transcript.get("text") or " ".join(s.get("text", "") for s in transcript.get("segments", []))
    segs = transcript.get("segments", [])
    times = char_times(text, segs)
    out = []
    for idx, (a, b) in enumerate(split_sentences(text)):
        first = next(i for i in range(a, b) if text[i].isalnum())
        last = max(i for i in range(a, b) if text[i].isalnum())
        start, end = times[first], times[last]
        if start is None or end is None:
            continue
        out.append(Sentence(idx, text[a:b], round(start, 3), round(end, 3)))
    if out and segs:  # the last character is spoken a little before its segment ends
        out[-1].end = round(max(out[-1].end, float(segs[-1]["end"])), 3)
    if duration:
        for s in out:
            s.end = min(s.end, duration)
    return out


def find_sentence(sentences: list[Sentence], needle: str) -> Sentence:
    """Find the one sentence that contains ``needle`` (case/punctuation-insensitive)."""
    key = "".join(_norm(c) for c in needle)
    hits = [s for s in sentences if key and key in "".join(_norm(c) for c in s.text)]
    if not hits:
        raise LookupError(f'"{needle}" was not found in the transcript. Run `retake transcribe` to see the exact words.')
    if len(hits) > 1:
        lines = "; ".join(f"#{s.index} at {s.start:.1f}s" for s in hits)
        raise LookupError(f'"{needle}" appears in {len(hits)} sentences ({lines}). Add more words or use --line N.')
    return hits[0]


def apply_replacement(sentence: str, old: str, new: str) -> str:
    """Replace ``old`` with ``new`` inside ``sentence``, ignoring case and spacing differences."""
    pattern = r"\s+".join(re.escape(w) for w in old.split())
    result, n = re.subn(pattern, new, sentence, count=1, flags=re.I)
    if n == 0:
        raise LookupError(f'"{old}" was not found verbatim in: {sentence}')
    return result
