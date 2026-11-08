"""Retake pipeline: transcribe once (cached), find the line, regenerate it in the
same voice with Miragea, match loudness/EQ/room tone, splice at real pauses."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import dsp
from .align import Sentence, apply_replacement, find_sentence, sentences_from_transcript
from .miragea import Client, MirageaError


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def cache_path(take: Path) -> Path:
    return take.with_name(take.name + ".retake.json")


def transcribe(take: str | Path, client: Client | None = None, force: bool = False) -> dict:
    """Transcribe ``take`` with Miragea; cached next to the file so you only pay once."""
    take = Path(take)
    if not take.exists():
        raise FileNotFoundError(f"{take} does not exist.")
    cp = cache_path(take)
    digest = file_hash(take)
    if cp.exists() and not force:
        data = json.loads(cp.read_text())
        if data.get("sha") == digest:
            return data
    client = client or Client()
    # Upload a compact mono 16 kHz FLAC: faster upload, same transcript, same price.
    with tempfile.TemporaryDirectory() as td:
        small = Path(td) / (take.stem + ".flac")
        audio, sr = dsp.load(take, sr=16000, channels=1)
        dsp.save(small, audio, sr)
        duration = len(audio) / sr
        result = client.transcribe(small)
    sentences = sentences_from_transcript(result, duration)
    data = {"version": 1, "sha": digest, "file": take.name, "duration": round(duration, 3),
            "language": result.get("language"), "text": result.get("text"),
            "segments": result.get("segments", []), "sentences": [s.to_dict() for s in sentences]}
    cp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return data


@dataclass
class Edit:
    old_text: str
    new_text: str
    start: float          # original span in the take (seconds)
    end: float
    new_start: float = 0.0  # span in the fixed file
    new_end: float = 0.0
    gain_db: float = 0.0
    sentence_index: int = -1


@dataclass
class FixResult:
    output: Path
    edits: list[Edit] = field(default_factory=list)
    report: Path | None = None
    markers: Path | None = None


def _pick_reference(sentences: list[Sentence], exclude: set[int], target: float = 9.0) -> tuple[float, float, str]:
    """Pick 5-12 s of contiguous, unedited sentences to clone the voice from."""
    best = None
    for i in range(len(sentences)):
        if i in exclude:
            continue
        j, text = i, []
        while j < len(sentences) and j not in exclude:
            text.append(sentences[j].text)
            span = sentences[j].end - sentences[i].start
            if span >= target or j == len(sentences) - 1 or (j + 1) in exclude:
                break
            j += 1
        span = sentences[j].end - sentences[i].start
        score = -abs(span - target) - (5 if span < 4 else 0) - (8 if span > 15 else 0)
        if best is None or score > best[0]:
            best = (score, sentences[i].start, sentences[j].end, " ".join(text))
    if best is None:
        raise MirageaError("No unedited speech is left to clone the voice from.",
                           "Pass --voice VOICE_ID or --ref-url with a clip of the same speaker.")
    return best[1], best[2], best[3]


def host_reference_on_hf(clip: Path, repo: str) -> str:
    """Upload a reference clip to a PUBLIC Hugging Face dataset you own; returns its URL.

    Used only when you opt in with --host hf:<user>/<repo> (Miragea needs a public URL until
    its upload endpoint ships). The clip becomes public: use it only for your own voice."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise MirageaError("--host hf needs huggingface_hub.", "pip install 'retake-vo[hf]'") from None
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    name = f"retake-ref/{file_hash(clip)}{clip.suffix}"
    api.upload_file(path_or_fileobj=str(clip), path_in_repo=name, repo_id=repo, repo_type="dataset",
                    commit_message="retake: voice reference clip")
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{name}"


def fix(take: str | Path, replacements: list[tuple[str, str]] | None = None, *,
        lines: list[tuple[int, str]] | None = None, output: str | Path | None = None,
        voice: str | None = None, ref_url: str | None = None, ref_text: str | None = None,
        host: str | None = None, model: str = "expressive", room: bool = True, eq: bool = True,
        client: Client | None = None, keep_patches: bool = False) -> FixResult:
    """Fix one or more lines of a recorded voiceover.

    replacements: [(old words, new words)] - the sentence containing the old words is re-voiced.
    lines:        [(sentence index, full new sentence text)].
    voice / ref_url(+ref_text) / host="hf:user/repo": where the voice comes from.
    """
    take = Path(take)
    client = client or Client()
    meta = transcribe(take, client)
    sentences = [Sentence(**s) for s in meta["sentences"]]
    targets: dict[int, tuple[str, str]] = {}
    for old, new in replacements or []:
        s = find_sentence(sentences, old)
        base = targets.get(s.index, (s.text, s.text))[1]
        targets[s.index] = (s.text, apply_replacement(base, old, new))
    for idx, new_text in lines or []:
        if not 0 <= idx < len(sentences):
            raise LookupError(f"Line {idx} does not exist; the take has lines 0-{len(sentences) - 1}.")
        targets[idx] = (sentences[idx].text, new_text)
    if not targets:
        raise ValueError("Nothing to fix: give --replace OLD --with NEW, or --line N --say TEXT.")

    audio, sr = dsp.load(take)
    m = dsp.mono(audio)
    duration = len(audio) / sr
    out_path = Path(output) if output else take.with_name(take.stem + ".fixed.wav")
    work = Path(tempfile.mkdtemp(prefix="retake-"))

    # Voice source: saved voice > explicit URL > opt-in hosting of a clip from this take.
    reference_text = ref_text
    if not voice and not ref_url:
        if not host:
            raise MirageaError(
                "Retake needs to know which voice to use.",
                "Use one of: --voice VOICE_ID (a voice saved in your Miragea account), "
                "--ref-url https://... (a public clip of the same speaker), or "
                "--host hf:<your-hf-user>/<repo> (auto-cut a clip from this take and host it on your "
                "public Hugging Face dataset; needs HF_TOKEN).")
        a, b, reference_text = _pick_reference(sentences, set(targets))
        clip = work / "reference.wav"
        dsp.save(clip, audio[int(a * sr):int(b * sr)], sr)
        if not host.startswith("hf:"):
            raise ValueError("--host must look like hf:<user>/<repo>.")
        ref_url = host_reference_on_hf(clip, host[3:])

    # Cut points: snap each sentence edge to the pause next to it.
    order = sorted(targets)
    edits: list[Edit] = []
    for idx in order:
        s = sentences[idx]
        prev_end = sentences[idx - 1].end if idx > 0 else 0.0
        next_start = sentences[idx + 1].start if idx + 1 < len(sentences) else duration
        a = dsp.snap_to_pause(m, sr, s.start, max(prev_end, s.start - 0.45), s.start + 0.12)
        b = dsp.snap_to_pause(m, sr, s.end, s.end - 0.05, min(next_start, s.end + 0.6))
        if edits and a < edits[-1].end:  # adjacent edited sentences: never overlap the cuts
            a = edits[-1].end
        edits.append(Edit(targets[idx][0], targets[idx][1], round(a, 3), round(b, 3), sentence_index=idx))

    # Generate, clean and match each patch.
    pieces, cursor = [], 0
    pos = 0.0
    for e in edits:
        raw = client.speak(e.new_text, work / f"patch_{e.sentence_index}.wav", actor_id=voice,
                           reference_audio_url=ref_url, reference_text=reference_text if ref_url else None,
                           model=model, fmt="wav")
        patch, _ = dsp.load(raw, sr=sr, channels=audio.shape[1])
        patch = dsp.trim_silence(patch, sr)
        original = audio[int(e.start * sr):int(e.end * sr)]
        if eq and len(original) > sr * 0.5:
            patch = dsp.match_eq(patch, original, sr)
        patch, e.gain_db = dsp.match_loudness(patch, original, sr)
        lead = dsp.room_tone(audio, sr, int(0.06 * sr))
        tail = dsp.room_tone(audio, sr, int(0.08 * sr))
        body = np.concatenate([lead, patch, tail]) if room else patch
        if room:
            body[len(lead):len(lead) + len(patch)] += dsp.room_tone(audio, sr, len(patch))
        before = audio[cursor:int(e.start * sr)]
        pieces.append(before)
        pos += len(before) / sr
        e.new_start = round(pos, 3)
        pieces.append(body)
        pos += len(body) / sr
        e.new_end = round(pos, 3)
        cursor = int(e.end * sr)
        if keep_patches:
            dsp.save(out_path.with_name(f"{out_path.stem}.patch{e.sentence_index}.wav"), patch, sr)
    pieces.append(audio[cursor:])

    result = pieces[0]
    for p in pieces[1:]:
        result = dsp.crossfade_join(result, p, sr, 15)
    if out_path.exists() and out_path.resolve() == take.resolve():
        raise ValueError("Refusing to overwrite the original take; choose another --output.")
    dsp.save(out_path, result, sr)

    report = out_path.with_suffix(".edits.json")
    report.write_text(json.dumps({
        "source": take.name, "output": out_path.name, "engine": "Miragea (https://miragea.ai)",
        "model": model, "voice": voice or ("reference_url" if ref_url else None),
        "duration_before": round(duration, 3), "duration_after": round(len(result) / sr, 3),
        "edits": [e.__dict__ for e in edits]}, indent=2, ensure_ascii=False))
    markers = out_path.with_suffix(".markers.csv")
    rows = ["Name,Start,End,Comment"] + [
        f'Retake {i + 1},{e.new_start:.3f},{e.new_end:.3f},"{e.new_text.replace(chr(34), chr(39))}"'
        for i, e in enumerate(edits)]
    markers.write_text("\n".join(rows) + "\n")
    return FixResult(out_path, edits, report, markers)
