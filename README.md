# Retake: fix a voiceover mistake in your own voice (open-source Overdub alternative)

**Said "Wednesday" instead of "Thursday"? Don't re-record.** Retake re-voices only the sentence you
changed, in the same voice, and splices it back into your recording with matched loudness, tone (EQ) and
room tone. Works with any editor (Premiere, Resolve, Final Cut, Audacity, CapCut) because it simply gives
you a fixed WAV plus edit markers. CLI for humans, MCP server for AI agents. Voice engine:
[Miragea](https://miragea.ai).

```bash
pip install retake-vo
export MIRAGEA_API_KEY=your_key            # https://miragea.ai
retake transcribe take.wav                 # numbered sentences + timestamps (cached, pay once)
retake fix take.wav --replace "Wednesday" --with "Thursday" --voice YOUR_VOICE_ID
# -> take.fixed.wav, take.fixed.edits.json, take.fixed.markers.csv
```

<!-- mcp-name: io.github.yyert123/retake -->

> **Status (v0.1.1, 2026-11-08):** transcription, sentence finding, pause-snapped cut points, EQ/loudness/
> room-tone matching, splicing, exports, CLI and MCP server are tested (9 offline tests + live Miragea
> transcription on real recordings). A listen-to-it before/after demo will be added to the
> [demo page](https://huggingface.co/spaces/headvox/retake) as soon as the voice side is verified end to end.

## What it does automatically

1. **Transcribes** the take with Miragea speech recognition and splits it into sentences (initials like
   "O. Henry" handled). The transcript is cached next to the file, so later fixes cost nothing extra.
2. **Finds the sentence** that contains your words; refuses to guess if they appear twice.
3. **Finds real cut points**: sentence edges are snapped to the quietest stretch of the pause around them,
   so cuts never land inside a word, even though recognition segments can split words.
4. **Re-voices** the edited sentence with Miragea in your saved voice, or cloned from a reference clip.
5. **Matches it to the take**: trims silence, matches the long-term spectrum (EQ, 1/3-octave smoothed,
   clamped), matches speech loudness, lays your own room tone under it, equal-power crossfades the seams.
6. **Exports** a 24-bit WAV at the original sample rate and channels, a JSON edit list (old/new text,
   old/new times, gain), and a CSV of markers you can import into your editor. The original file is never
   modified.

## Why Retake (honest comparison)

| | **Retake** | Descript Overdub / Regenerate | DaVinci Resolve 21 Speech Generator | Reasonate (REAPER) | Manual ElevenLabs workflow |
|---|---|---|---|---|---|
| Price | Free, MIT; pay-as-you-go voice | Subscription | Resolve Studio ($295) | Free + ElevenLabs plan | ElevenLabs plan |
| Works with your editor | Any (WAV + markers) | Only inside Descript | Only inside Resolve | Only inside REAPER | Any, by hand |
| Finds the line for you | Yes, by words or line number | Yes | No, you place it | Yes (forced alignment) | No |
| Cut points at real pauses | Automatic | Automatic | Manual | Automatic | Manual |
| Loudness match | Automatic | Automatic | Manual | Automatic | Manual |
| EQ / tone match | Automatic | Partial | Manual | No | Manual (the most-cited pain) |
| Room tone under patch | Automatic | Partial | No | No | Manual |
| Length per fix | Up to 2,000 characters | - | ~350 characters | - | - |
| AI agents (MCP) | Yes, `retake-mcp` | No | No | No | No |
| Voice engine | Miragea (saved voice or clone) | Descript | Resolve built-in voices / custom WAV | ElevenLabs only | ElevenLabs |

Sources: Resolve limits per [davinciresolveclub.com](https://davinciresolveclub.com/davinci-resolve-text-to-speech/);
Reasonate per its [README](https://github.com/b451c/Reasonate); the EQ-matching pain per an
[r/editors thread](https://www.reddit.com/r/editors/comments/15s6lau/).

## Choosing the voice

| Option | When | Example |
|---|---|---|
| `--voice ID` | You saved your voice in your Miragea account (best: consistent, no upload) | `--voice v_abc123` |
| `--ref-url URL --ref-text "..."` | You have a public clip of the same speaker; giving its exact words uses the cheaper 1x rate | `--ref-url https://raw.githubusercontent.com/you/voice/main/me.wav` |
| `--host hf:user/repo` | Fully automatic: Retake cuts ~9 s of untouched speech from the take and uploads it to **your public** Hugging Face dataset (needs `pip install "retake-vo[hf]"` and `HF_TOKEN`) | `--host hf:me/voice-refs` |

Reference URLs must be public, served directly (no long signed redirects) and under 1,024 characters.
Miragea has agreed to add a direct upload endpoint; when it ships, Retake will upload the clip from your
take itself and these options become optional.

## More examples

```bash
# several fixes in one pass (each sentence re-voiced once)
retake fix vo.mp3 --replace "twenty twenty-five" --with "twenty twenty-six" --replace "Jon" --with "John" --voice v1

# rewrite a whole sentence by its number from `retake transcribe`
retake fix vo.wav --line 7 --say "Click Export, then choose MP4." --voice v1 -o vo.v2.wav

# keep each generated patch as its own file (for manual placement in a DAW)
retake fix vo.wav --replace "fourty" --with "forty" --voice v1 --keep-patches
```

## Use it from Claude, Cursor or Codex (MCP)

No install step needed with [uv](https://docs.astral.sh/uv/):

```json
{ "mcpServers": { "retake": { "command": "uvx", "args": ["retake-vo"], "env": { "MIRAGEA_API_KEY": "your_key" } } } }
```

Or `pip install retake-vo` and use `"command": "retake-mcp"`.

Then ask: *"In ~/Videos/launch/vo.wav, change 'Wednesday' to 'Thursday' using my voice v_abc123."*
Tools: `transcribe_take(path)` and `fix_line(path, old_words, new_words | line, new_sentence, voice_id | ref_url | host)`.

## Python API

```python
from retake import fix, transcribe
lines = transcribe("take.wav")["sentences"]
res = fix("take.wav", [("Wednesday", "Thursday")], voice="v_abc123")
print(res.output, res.edits[0].new_start)
```

## FAQ

**Will people hear the patch?** Retake does everything an editor would do by hand: cut in the pause,
match level, tone and room noise, crossfade. Quality still depends on the clone; a saved Miragea voice
made from a clean 10-20 s sample of the same mic gives the closest match. Listen before you publish.

**Does it change the rest of my audio?** No. Everything outside the edited sentences is copied sample for
sample; later audio shifts by the length difference, which the edit list and markers report.

**What does it cost?** Transcription: 3 Miragea points per second of audio, once per file (cached).
Each fix: Miragea's per-character rate for the new sentence (1x with a saved voice or exact reference text).

**Is my key safe?** It is read from `MIRAGEA_API_KEY` only, never logged or written to disk.

**Can I fix someone else's recording?** Only with the speaker's consent. Retake is built for fixing your own
narration or a talent's read you are licensed to edit.

**Which formats?** Anything ffmpeg reads (WAV, MP3, M4A, FLAC, MP4, MOV...). Output is WAV by default; use
`-o fixed.mp3` for MP3.

**Error messages?** Every error says what went wrong and how to fix it (missing key, ffmpeg not installed,
word not found, word found twice, balance, rate limit). Re-running is always safe.

## Requirements

Python 3.9+, ffmpeg on PATH, a [Miragea](https://miragea.ai) API key.

## Related

- [miragea-skills](https://github.com/yyert123/miragea-skills): Agent Skills for captions and Remotion narration.
- [HeadVox](https://github.com/yyert123/headvox): script to voiceover with word-aligned captions.

MIT License.
