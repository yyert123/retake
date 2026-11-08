"""retake - fix a flubbed voiceover line in your own voice, from the terminal."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .core import fix, transcribe


def _fmt(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="retake",
        description="Fix a wrong word in a recorded voiceover by re-voicing only that sentence in the same voice "
                    "(engine: Miragea, https://miragea.ai). Loudness, EQ and room tone are matched automatically.")
    p.add_argument("--version", action="version", version=f"retake {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe", help="list the take's sentences with line numbers and timestamps")
    t.add_argument("take", type=Path)
    t.add_argument("--force", action="store_true", help="ignore the cached transcript and pay to re-transcribe")

    f = sub.add_parser("fix", help="re-voice the sentence(s) you changed and splice them in")
    f.add_argument("take", type=Path)
    f.add_argument("--replace", action="append", default=[], metavar="OLD", help="words to change (repeatable)")
    f.add_argument("--with", dest="with_", action="append", default=[], metavar="NEW", help="replacement words")
    f.add_argument("--line", action="append", type=int, default=[], metavar="N", help="sentence number to rewrite")
    f.add_argument("--say", action="append", default=[], metavar="TEXT", help="full new text for --line")
    f.add_argument("-o", "--output", type=Path, help="output file (default: <take>.fixed.wav)")
    v = f.add_argument_group("voice (pick one)")
    v.add_argument("--voice", help="voice id saved in your Miragea account")
    v.add_argument("--ref-url", help="public URL of a clip of the same speaker")
    v.add_argument("--ref-text", help="exact words spoken in --ref-url (cheaper: 1x instead of 3x)")
    v.add_argument("--host", metavar="hf:USER/REPO",
                   help="auto-cut a reference clip from this take and host it on your PUBLIC Hugging Face dataset")
    f.add_argument("--model", default="expressive", choices=["expressive", "standard"])
    f.add_argument("--no-room-tone", action="store_true")
    f.add_argument("--no-eq", action="store_true")
    f.add_argument("--keep-patches", action="store_true", help="also save each generated patch as its own file")

    a = p.parse_args(argv)
    try:
        if a.cmd == "transcribe":
            data = transcribe(a.take, force=a.force)
            for s in data["sentences"]:
                print(f"[{s['index']:>3}] {_fmt(s['start'])}-{_fmt(s['end'])}  {s['text']}")
            print(f"\n{len(data['sentences'])} lines, cached in {a.take.name}.retake.json (re-runs are free).",
                  file=sys.stderr)
            return 0
        if len(a.replace) != len(a.with_):
            p.error("every --replace needs a matching --with")
        if len(a.line) != len(a.say):
            p.error("every --line needs a matching --say")
        res = fix(a.take, list(zip(a.replace, a.with_)), lines=list(zip(a.line, a.say)), output=a.output,
                  voice=a.voice, ref_url=a.ref_url, ref_text=a.ref_text, host=a.host, model=a.model,
                  room=not a.no_room_tone, eq=not a.no_eq, keep_patches=a.keep_patches)
        for e in res.edits:
            print(f"fixed {_fmt(e.start)}: \"{e.old_text}\"\n   -> \"{e.new_text}\"  (gain {e.gain_db:+.1f} dB, "
                  f"now at {_fmt(e.new_start)})")
        print(f"\nwrote {res.output}\n      {res.report}\n      {res.markers} (import as markers in your editor)")
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - show a clean message, never a traceback, to end users
        print(f"retake: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
