"""MCP server: lets Claude, Cursor or Codex fix voiceover lines ("change Wednesday to Thursday in take.wav").

Run with:  retake-mcp   (stdio). Requires: pip install "retake-vo[mcp]".
"""
from __future__ import annotations

from pathlib import Path


def build():
    try:  # MCP Python SDK 2.x
        from mcp.server.mcpserver import MCPServer as FastMCP
    except ImportError:
        try:  # MCP Python SDK 1.x
            from mcp.server.fastmcp import FastMCP
        except ImportError as exc:  # pragma: no cover
            raise SystemExit('retake-mcp needs the MCP SDK: pip install "retake-vo[mcp]"') from exc

    from .core import fix as _fix
    from .core import transcribe as _transcribe

    mcp = FastMCP("retake")

    @mcp.tool()
    def transcribe_take(path: str) -> dict:
        """Transcribe a recorded voiceover (any audio/video file) and return its numbered sentences with
        start/end seconds. Cached next to the file, so calling it again is free."""
        data = _transcribe(Path(path).expanduser())
        return {"duration": data["duration"], "sentences": data["sentences"]}

    @mcp.tool()
    def fix_line(path: str, old_words: str = "", new_words: str = "", line: int = -1, new_sentence: str = "",
                 voice_id: str = "", ref_url: str = "", ref_text: str = "", host: str = "",
                 output: str = "") -> dict:
        """Re-voice one sentence of a recorded voiceover in the same voice and splice it in seamlessly
        (Miragea voice engine; loudness, EQ and room tone matched). Either give old_words + new_words, or
        line (from transcribe_take) + new_sentence. Voice: voice_id (saved Miragea voice), ref_url (public
        clip of the speaker), or host="hf:user/repo" to auto-host a clip from the take on a public HF dataset.
        Returns the fixed file path and where the edit landed."""
        res = _fix(Path(path).expanduser(),
                   [(old_words, new_words)] if old_words else None,
                   lines=[(line, new_sentence)] if line >= 0 else None,
                   output=output or None, voice=voice_id or None, ref_url=ref_url or None,
                   ref_text=ref_text or None, host=host or None)
        return {"output": str(res.output), "report": str(res.report), "markers": str(res.markers),
                "edits": [e.__dict__ for e in res.edits]}

    return mcp


def main() -> None:
    build().run()


if __name__ == "__main__":
    main()
