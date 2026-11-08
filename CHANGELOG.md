# Changelog

## 0.1.1 - 2026-11-08
- MCP SDK is now a regular dependency; `uvx retake-vo` starts the MCP server with zero setup.
- Adjacent edited sentences can no longer produce overlapping cuts.
- Listed in the official MCP Registry as io.github.yyert123/retake.

## 0.1.0 - 2026-11-08
- First release: `retake transcribe`, `retake fix` (by words or line number, multiple fixes per pass).
- Pause-snapped cut points, EQ match, loudness match, room tone, equal-power crossfades.
- Exports fixed WAV, JSON edit list and CSV markers; never touches the original.
- Voice from a saved Miragea voice, a public reference URL, or an auto-hosted clip on your Hugging Face dataset.
- MCP server `retake-mcp` (MCP Python SDK 1.x and 2.x).
