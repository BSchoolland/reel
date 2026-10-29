# reel

The shared rig for code-built music videos: everything the director agents behind grok-hype, rocket-cars, ssd-brawl and
the V2V videos rebuilt in each project, pulled into one library. A video project keeps only its brief, art and shots.

- `GUIDE.md`: what a director reads first (project layout, the library, what the critics keep flagging).
- `CRITIC.md`: the blind critic's prompt template (`{CANDIDATE}`, `{PROJECT}`, `{WORK}`, `{OUT}`).
- `tests/`: end to end on a tiny click-track project (`.venv/bin/python -m pytest tests`).

Setup: `uv venv --python 3.10 .venv && uv pip install --python .venv/bin/python -e . pytest`. Needs ffmpeg, and
`codex` logged in for `reel.art`.

## License

MIT. The bundled fonts (Inter, DM Sans, IBM Plex Mono, Space Mono) are under the SIL Open Font License.
