"""The video being made: its directory and video.toml. One project per process.

video.toml (every key but song is optional):
    song = "assets/song.mp3"      # the soundtrack; the video is its length unless duration is set
    duration = 89.48
    size = [1080, 1080]
    fps = 30
    audio = "work/mix.wav"        # what renders are muxed with (default: work/mix.wav once it exists, else song)
    shots = "mv/shots.py"         # defines frame(i) -> float32 RGB canvas, optionally shot_of(i)
    [fonts]                       # extra or overriding font files, relative to the project
    hand = "fonts/Caveat.ttf"

The project is REEL_PROJECT if set, else the nearest directory at or above cwd holding video.toml.
"""
import json, math, os, subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


@dataclass(frozen=True)
class Project:
    root: Path
    song: Path
    duration: float
    size: tuple
    fps: int
    shots: Path
    fonts: dict = field(default_factory=dict)
    audio_override: Path | None = None

    @property
    def art(self) -> Path:
        return self.root / "art"

    @property
    def work(self) -> Path:
        return self.root / "work"

    @property
    def audio(self) -> Path:
        if self.audio_override:
            return self.audio_override
        mix = self.work / "mix.wav"
        return mix if mix.exists() else self.song

    @property
    def nframes(self) -> int:
        return int(math.ceil(self.duration * self.fps))


def find_root() -> Path:
    env = os.environ.get("REEL_PROJECT")
    if env:
        p = Path(env).expanduser().resolve()
        if not (p / "video.toml").exists():
            raise FileNotFoundError(f"REEL_PROJECT={env} has no video.toml")
        return p
    here = Path.cwd().resolve()
    for d in (here, *here.parents):
        if (d / "video.toml").exists():
            return d
    raise FileNotFoundError(f"no video.toml at or above {here} and REEL_PROJECT is unset")


def probe_duration(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
                       capture_output=True, text=True, check=True)
    return float(json.loads(r.stdout)["format"]["duration"])


@lru_cache(maxsize=1)
def project() -> Project:
    root = find_root()
    cfg = tomllib.loads((root / "video.toml").read_text())
    if "song" not in cfg:
        raise KeyError(f"{root}/video.toml must set song")
    song = root / cfg["song"]
    if not song.exists():
        raise FileNotFoundError(f"song {song} does not exist")
    return Project(
        root=root,
        song=song,
        duration=float(cfg["duration"]) if "duration" in cfg else probe_duration(song),
        size=tuple(cfg.get("size", (1080, 1080))),
        fps=int(cfg.get("fps", 30)),
        shots=root / cfg.get("shots", "mv/shots.py"),
        fonts={k: root / v for k, v in cfg.get("fonts", {}).items()},
        audio_override=root / cfg["audio"] if "audio" in cfg else None,
    )
