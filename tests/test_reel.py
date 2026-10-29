"""End to end on a tiny project: a 6 s click track at 120 BPM and one shot of a ball that cuts on the beat."""
import json, os, subprocess, sys
from pathlib import Path

import numpy as np
import pytest

PY = sys.executable
# the subprocesses run from the test project, so put this checkout ahead of whatever reel the venv has installed
REPO = str(Path(__file__).parents[1])
ENV = {**os.environ, "PYTHONPATH": os.pathsep.join([REPO, os.environ.get("PYTHONPATH", "")])}
BPM, DUR = 120, 6.0
SHOTS = '''
import reel.core as C
from reel.music import beats

B = beats()
CUT = [0.0, B.beat(4), B.beat(8)]


def shot_of(i):
    return sum(i / C.FPS >= s for s in CUT) - 1


def frame(i):
    t = i / C.FPS
    k = shot_of(i)
    cv = C.canvas(C.hexc(["#000000", "#ffffff", "#2040ff"][k]))
    C.circle(cv, C.W / 2 + 120 * C.noise1(t * 2, C.seed("ball")), C.H / 2 - 100 * abs(C.noise1(t * 3)), 40, (1, 0.8, 0.2))
    C.text(cv, f"SHOT {k}", 20, 20, 40)
    return C.finish(cv)
'''


@pytest.fixture(scope="module")
def proj(tmp_path_factory):
    d = tmp_path_factory.mktemp("proj")
    (d / "assets").mkdir(); (d / "mv").mkdir()
    sr = 22050
    x = np.zeros(int(DUR * sr), np.float32)
    for k in range(int(DUR * BPM / 60)):
        i = int((0.25 + k * 60 / BPM) * sr)
        n = np.arange(2000)
        x[i:i + 2000] += np.sin(2 * np.pi * 1000 * n / sr) * np.exp(-n / 300)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-", str(d / "assets/song.wav")],
                   input=x.tobytes(), check=True)
    (d / "video.toml").write_text('song = "assets/song.wav"\nsize = [320, 320]\n')
    (d / "mv/shots.py").write_text(SHOTS)
    return d


def run(proj, *args):
    r = subprocess.run([PY, "-m", *args], cwd=proj, capture_output=True, text=True, env={**ENV, "PROCS": "2"})
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_music_finds_the_grid(proj):
    run(proj, "reel.music", "--bpm", "118")
    a = json.loads((proj / "work/audio.json").read_text())
    assert abs(a["bpm"] - BPM) < 0.5
    assert abs(a["phase"] - 0.25) < 0.03
    assert len(a["onsets"]) >= 10


def test_mix_keeps_length_and_mutes(proj):
    code = ("from reel import sfx; sfx.mix([(1.0, 'thud', 0.3, 0.0), (2.0, 'pop', 0.2, 0.5)], mutes=[(3.2, 3.6)])")
    r = subprocess.run([PY, "-c", code], cwd=proj, capture_output=True, text=True, env=ENV)
    assert r.returncode == 0, r.stderr
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(proj / "work/mix.wav"), "-f", "f32le", "-ac", "1", "-ar", "48000", "-"],
                       capture_output=True, check=True)
    y = np.frombuffer(p.stdout, np.float32)
    assert abs(len(y) / 48000 - DUR) < 0.01
    assert np.abs(y[int(3.25 * 48000):int(3.55 * 48000)]).max() == 0


def test_render_measure_deliver(proj):
    out = proj / "out.mp4"
    run(proj, "reel.render", "out", str(out))
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
                               capture_output=True, text=True, check=True).stdout)
    assert abs(dur - DUR) < 0.1
    times = json.loads((proj / "work/chunks/times.json").read_text())
    assert sorted(times) == [str(k) for k in range((round(DUR * 30) + 59) // 60)], times
    m = run(proj, "reel.qa", "measure", str(out))
    assert "cuts=2 " in m, m
    assert "cuts on an onset (within 50 ms): 2/2" in m, m
    run(proj, "reel.qa", "motion", str(out))
    run(proj, "reel.render", "sheet", str(proj / "work/sheet"), "--step", "1")
    assert (proj / "work/sheet/sheet_00.jpg").exists()
    share = proj / "share.mp4"
    run(proj, "reel.deliver", str(out), str(share), "--mb", "1", "--size", "320")
    assert share.stat().st_size < 1024 * 1024


def test_draft_is_half_size(proj):
    out = proj / "draft.mp4"
    run(proj, "reel.render", "out", str(out), "--draft")
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=width,height:format=duration",
                        "-of", "json", str(out)], capture_output=True, text=True, check=True)
    info = json.loads(r.stdout)
    assert (info["streams"][0]["width"], info["streams"][0]["height"]) == (160, 160)
    assert abs(float(info["format"]["duration"]) - DUR) < 0.1
    assert (proj / "work/chunks-draft/times.json").exists()


def test_seed_is_stable_across_processes(proj):
    code = "import reel.core as C; print(C.seed('teal'))"
    a = [subprocess.run([PY, "-c", code], cwd=proj, capture_output=True, text=True, check=True, env=ENV).stdout
         for _ in range(2)]
    assert a[0] == a[1]


def test_art_failure_reasons():
    from reel.art import failure
    mod = 'http 400 Bad Request: "code": "moderation_blocked" ... request ID 1002ed90-7c58-4aaf-8b87-a8fa14294072'
    assert failure(mod) == "moderation"
    assert failure("stream error: unexpected status 429 Too Many Requests") == "limit"
    assert failure('{"type": "usage_limit_reached"}') == "limit"
    assert failure("http 401 Unauthorized") == "auth"
    assert failure("session id: 01a185bd-4d51-7429-bf5b-acf577509db0\ndone, no tool call") is None


def test_limit_caps_peaks_and_leaves_the_rest():
    from reel import sfx
    n = sfx.SR * 2
    x = np.zeros((n, 2), np.float32)
    x[:, 0] = 0.3 * np.sin(np.arange(n) / sfx.SR * 2 * np.pi * 220)
    x[:, 1] = x[:, 0]
    x[sfx.SR: sfx.SR + 2000] += 2.5  # a hit far over full scale
    y = sfx.limit(x, ceil_db=-2.0)
    assert np.abs(y).max() <= 10 ** (-2 / 20) + 1e-6
    far = slice(0, sfx.SR // 2)
    assert np.allclose(y[far], x[far])
