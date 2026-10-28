"""The song's timing, measured the way the critic measures it: spectral-flux onsets (hop 256 at 22.05 kHz) and a beat
grid fitted to the strong ones.

    python -m reel.music [--bpm 140] [SONG] [OUT]    # default: the project's song -> work/audio.json

In shots: `from reel.music import beats; B = beats()`, then B.snap(t), B.beat(k), B.phase(t), B.onsets.
"""
import argparse, json, subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

SR, HOP, NFFT = 22050, 256, 1024
GRID_TOL = 0.03


def flux(path):
    """(times, z-scored spectral flux) of an audio file."""
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                       capture_output=True, check=True)
    x = np.frombuffer(p.stdout, np.float32)
    fr = np.lib.stride_tricks.sliding_window_view(x, NFFT)[::HOP] * np.hanning(NFFT)
    S = np.log1p(np.abs(np.fft.rfft(fr, axis=1)))
    f = np.maximum(np.diff(S, axis=0), 0).sum(1)
    f = (f - f.mean()) / f.std()
    t = (np.arange(len(f)) + 1) * HOP / SR + NFFT / 2 / SR
    return t, f


def peaks(t, f, thr=1.5):
    """Local maxima of the flux above thr: (times, strengths)."""
    i = np.nonzero((f[1:-1] > thr) & (f[1:-1] >= f[:-2]) & (f[1:-1] >= f[2:]))[0] + 1
    return t[i], f[i]


def fit_grid(pt, pv, bpm_lo, bpm_hi):
    """The (score, bpm, phase) whose grid catches the most onset strength within GRID_TOL seconds; among grids that
    catch the same onsets, the one closest to them."""
    best = (-1.0, 0.0, 0.0, 0.0)
    for bpm in np.arange(bpm_lo, bpm_hi + 1e-9, 0.02):
        P = 60 / bpm
        ph = np.arange(0, P, 0.002)[:, None]
        d = (pt[None, :] - ph) / P
        err = np.abs(d - np.round(d)) * P
        hit = err < GRID_TOL
        sc = (pv[None, :] * hit).sum(1)
        miss = (pv[None, :] * err * hit).sum(1)
        k = int(np.lexsort((miss, -sc))[0])
        if sc[k] > best[0] or (sc[k] == best[0] and miss[k] < best[3]):
            best = (float(sc[k]), float(bpm), float(ph[k, 0]), float(miss[k]))
    return best[:3]


def analyze(song, out, bpm=None):
    import librosa
    t, f = flux(song)
    pt, pv = peaks(t, f)
    y, sr = librosa.load(str(song), sr=SR, mono=True)
    tempo, lb = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP, start_bpm=bpm or 120, units="time")
    centre = bpm or float(np.atleast_1d(tempo)[0])
    score, gbpm, ph = fit_grid(pt, pv, centre * 0.96, centre * 1.04)
    P = 60 / gbpm
    d = (pt - ph) / P
    err = (d - np.round(d)) * P
    on = np.abs(err) < GRID_TOL
    dur = float(len(y) / sr)
    res = {"bpm": gbpm, "phase": ph, "period": P, "duration": dur,
           "onsets": [[float(a), float(b)] for a, b in zip(pt, pv)],
           "grid": np.arange(ph, dur, P).tolist(), "librosa_tempo": centre if bpm else float(np.atleast_1d(tempo)[0]),
           "librosa_beats": [float(b) for b in lb]}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res))
    print(f"grid {gbpm:.2f} BPM, phase {ph:.3f}, period {P:.4f}; {len(pt)} strong onsets, {on.sum()} on the grid, "
          f"median error {np.median(np.abs(err[on])) * 1000 if on.any() else float('nan'):.1f} ms -> {out}")
    print("strongest onsets per 2 s (time(strength)):")
    for s in range(0, int(dur) + 1, 2):
        m = (pt >= s) & (pt < s + 2)
        top = sorted(sorted(zip(pv[m], pt[m]), reverse=True)[:4], key=lambda x: x[1])
        print(f"{s:4d}: " + "  ".join(f"{b:.2f}({a:.1f})" for a, b in top))
    return res


@dataclass(frozen=True)
class Beats:
    bpm: float
    period: float
    phase: float
    onsets: tuple  # ((time, strength), ...)

    def beat(self, k):
        """Time of grid beat k."""
        return round(self.phase + self.period * k, 3)

    def phase_at(self, t, off=0.0):
        """0..1 position of t within its beat."""
        return ((t - self.phase) / self.period + off) % 1.0

    def snap(self, t, win=0.2, thr=1.0):
        """The nearest measured onset of strength >= thr within win seconds of t, else t."""
        best = None
        for a, b in self.onsets:
            if b >= thr and abs(a - t) <= win and (best is None or abs(a - t) < abs(best - t)):
                best = a
        return round(best, 3) if best is not None else t


@lru_cache(maxsize=None)
def beats(path=None):
    """The project's work/audio.json (or path) as Beats."""
    if path is None:
        from .project import project
        path = project().work / "audio.json"
    a = json.loads(Path(path).read_text())
    return Beats(a["bpm"], a["period"], a["phase"], tuple((x, y) for x, y in a["onsets"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("song", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--bpm", type=float, help="tempo hint; the grid is fitted within 4%% of it (default: librosa's)")
    a = ap.parse_args()
    if a.song and a.out:
        song, out = a.song, a.out
    else:
        from .project import project
        song, out = a.song or project().song, project().work / "audio.json"
    analyze(song, out, a.bpm)
