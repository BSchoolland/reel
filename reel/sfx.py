"""Quiet synthesized SFX under the song, and the mix the renders are muxed with.

    from reel import sfx
    sfx.mix([(28.0, "click", 0.2, 0.0), (39.5, sfx.blast(2.0), 0.3, -0.2)])

A cue is (time, kind, gain, pan): kind is a name in KINDS or a mono float signal at SR; pan -1 (left) .. 1 (right).
A mute hard-silences the song over [a, b] with millisecond fades. Don't use it for drama: dropouts in the music read as
glitches. It's for trimming a song's own dead air.
Writes work/mix.wav, which renders then use instead of the bare song.
"""
import subprocess

import numpy as np

SR = 48000


def _t(d):
    return np.arange(int(d * SR)) / SR


def _norm(s):
    return s / np.abs(s).max()


def clang(d=1.4, seed=0):
    """Metal impact: inharmonic partials plus a noise strike."""
    t = _t(d)
    s = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t * k) for f, a, k in
            ((233, 1.0, 3.5), (611, 0.7, 4.5), (1129, 0.5, 5.5), (1874, 0.35, 7), (2903, 0.25, 9)))
    s += np.random.RandomState(seed).randn(len(t)) * np.exp(-t * 40) * 0.6
    return _norm(s)


def pop(f0=900, f1=180, d=0.09, seed=0):
    """A pitch-dropping pop (hatch, cork, cartoon appear)."""
    t = _t(d)
    ph = 2 * np.pi * np.cumsum(f0 * (f1 / f0) ** (t / d)) / SR
    return _norm(np.sin(ph) * np.exp(-t * 35) + np.random.RandomState(seed).randn(len(t)) * np.exp(-t * 200) * 0.4)


def whoosh(d=0.7, up=True, seed=0):
    """Swept low-passed noise, rising (up) or falling."""
    t = _t(d)
    w = np.random.RandomState(seed).randn(len(t))
    fc = 300 + 3000 * (t / d if up else 1 - t / d)
    a = 1 - np.exp(-2 * np.pi * fc / SR)
    out = np.empty_like(w); y = 0.0
    for i in range(len(w)):
        y += a[i] * (w[i] - y); out[i] = y
    return _norm(out * np.sin(np.pi * np.clip(t / d, 0, 1)) ** 1.5)


def thud(d=0.6, seed=0):
    """A low body hit (landing, door, drop)."""
    t = _t(d)
    return _norm(np.sin(2 * np.pi * 55 * t * (1 - 0.3 * t)) * np.exp(-t * 7)
                 + np.random.RandomState(seed).randn(len(t)) * np.exp(-t * 60) * 0.5)


def bandnoise(d, f0, f1, decay, seed=0):
    """Noise band-limited to [f0, f1] Hz with an exponential decay."""
    t = _t(d)
    S = np.fft.rfft(np.random.RandomState(seed).randn(len(t)))
    f = np.fft.rfftfreq(len(t), 1 / SR)
    S *= (f > f0) & (f < f1)
    return _norm(np.fft.irfft(S, len(t)) * np.exp(-t * decay))


def blast(d=1.4, seed=0):
    """Thruster or cold-gas blast: broadband hiss that thumps in and decays."""
    t = _t(d)
    s = bandnoise(d, 150, 9000, 2.6, seed) * (1 - np.exp(-t * 300)) + np.sin(2 * np.pi * 48 * t) * np.exp(-t * 9) * 0.8
    return _norm(s)


def spool(d=7.0, seed=0):
    """Spool-up: a rising whine and growing hiss over d seconds, cut hard at the end."""
    t = _t(d)
    ph = 2 * np.pi * np.cumsum(180 * 9 ** (t / d)) / SR
    env = (t / d) ** 1.6
    return _norm((np.sin(ph) * 0.6 + np.sin(ph * 2.01) * 0.25) * env + bandnoise(d, 2000, 8000, 0.0, seed) * env * 0.35)


def click(seed=0):
    t = _t(0.05)
    return _norm(np.random.RandomState(seed).randn(len(t)) * np.exp(-t * 300) + np.sin(2 * np.pi * 2400 * t) * np.exp(-t * 200))


KINDS = {
    "blast": blast, "spool": spool, "click": click, "clang": clang, "clang_s": lambda: clang(0.9), "thud": thud,
    "thud_s": lambda: thud(0.4), "pop": pop, "pop_hi": lambda: pop(1400, 300, 0.07), "pop_lo": lambda: pop(500, 120, 0.12),
    "clack": lambda: pop(700, 300, 0.06), "flip": lambda: pop(1600, 600, 0.05), "whoosh": whoosh,
    "whoosh_down": lambda: whoosh(0.7, up=False), "hiss": lambda: bandnoise(2.2, 300, 7000, 1.8),
    "screech": lambda: bandnoise(0.6, 1800, 3200, 5.0),
}


def mute_gain(n, a, b, fade=0.006):
    """Per-sample gain: 1, fading to 0 over [a, b] (fade in seconds before a, half that after b)."""
    t = np.arange(n) / SR
    g = np.where((t > a) & (t < b), 0.0, 1.0).astype(np.float32)
    fi = (t >= a - fade) & (t <= a)
    g[fi] = np.minimum(g[fi], (a - t[fi]) / fade)
    fo = (t >= b) & (t <= b + fade * 0.5)
    g[fo] = np.minimum(g[fo], (t[fo] - b) / (fade * 0.5))
    return g


def limit(x, ceil_db=-3.0, look=0.005, hold=0.03):
    """Lookahead peak limiter for an (n, channels) signal: no sample above ceil_db dBFS, everything under it untouched.
    The ceiling sits below 0 dB because the AAC encode overshoots sample peaks: a mix peaking at 0 dBFS comes out of
    the master clipping. Gain dips start `look` s before a peak and hold `hold` s after it."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    c = 10 ** (ceil_db / 20)
    pk = np.abs(x).max(axis=1) if x.ndim == 2 else np.abs(x)
    g = np.minimum(1.0, c / np.maximum(pk, 1e-9))
    L = max(1, int(look * SR))
    g = minimum_filter1d(g, 2 * L + 1)
    g = minimum_filter1d(g, max(1, int(hold * SR)) | 1)
    g = uniform_filter1d(g, L + 1)
    y = x * (g[:, None] if x.ndim == 2 else g)
    return np.clip(y, -c, c).astype(np.float32)


def mix(cues, mutes=(), song=None, out=None):
    """Song + mutes + cues -> a 16-bit stereo wav (default the project's song -> work/mix.wav). Returns the out path."""
    if song is None or out is None:
        from .project import project
        song = song or project().song
        out = out or project().work / "mix.wav"
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(song), "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"],
                       capture_output=True, check=True)
    x = np.frombuffer(p.stdout, np.float32).reshape(-1, 2).copy()
    n = len(x)
    for a, b in mutes:
        x *= mute_gain(n, a, b)[:, None]
    for t0, kind, gain, pan in cues:
        sig = KINDS[kind]() if isinstance(kind, str) else np.asarray(kind, np.float32)
        i = int(t0 * SR)
        if not 0 <= i < n:
            raise ValueError(f"cue at {t0}s is outside the song (0..{n / SR:.2f}s)")
        j = min(n, i + len(sig))
        x[i:j, 0] += sig[:j - i] * gain * (1 - max(0, pan))
        x[i:j, 1] += sig[:j - i] * gain * (1 + min(0, pan))
    pk = float(np.abs(x).max())
    x = limit(x)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ac", "2", "-ar", str(SR), "-i", "-", "-c:a", "pcm_s16le",
                    str(out)], input=x.astype(np.float32).tobytes(), check=True)
    print(f"mix {out}: {n / SR:.2f}s, {len(cues)} cues, {len(mutes)} mutes, peak {pk:.2f}")
    return out
