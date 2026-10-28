"""Character acting, pure in time t: beat-synced bobs, deterministic blinks and eye-darts, talk tilt while a line is up,
hop arcs and landing squash. Feed the results to whatever draws the character (scale, offset, eye positions)."""
import math

from .core import smooth, lerp, clamp, spring
from .music import beats


def bob(t, amp=10.0, off=0.0, squash=0.10, every=1):
    """Bounce once per `every` beats, contact on the beat. -> (dy, sx, sy)."""
    B = beats()
    x = ((t - B.phase) / (B.period * every) + off) % 1.0
    h = 4 * x * (1 - x)
    s = math.exp(-min(x, 1 - x) * 14)
    return -amp * h, 1 + squash * s, 1 - squash * s


def blink(t, seed=0, period=3.2):
    """0 open .. 1 shut; ~0.14 s blinks at pseudo-random intervals."""
    k = math.floor((t + seed * 1.37) / period)
    jit = (math.sin(k * 12.9898 + seed * 78.233) * 43758.5453) % 1.0
    d = t - (k * period + jit * (period - 0.4) - seed * 1.37)
    return 1 - abs(d - 0.07) / 0.07 if 0 <= d < 0.14 else 0.0


def look(t, seed=0, amp=0.07, hold=1.1):
    """Eye-dart target (dx, dy): holds, then snaps (0.07 s) to a new spot."""
    k = math.floor((t + seed * 0.71) / hold)
    def tgt(i):
        a = (math.sin(i * 91.7 + seed * 3.1) * 43758.5453) % 1.0
        b = (math.sin(i * 47.3 + seed * 5.7) * 24634.6345) % 1.0
        return ((a - 0.5) * 2 * amp, (b - 0.5) * amp)
    u = (t + seed * 0.71) / hold - k
    p0, p1 = tgt(k - 1), tgt(k)
    f = smooth(u / 0.07 * hold) if u * hold < 0.07 else 1.0
    return lerp(p0[0], p1[0], f), lerp(p0[1], p1[1], f)


def talk(t, t0, t1, amp=4.0, rate=2.6):
    """Talk tilt (deg) while a line is up in [t0, t1)."""
    if not (t0 <= t < t1):
        return 0.0
    env = smooth((t - t0) / 0.12) * smooth((t1 - t) / 0.12)
    return amp * env * math.sin(2 * math.pi * rate * (t - t0)) * (0.7 + 0.3 * math.sin(t * 7.1))


def hop(u, x0, y0, x1, y1, dur, height):
    """Position along a hop arc from (x0, y0) to (x1, y1), u in seconds since takeoff."""
    f = clamp(u / dur)
    return lerp(x0, x1, f), lerp(y0, y1, f) - height * 4 * f * (1 - f)


def land_squash(u, k=0.28, freq=3.0, damp=7.0):
    """(sx, sy) after a landing at u = 0."""
    if u < 0:
        return 1.0, 1.0
    s = k * spring(u, freq, damp)
    return 1 + s, 1 - s


def pose(t, seed=0, base=None, talk_win=None, bob_amp=0.0, look_amp=0.07, blinkp=3.2, off=0.0, squash=0.06):
    """Idle acting: blink + eye-dart + optional beat bob (squash) + talk tilt over talk_win [(t0, t1), ...].
    -> (pose dict with blink, look, tilt, sx, sy; vertical offset dy)."""
    p = dict(base or {})
    p.setdefault("blink", blink(t, seed, blinkp))
    p.setdefault("look", look(t, seed, look_amp))
    p["tilt"] = p.get("tilt", 0.0) + sum(talk(t, a, b) for a, b in (talk_win or ()))
    dy = 0.0
    if bob_amp:
        dy, sx, sy = bob(t, bob_amp, off, squash)
        p["sx"] = p.get("sx", 1.0) * sx
        p["sy"] = p.get("sy", 1.0) * sy
    return p, dy
