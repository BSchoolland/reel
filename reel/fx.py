"""Flat-vector effects: rocket/thruster plumes, cartoon smoke puffs and particle clouds of them, star fields.
Everything is a pure function of time and a seed, so any frame renders on its own."""
import math

import numpy as np, cv2

from .core import W, H, SUB, SHIFT, hexc, noise1, ease_out, glow, poly, circle, _mask_comp

PLUME_COLS = [hexc("#8b5cf6"), hexc("#ff7a1a"), hexc("#ffc23d"), hexc("#fff4dc")]


def plume(cv, x, y, ang, length, width, t, seed=0, intensity=1.0, cols=None, glow_k=1.0, glow_col=None):
    """Plume from nozzle (x, y) pointing along `ang` (deg, 90 = down): four nested flickering teardrops, outer to core
    (`cols`, default violet -> orange -> yellow -> white), plus an additive glow."""
    cols = cols or PLUME_COLS
    a = math.radians(ang)
    d = np.array([math.cos(a), math.sin(a)]); n = np.array([-d[1], d[0]])
    if glow_k > 0:
        gl = length * 0.55
        glow(cv, x + d[0] * gl * 0.5, y + d[1] * gl * 0.5, max(width * 2.2, gl),
             hexc("#ff8a3a") if glow_col is None else glow_col, 0.35 * intensity * glow_k)
    for k, (col, wk, lk) in enumerate(zip(cols, (1.0, 0.78, 0.55, 0.3), (1.0, 0.82, 0.6, 0.35))):
        f = 1 + 0.12 * noise1(t * 22 + k * 3.1, seed) + 0.06 * math.sin(t * 61 + k + seed)
        L = length * lk * f * intensity ** 0.5
        wd = width * wk * (1 + 0.08 * noise1(t * 30 + k, seed + 9))
        s = np.linspace(0, 1, 24)
        prof = wd / 2 * (np.sqrt(np.clip(s * 6, 0, 1)) * (1 - s) ** 0.8 + 0.15 * (1 - s))  # bell at the nozzle, tapering
        wob = 0.04 * width * np.sin(s * 9 + t * 40 + k)
        left = [np.array([x, y]) + d * (si * L) + n * (pi + wo) for si, pi, wo in zip(s, prof, wob)]
        right = [np.array([x, y]) + d * (si * L) - n * (pi - wo) for si, pi, wo in zip(s, prof, wob)]
        poly(cv, np.array(left + right[::-1]), col, alpha=0.95 if k else 0.85)


def puff(cv, x, y, r, color, alpha=1.0, seed=0, lobes=5):
    """A flat cartoon smoke/steam puff: the union of overlapping circles, composited once (uniform alpha)."""
    rng = np.random.RandomState(seed)
    circ = [(x, y, r)]
    for _ in range(lobes):
        a = rng.uniform(0, 2 * math.pi)
        circ.append((x + math.cos(a) * r * 0.7, y + math.sin(a) * r * 0.55, r * rng.uniform(0.45, 0.75)))
    x0 = int(max(0, min(c[0] - c[2] for c in circ) - 2)); x1 = int(min(cv.shape[1], max(c[0] + c[2] for c in circ) + 2))
    y0 = int(max(0, min(c[1] - c[2] for c in circ) - 2)); y1 = int(min(cv.shape[0], max(c[1] + c[2] for c in circ) + 2))
    if x1 <= x0 or y1 <= y0 or alpha <= 0:
        return
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for cx, cy, rr in circ:
        cv2.circle(m, (int(round((cx - x0) * SUB)), int(round((cy - y0) * SUB))), int(rr * SUB), 255, -1, cv2.LINE_AA, SHIFT)
    _mask_comp(cv[y0:y1, x0:x1], m, color, alpha)


class Puffs:
    """Deterministic cloud of puffs emitted at t0 from (x, y): each flies out at a random angle and speed under drag and
    gravity, grows and fades over its life."""

    def __init__(self, n, seed, x, y, t0, speed=(100, 300), up=0.0, r=(20, 50), life=(1.2, 2.2), grow=2.0, gravity=0.0,
                 color=(0.75, 0.75, 0.85), ang=(0, 360), stagger=0.0, drag=1.5):
        rng = np.random.RandomState(seed)
        self.p = []
        for _ in range(n):
            a = math.radians(rng.uniform(*ang))
            sp = rng.uniform(*speed)
            self.p.append(dict(x=x, y=y, vx=math.cos(a) * sp, vy=math.sin(a) * sp - up, r=rng.uniform(*r),
                               life=rng.uniform(*life), t0=t0 + rng.uniform(0, stagger), s=rng.randint(1e6)))
        self.grow, self.gravity, self.color, self.drag = grow, gravity, np.asarray(color, np.float32), drag

    def draw(self, cv, t, dx=0.0, dy=0.0, scale=1.0, alpha=1.0, color=None):
        col = self.color if color is None else np.asarray(color, np.float32)
        for p in self.p:
            u = t - p["t0"]
            if u < 0 or u > p["life"]:
                continue
            k = (1 - math.exp(-self.drag * u)) / self.drag
            x = p["x"] + p["vx"] * k + dx
            y = p["y"] + p["vy"] * k + 0.5 * self.gravity * u * u + dy
            f = u / p["life"]
            r = p["r"] * (1 + self.grow * ease_out(f, 2)) * scale
            puff(cv, x, y, r, col, alpha * (1 - f) ** 1.5, seed=p["s"], lobes=4)


def stars(cv, seed, n, t=0.0, ox=0.0, oy=0.0, rmin=0.6, rmax=2.2, twinkle=True, color=(0.9, 0.93, 1.0), alpha=1.0,
          scale=1.0):
    """Crisp star dots on a wrapping field; (ox, oy) scroll in px (parallax), scale zooms about the centre."""
    rng = np.random.RandomState(seed)
    xs = rng.uniform(0, W, n); ys = rng.uniform(0, H, n)
    rs = rmin + (rmax - rmin) * rng.uniform(0, 1, n) ** 3
    ph = rng.uniform(0, 6.28, n)
    cols = rng.uniform(0, 1, n)
    for x, y, r, p, c in zip(xs, ys, rs, ph, cols):
        X = ((x + ox) % W - W / 2) * scale + W / 2
        Y = ((y + oy) % H - H / 2) * scale + H / 2
        if not (-5 < X < W + 5 and -5 < Y < H + 5):
            continue
        a = alpha * ((0.6 + 0.4 * math.sin(t * 3 + p)) if twinkle else 1.0)
        col = np.asarray(color, np.float32) * (0.8 + 0.2 * c) if c < 0.8 else hexc("#a9c6ff")
        rr = r * max(1.0, scale ** 0.5)
        if rr < 1.0:
            cv[int(Y) % H, int(X) % W] += col * a * rr
        else:
            circle(cv, X, Y, rr, col, a)
