"""Frame engine: canvas, easing, cached art (premultiplied RGBA float32), affine/perspective placement, AA vector
shapes, glows, text, camera and finishing. A canvas is float32 RGB in [0, 1] at W x H.

W, H, FPS, DUR and NF come from the project's video.toml (reel.project)."""
import ctypes, math, sys, zlib
from functools import lru_cache
from pathlib import Path

import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

from .project import project

# OpenCV's worker threads spin (sched_yield) after every parallel call: with renders running one process per core
# they burned more CPU than the frames did, for no wall-clock gain.
cv2.setNumThreads(1)
if sys.platform.startswith("linux"):
    # Frame-sized temporaries die every few calls; by default glibc unmaps them and page-faults them back in. Keep
    # freed memory in the heap (allocations up to 32 MB, the most glibc allows).
    _libc = ctypes.CDLL("libc.so.6")
    for _k, _v in ((-3, 32 << 20), (-1, 1 << 30), (-2, 64 << 20)):  # M_MMAP_THRESHOLD, M_TRIM_THRESHOLD, M_TOP_PAD
        if not _libc.mallopt(_k, _v):
            raise OSError(f"mallopt({_k}, {_v}) failed")

P = project()
ROOT = str(P.root)
ART = str(P.art)
W, H = P.size
FPS = P.fps
DUR = P.duration
NF = P.nframes
SHIFT = 4
SUB = 1 << SHIFT
_LEVELS = np.arange(256, dtype=np.float32) / 255  # uint8 v -> v / 255 in float32


def canvas(color=(0, 0, 0)):
    c = np.empty((H, W, 3), np.float32)
    c[0] = color
    c[1:] = c[0]
    return c


def hexc(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)], np.float32)


# ---------------------------------------------------------------- easing / timing
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def lerp(a, b, t):
    return a + (b - a) * t


def smooth(t):
    t = clamp(t)
    return t * t * (3 - 2 * t)


def ease_out(t, p=3):
    t = clamp(t)
    return 1 - (1 - t) ** p


def ease_in(t, p=3):
    t = clamp(t)
    return t ** p


def ease_io(t, p=3):
    t = clamp(t)
    return 2 ** (p - 1) * t ** p if t < 0.5 else 1 - (-2 * t + 2) ** p / 2


def back_out(t, s=1.7):
    t = clamp(t) - 1
    return t * t * ((s + 1) * t + s) + 1


def seg(t, a, b):
    """0..1 progress of t through [a, b]."""
    return clamp((t - a) / (b - a)) if b > a else float(t >= b)


def spring(t, freq=4.0, damp=5.0):
    """Damped oscillation starting at 1 when t=0, -> 0. For settles/jiggles after an impact."""
    if t < 0:
        return 0.0
    return math.exp(-damp * t) * math.cos(2 * math.pi * freq * t)


def bounce_beat(t, beats, amp=1.0, dur=0.22):
    """Sum of short squash pulses at given beat times: returns 0..1 (1 at the beat)."""
    v = 0.0
    for b in beats:
        d = t - b
        if 0 <= d < dur:
            v = max(v, (1 - d / dur) ** 2)
    return v * amp


def seed(name):
    """A stable integer seed from a string. Never hash(): it changes every process, so frames rendered in different
    processes (chunks) disagree."""
    return zlib.crc32(name.encode())


def noise1(x, seed=0):
    """Smooth 1-D value noise in [-1, 1]."""
    i = math.floor(x)
    f = x - i
    def r(k):
        return (((k * 374761393 + seed * 668265263) ^ 0x5bd1e995) * 2246822519 % 4294967296) / 4294967296 * 2 - 1
    u = f * f * (3 - 2 * f)
    return r(i) * (1 - u) + r(i + 1) * u


# ---------------------------------------------------------------- art
def _art_path(name):
    return name if name.startswith("/") else f"{ART}/{name}.png"


@lru_cache(maxsize=None)
def art(name):
    """Premultiplied float RGBA of art/<name>.png, or of an absolute path (RGB images get alpha 1)."""
    a = cv2.LUT(np.asarray(Image.open(_art_path(name)).convert("RGBA")), _LEVELS)
    al = a[..., 3].copy()
    a *= cv2.merge((al, al, al, np.ones_like(al)))
    return a


@lru_cache(maxsize=None)
def art_crop(name, pad=4):
    """art() cropped to its alpha bbox."""
    a = art(name)
    ys, xs = np.nonzero(a[..., 3] > 0.02)
    if not len(ys):
        raise ValueError(f"art {name} is fully transparent")
    y0, y1, x0, x1 = max(0, ys.min() - pad), ys.max() + pad + 1, max(0, xs.min() - pad), xs.max() + pad + 1
    return np.ascontiguousarray(a[y0:y1, x0:x1])


_store = {}


def put(name, arr):
    """Register processed art (e.g. a keyed or recoloured sprite) under a name for get()."""
    _store[name] = arr


def get(name):
    """Art registered with put(), else art_crop(name)."""
    return _store[name] if name in _store else art_crop(name)


def has_art(name):
    return Path(_art_path(name)).exists()


# ---------------------------------------------------------------- placement
def M_place(img_hw, x, y, scale, rot=0.0, anchor=(0.5, 0.5), flip=False, sx=1.0, sy=1.0):
    """2x3 affine image px -> screen px: image point `anchor` (fractions of w, h) lands on (x, y)."""
    h, w = img_hw
    ax, ay = anchor[0] * w, anchor[1] * h
    a = math.radians(rot)
    ca, sa = math.cos(a), math.sin(a)
    fx = -1 if flip else 1
    A = np.array([[ca, -sa], [sa, ca]]) @ np.diag([scale * sx * fx, scale * sy])
    t = np.array([x, y]) - A @ np.array([ax, ay])
    return np.hstack([A, t[:, None]]).astype(np.float64)


def _bbox(pts, cv, pad=1):
    x0 = int(max(0, math.floor(pts[:, 0].min()) - pad)); x1 = int(min(cv.shape[1], math.ceil(pts[:, 0].max()) + pad))
    y0 = int(max(0, math.floor(pts[:, 1].min()) - pad)); y1 = int(min(cv.shape[0], math.ceil(pts[:, 1].max()) + pad))
    return x0, y0, x1, y1


def draw(cv, img, M, alpha=1.0, mode="over", tint=None, interp=cv2.INTER_LINEAR):
    """Warp premultiplied RGBA img by 2x3 M onto canvas cv (only the touched bbox).
    mode: over | add | screen. tint = (color, k) pulls the art's colour toward color by k."""
    h, w = img.shape[:2]
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], float) @ M.T
    x0, y0, x1, y1 = _bbox(corners, cv)
    if x1 <= x0 or y1 <= y0:
        return
    M2 = M.copy(); M2[0, 2] -= x0; M2[1, 2] -= y0
    out = cv2.warpAffine(img, M2, (x1 - x0, y1 - y0), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    _comp(cv, out, x0, y0, alpha, mode, tint)


def draw_persp(cv, img, quad, alpha=1.0, mode="over", tint=None):
    """Warp img so its corners (tl, tr, br, bl) land on quad (4x2 screen px)."""
    h, w = img.shape[:2]
    quad = np.asarray(quad, np.float32)
    x0, y0, x1, y1 = _bbox(quad, cv)
    if x1 <= x0 or y1 <= y0:
        return
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    Pm = cv2.getPerspectiveTransform(src, (quad - np.array([x0, y0], np.float32)).astype(np.float32))
    out = cv2.warpPerspective(img, Pm, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    _comp(cv, out, x0, y0, alpha, mode, tint)


# numpy runs (h, w, 3) op (h, w, 1) or (h, w, 3) op (3,) as millions of 3-element inner loops, 1.5-6x slower than the
# same arithmetic on matching contiguous arrays. The compositing below builds those (same values, same dtypes).
def _ch3(a):
    """(h, w) -> contiguous (h, w, 3) with the value in every channel."""
    return cv2.merge((a, a, a))


def _rgb(out):
    """Contiguous copy of an (h, w, 4) image's first three channels."""
    return cv2.cvtColor(out, cv2.COLOR_RGBA2RGB) if out.dtype == np.float32 else np.ascontiguousarray(out[..., :3])


def _chmul(x, col):
    """x * col for a contiguous (h, w, 3) x and a per-channel col."""
    h, w = x.shape[:2]
    return (x.reshape(h, w * 3) * np.tile(col, w)).reshape(h, w, 3)


def _mul(x, k):
    """x * k, in place when the result keeps x's dtype."""
    return np.multiply(x, k, out=x) if np.result_type(x, k) == x.dtype else x * k


def _add(x, y):
    """x + y, in place when the result keeps x's dtype."""
    return np.add(x, y, out=x) if np.result_type(x, y) == x.dtype else x + y


# Renders run one process per core and are memory-bound: big regions are composited a band of rows at a time, so
# each band's temporaries stay in cache through all the passes over it.
_BAND = 32


def _bands(n):
    return (slice(y, y + _BAND) for y in range(0, n, _BAND))


def _comp(cv, out, x0, y0, alpha=1.0, mode="over", tint=None):
    if mode not in ("over", "add", "screen"):
        raise ValueError(f"blend mode {mode}")
    reg = cv[y0:y0 + out.shape[0], x0:x0 + out.shape[1]]
    for s in _bands(len(out)):
        r, o = reg[s], out[s]
        rgb, a = _rgb(o), o[..., 3]
        if tint is not None:
            col, k = tint
            rgb = _add(_mul(rgb, 1 - k), _mul(_chmul(_ch3(a), np.asarray(col, np.float32)), k))
        if alpha != 1.0:
            rgb = _mul(rgb, alpha); a = a * alpha
        if mode == "over":
            r *= _ch3(1 - a); r += rgb
        elif mode == "add":
            r += rgb
        else:
            r[:] = 1 - (1 - r) * (1 - np.clip(rgb, 0, 1))


def _mask_comp(reg, m, color, alpha=1.0, mode="over"):
    """Flat color through uint8 coverage m: reg * (1 - a) + color * a ("over", else reg + color * a), a = m / 255 * alpha,
    looked up per coverage level."""
    a = (_LEVELS * alpha)[:, None]
    keep = np.repeat(1 - a, 3, 1)[None]
    add = (np.asarray(color, np.float32) * a)[None]
    for s in _bands(len(m)):
        m3, r = cv2.merge((m[s],) * 3), reg[s]
        if mode == "over":
            r *= cv2.LUT(m3, keep)
        r += cv2.LUT(m3, add)


def cover(cv, img, cx, cy, zoom=1.0, rot=0.0, alpha=1.0):
    """Draw a background plate so that it covers the frame at `zoom` (1 = just covers), image point (cx, cy) (fractions)
    at screen centre (clamped so no edge shows when rot == 0)."""
    h, w = img.shape[:2]
    s = max(W / w, H / h) * zoom
    if rot == 0:
        hw, hh = W / 2 / s, H / 2 / s
        px = clamp(cx * w, hw, w - hw); py = clamp(cy * h, hh, h - hh)
    else:
        px, py = cx * w, cy * h
    M = M_place((h, w), W / 2, H / 2, s, rot, anchor=(px / w, py / h))
    draw(cv, img, M, alpha)


# ---------------------------------------------------------------- vector shapes
def poly(cv, pts, color, alpha=1.0, mode="over"):
    pts = np.asarray(pts, float)
    x0, y0, x1, y1 = _bbox(pts, cv, pad=2)
    if x1 <= x0 or y1 <= y0:
        return
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(m, [np.round((pts - [x0, y0]) * SUB).astype(np.int32)], 255, cv2.LINE_AA, SHIFT)
    _mask_comp(cv[y0:y1, x0:x1], m, color, alpha, mode)


def ellipse_pts(cx, cy, rx, ry, rot=0.0, n=64, a0=0.0, a1=2 * math.pi):
    t = np.linspace(a0, a1, n, endpoint=(a1 - a0) < 2 * math.pi - 1e-6)
    p = np.stack([rx * np.cos(t), ry * np.sin(t)], 1)
    a = math.radians(rot)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return p @ R.T + [cx, cy]


def circle(cv, cx, cy, r, color, alpha=1.0, mode="over"):
    if r < 0.3:
        return
    poly(cv, ellipse_pts(cx, cy, r, r, n=max(12, min(96, int(r * 2)))), color, alpha, mode)


def rrect_pts(x0, y0, x1, y1, r, n=8):
    r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
    pts = []
    for cx, cy, a0 in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
        for t in np.linspace(a0, a0 + 90, n):
            pts.append((cx + r * math.cos(math.radians(t)), cy + r * math.sin(math.radians(t))))
    return np.array(pts)


def line(cv, p0, p1, width, color, alpha=1.0, mode="over"):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = np.hypot(*d)
    if L < 1e-6:
        return
    n = np.array([-d[1], d[0]]) / L * width / 2
    poly(cv, [p0 + n, p1 + n, p1 - n, p0 - n], color, alpha, mode)


def glow(cv, x, y, r, color, strength=1.0):
    """Additive soft radial glow (gaussian-ish falloff)."""
    glow_ellipse(cv, x, y, r, r, color, strength)


def glow_ellipse(cv, x, y, rx, ry, color, strength=1.0):
    x0, x1 = int(max(0, x - rx)), int(min(cv.shape[1], x + rx))
    y0, y1 = int(max(0, y - ry)), int(min(cv.shape[0], y + ry))
    if x1 <= x0 or y1 <= y0:
        return
    dx = ((np.arange(x0, x1, dtype=np.float32)[None, :] - x) / rx) ** 2
    dy = ((np.arange(y0, y1, dtype=np.float32)[:, None] - y) / ry) ** 2
    reg, col = cv[y0:y1, x0:x1], np.asarray(color, np.float32)
    for s in _bands(y1 - y0):
        reg[s] += _chmul(_ch3(np.exp(-(dx + dy[s]) * 4.0) * strength), col)


def vgrad(cv, top, bottom, y0=0, y1=H):
    t = np.linspace(0, 1, y1 - y0, dtype=np.float32)[:, None]
    rows = cv[y0:y1]
    rows[:, 0] = np.asarray(top, np.float32) * (1 - t) + np.asarray(bottom, np.float32) * t
    w = 1
    while w < rows.shape[1]:  # double the filled columns: long contiguous copies
        k = min(w, rows.shape[1] - w)
        rows[:, w:w + k] = rows[:, :k]
        w += k


# ---------------------------------------------------------------- text
_FONTDIR = Path(__file__).parent / "fonts"
FONTS = {
    "sans": _FONTDIR / "Inter.ttf",            # variable: weight="Regular" .. "Black"
    "dm": _FONTDIR / "DMSans.ttf",
    "mono": _FONTDIR / "IBMPlexMono-Bold.ttf",
    "monom": _FONTDIR / "IBMPlexMono-Medium.ttf",
    "spacemono": _FONTDIR / "SpaceMono-Bold.ttf",
    **P.fonts,
}


@lru_cache(maxsize=4096)
def font(kind, size, weight="Bold"):
    """A font at a size; weight applies to variable fonts (Inter: "Regular" .. "Black") and is ignored by static ones."""
    f = ImageFont.truetype(str(FONTS[kind]), size)
    if _is_variable(kind):
        f.set_variation_by_name(weight)
    return f


@lru_cache(maxsize=None)
def _is_variable(kind):
    try:
        ImageFont.truetype(str(FONTS[kind]), 10).get_variation_names()
    except OSError:  # Pillow raises this for a font with no variation axes
        return False
    return True


@lru_cache(maxsize=4096)
def text_img(s, kind, size, color, weight="Bold", tracking=0):
    """Premultiplied RGBA of a single text line (tight bbox, small pad)."""
    f = font(kind, size, weight)
    if tracking:
        widths = [f.getlength(ch) + tracking for ch in s]
        tw = int(sum(widths)) + 8
    else:
        tw = int(f.getlength(s)) + 8
    asc, desc = f.getmetrics()
    im = Image.new("L", (tw + 8, asc + desc + 8))
    d = ImageDraw.Draw(im)
    if tracking:
        x = 4
        for ch, w_ in zip(s, widths):
            d.text((x, 4), ch, font=f, fill=255)
            x += w_
    else:
        d.text((4, 4), s, font=f, fill=255)
    a = np.asarray(im, np.float32) / 255
    out = np.zeros(a.shape + (4,), np.float32)
    out[..., :3] = np.asarray(color, np.float32) * a[..., None]
    out[..., 3] = a
    return out


def text(cv, s, x, y, size, color=(1, 1, 1), kind="sans", weight="Bold", anchor=(0, 0), alpha=1.0, tracking=0,
         scale=1.0, rot=0.0):
    """Draw one line; returns its (w, h) on screen."""
    img = text_img(s, kind, size, tuple(float(c) for c in color), weight, tracking)
    draw(cv, img, M_place(img.shape[:2], x, y, scale, rot, anchor=anchor), alpha)
    return img.shape[1] * scale, img.shape[0] * scale


def text_w(s, kind, size, weight="Bold"):
    return font(kind, size, weight).getlength(s)


def wrap(s, size, maxw, kind="sans", weight="Bold"):
    """Fewest lines that fit maxw, balanced so a two-line break leaves no lonely word."""
    words = s.split()
    if text_w(s, kind, size, weight) <= maxw:
        return [s]
    best = None
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        m = max(text_w(a, kind, size, weight), text_w(b, kind, size, weight))
        if m <= maxw and (best is None or m < best[0]):
            best = (m, [a, b])
    if best:
        return best[1]
    lines, cur = [], ""
    for w in words:
        nxt = (cur + " " + w).strip()
        if text_w(nxt, kind, size, weight) <= maxw or not cur:
            cur = nxt
        else:
            lines.append(cur); cur = w
    lines.append(cur)
    return lines


# ---------------------------------------------------------------- camera and finishing
def shake(t, t0, amp, dur=0.5, freq=24, seed=3):
    """Camera shake offset (dx, dy) decaying after t0."""
    u = t - t0
    if u < 0 or u > dur:
        return 0.0, 0.0
    k = amp * (1 - u / dur) ** 2
    return k * noise1(u * freq, seed), k * noise1(u * freq + 50, seed + 1)


def apply_cam(cv, zoom=1.0, rot=0.0, dx=0.0, dy=0.0, cx=None, cy=None):
    """Whole-frame camera transform: zoom/rotate about (cx, cy) (default centre), then offset. Edges reflect, so keep
    zoom >= 1."""
    if zoom == 1.0 and rot == 0.0 and dx == 0 and dy == 0:
        return cv
    M = cv2.getRotationMatrix2D((W / 2 if cx is None else cx, H / 2 if cy is None else cy), rot, zoom)
    M[0, 2] += dx; M[1, 2] += dy
    return cv2.warpAffine(cv, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _glare(cv, thresh, radius):
    """What bloom adds, before its strength."""
    small = cv2.resize(cv, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    b = np.clip(small - thresh, 0, None)
    b = cv2.GaussianBlur(b, (0, 0), radius / 4)
    return cv2.resize(b, (W, H), interpolation=cv2.INTER_LINEAR)


def bloom(cv, thresh=0.75, strength=0.6, radius=24):
    cv += _mul(_glare(cv, thresh, radius), strength)


@lru_cache(maxsize=8)
def _vig(k):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return _ch3(1 - k * (((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / (W * W / 2)))


def vignette(cv, k=0.35):
    cv *= _vig(k)


def finish(cv, bloom_k=0.5, gamma=0.88, vig=0.32):
    """Bloom, vignette and gamma: the last step of a frame."""
    glare = _glare(cv, 0.72, 24) if bloom_k else None
    v = _vig(vig) if vig else None
    for s in _bands(len(cv)):
        c = cv[s]
        if glare is not None:
            c += _mul(glare[s], bloom_k)
        if v is not None:
            c *= v[s]
        if gamma != 1.0:
            # cv2.pow is 5x np.power, within a few float32 ulps of it (a pixel in ~10^5 lands 1/255 apart)
            cv2.pow(np.clip(c, 0, None, out=c), gamma, dst=c)
    return cv
