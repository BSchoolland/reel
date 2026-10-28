"""reel's fast compositing paths against the plain numpy they replaced: the same bits, for float32 and float64 inputs."""
import atexit, math, os, shutil, tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

_proj = Path(tempfile.mkdtemp())
atexit.register(shutil.rmtree, _proj, True)
(_proj / "song.wav").write_bytes(b"")  # only has to exist: duration is set
(_proj / "video.toml").write_text('song = "song.wav"\nduration = 2.0\nsize = [320, 240]\n')
_env = os.environ.get("REEL_PROJECT")
os.environ["REEL_PROJECT"] = str(_proj)
try:
    import reel.core as C
    import reel.fx as FX
    import reel.render as R
finally:  # the end-to-end tests' subprocesses inherit os.environ
    if _env is None:
        del os.environ["REEL_PROJECT"]
    else:
        os.environ["REEL_PROJECT"] = _env

W, H = C.W, C.H


# ---------------------------------------------------------------- the code as it was
def ref_art(path):
    a = np.asarray(Image.open(path).convert("RGBA"), np.float32) / 255
    a[..., :3] *= a[..., 3:4]
    return a


def ref_canvas(color):
    c = np.empty((H, W, 3), np.float32)
    c[:] = color
    return c


def ref_comp(cv, out, x0, y0, alpha=1.0, mode="over", tint=None):
    reg = cv[y0:y0 + out.shape[0], x0:x0 + out.shape[1]]
    rgb, a = out[..., :3], out[..., 3:4]
    if tint is not None:
        col, k = tint
        rgb = rgb * (1 - k) + np.asarray(col, np.float32) * a * k
    if alpha != 1.0:
        rgb = rgb * alpha; a = a * alpha
    if mode == "over":
        reg *= (1 - a); reg += rgb
    elif mode == "add":
        reg += rgb
    elif mode == "screen":
        reg[:] = 1 - (1 - reg) * (1 - np.clip(rgb, 0, 1))


def ref_draw(cv, img, M, alpha=1.0, mode="over", tint=None):
    h, w = img.shape[:2]
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], float) @ M.T
    x0, y0, x1, y1 = C._bbox(corners, cv)
    if x1 <= x0 or y1 <= y0:
        return
    M2 = M.copy(); M2[0, 2] -= x0; M2[1, 2] -= y0
    out = cv2.warpAffine(img, M2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ref_comp(cv, out, x0, y0, alpha, mode, tint)


def ref_poly(cv, pts, color, alpha=1.0, mode="over"):
    pts = np.asarray(pts, float)
    x0, y0, x1, y1 = C._bbox(pts, cv, pad=2)
    if x1 <= x0 or y1 <= y0:
        return
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(m, [np.round((pts - [x0, y0]) * C.SUB).astype(np.int32)], 255, cv2.LINE_AA, C.SHIFT)
    a = m.astype(np.float32)[..., None] / 255 * alpha
    reg = cv[y0:y1, x0:x1]
    col = np.asarray(color, np.float32)
    if mode == "over":
        reg *= (1 - a); reg += col * a
    else:
        reg += col * a


def ref_puff(cv, x, y, r, color, alpha=1.0, seed=0, lobes=5):
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
        cv2.circle(m, (int(round((cx - x0) * C.SUB)), int(round((cy - y0) * C.SUB))), int(rr * C.SUB), 255, -1,
                   cv2.LINE_AA, C.SHIFT)
    a = m.astype(np.float32)[..., None] / 255 * alpha
    reg = cv[y0:y1, x0:x1]
    reg *= (1 - a); reg += np.asarray(color, np.float32) * a


def ref_glow_ellipse(cv, x, y, rx, ry, color, strength=1.0):
    x0, x1 = int(max(0, x - rx)), int(min(cv.shape[1], x + rx))
    y0, y1 = int(max(0, y - ry)), int(min(cv.shape[0], y + ry))
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d2 = ((xx - x) / rx) ** 2 + ((yy - y) / ry) ** 2
    g = np.exp(-d2 * 4.0) * strength
    cv[y0:y1, x0:x1] += g[..., None] * np.asarray(color, np.float32)


def ref_vgrad(cv, top, bottom, y0=0, y1=None):
    y1 = H if y1 is None else y1
    t = np.linspace(0, 1, y1 - y0, dtype=np.float32)[:, None, None]
    cv[y0:y1] = np.asarray(top, np.float32) * (1 - t) + np.asarray(bottom, np.float32) * t


def ref_bloom(cv, thresh=0.75, strength=0.6, radius=24):
    small = cv2.resize(cv, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    b = np.clip(small - thresh, 0, None)
    b = cv2.GaussianBlur(b, (0, 0), radius / 4)
    cv += cv2.resize(b, (W, H), interpolation=cv2.INTER_LINEAR) * strength


def ref_vignette(cv, k=0.35):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cv *= (1 - k * (((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / (W * W / 2)))[..., None]


def ref_finish(cv, bloom_k=0.5, gamma=0.88, vig=0.32):
    if bloom_k:
        ref_bloom(cv, 0.72, bloom_k)
    if vig:
        ref_vignette(cv, vig)
    if gamma != 1.0:
        np.power(np.clip(cv, 0, None, out=cv), gamma, out=cv)
    return cv


# ---------------------------------------------------------------- inputs
def scene(seed):
    return np.random.default_rng(seed).random((H, W, 3), dtype=np.float32) * 1.2


def sprite(seed, h=57, w=83, dtype=np.float32):
    rng = np.random.default_rng(seed)
    a = rng.random((h, w, 1)).astype(dtype)
    a[: h // 3] = 1.0
    a[-4:] = 0.0
    img = np.concatenate([rng.random((h, w, 3)).astype(dtype) * a, a], 2)
    return np.ascontiguousarray(img)


# python floats, numpy float64 (strong in NEP 50: promotes the arithmetic to float64) and float32 scalars
SCALARS = [0.7, np.float64(0.55), np.float32(0.4), 1.0, 1]


def same(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert np.array_equal(a, b), f"max diff {np.abs(a.astype(np.float64) - b).max()}"


@pytest.mark.parametrize("color", [(0.1, 0.5, 0.9), C.hexc("#22e3ff"), np.array([0.3, 0.2, 0.1]), 0.25])
def test_canvas(color):
    same(C.canvas(color), ref_canvas(color))


@pytest.mark.parametrize("mode", ["over", "add", "screen"])
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("tint", [None, ((0.02, 0.03, 0.08), 0.45), ((1.0, 0.15, 0.2), np.float64(0.3))])
def test_draw(mode, alpha, tint):
    for k, (x, y, s, rot) in enumerate([(100.3, 80.7, 1.3, 17.0), (-20, 230, 2.1, -40.0), (W / 2, H / 2, 6.0, 0.0)]):
        img = sprite(k)
        M = C.M_place(img.shape[:2], x, y, s, rot)
        a, b = scene(k), scene(k)
        C.draw(a, img, M, alpha, mode, tint)
        ref_draw(b, img, M, alpha, mode, tint)
        same(a, b)


@pytest.mark.parametrize("alpha", SCALARS)
def test_draw_float64_image(alpha):
    img = sprite(5, dtype=np.float64)
    M = C.M_place(img.shape[:2], 150.2, 120.9, 1.7, 33.0)
    a, b = scene(5), scene(5)
    C.draw(a, img, M, alpha, tint=((0.2, 0.3, 0.4), 0.5))
    ref_draw(b, img, M, alpha, tint=((0.2, 0.3, 0.4), 0.5))
    same(a, b)


@pytest.mark.parametrize("mode", ["over", "add", "screen"])
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("color", [(0.9, 0.4, 0.1), np.array([0.2, 0.7, 0.3]), 0.6])
def test_poly(mode, alpha, color):
    rng = np.random.default_rng(1)
    for k in range(6):
        pts = rng.uniform(-60, max(W, H) + 60, (rng.integers(3, 9), 2))
        a, b = scene(k), scene(k)
        C.poly(a, pts, color, alpha, mode)
        ref_poly(b, pts, color, alpha, mode)
        same(a, b)


def test_shapes_through_poly():
    a, b = scene(2), scene(2)
    for cv, P in ((a, C.poly), (b, ref_poly)):
        P(cv, C.ellipse_pts(100.4, 90.1, 40.2, 22.8, 30), (0.2, 0.9, 0.4), 0.8)
        P(cv, C.rrect_pts(10.5, 20.5, 200.25, 140.75, 18), (0.9, 0.9, 0.1), np.float64(0.66))
    same(a, b)


@pytest.mark.parametrize("alpha", SCALARS)
def test_puff(alpha):
    for k, (x, y, r) in enumerate([(160.5, 120.2, 30.3), (5, 230, 44.0), (300.7, 10.1, 12.5)]):
        a, b = scene(k), scene(k)
        FX.puff(a, x, y, r, (0.75, 0.75, 0.85), alpha, seed=k)
        ref_puff(b, x, y, r, (0.75, 0.75, 0.85), alpha, seed=k)
        same(a, b)


@pytest.mark.parametrize("strength", SCALARS)
@pytest.mark.parametrize("pos", [(100.5, 80.25, 60.0, 40.0), (np.float64(10.3), 200.0, 700.0, 760.0),
                                 (W / 2, np.float32(H / 3), np.float64(33.3), 18.0)])
def test_glow_ellipse(strength, pos):
    x, y, rx, ry = pos
    a, b = scene(3), scene(3)
    C.glow_ellipse(a, x, y, rx, ry, C.hexc("#5b2bd6"), strength)
    ref_glow_ellipse(b, x, y, rx, ry, C.hexc("#5b2bd6"), strength)
    same(a, b)


@pytest.mark.parametrize("span", [(0, None), (30, 200), (0, 1)])
def test_vgrad(span):
    y0, y1 = span
    a, b = scene(4), scene(4)
    C.vgrad(a, C.hexc("#03040c"), (0.04, 0.06, np.float64(0.18)), y0, H if y1 is None else y1)
    ref_vgrad(b, C.hexc("#03040c"), (0.04, 0.06, np.float64(0.18)), y0, y1)
    same(a, b)


@pytest.mark.parametrize("strength", SCALARS)
def test_bloom(strength):
    a, b = scene(6), scene(6)
    C.bloom(a, 0.72, strength)
    ref_bloom(b, 0.72, strength)
    same(a, b)


@pytest.mark.parametrize("k", [0.32, 0.28, np.float64(0.3)])
def test_vignette(k):
    a, b = scene(6), scene(6)
    C.vignette(a, k)
    ref_vignette(b, k)
    same(a, b)


def quantize(x):
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


@pytest.mark.parametrize("args", [(0.5, 0.88, 0.32), (0.4, 0.92, 0.28), (0.0, 1.0, 0.0), (np.float64(0.45), 0.9, 0.3)])
def test_finish(args):
    """Everything but the gamma is bit-identical (above); cv2.pow is within float32 rounding of np.power."""
    x = scene(6)
    x[:10] = 0; x[10:20] = 1; x[20:30] = 2.5; x[30:31] = 1e-30
    a, b = C.finish(x.copy(), *args), ref_finish(x.copy(), *args)
    assert a.dtype == b.dtype
    assert np.allclose(a, b, rtol=1e-6, atol=0)
    d = np.abs(quantize(a).astype(int) - quantize(b))
    assert d.max() <= 1 and (d > 0).mean() < 1e-4


@pytest.mark.parametrize("mode", ["RGBA", "RGB", "P", "LA"])
def test_art(mode):
    rng = np.random.default_rng(7)
    px = rng.integers(0, 256, (61, 47, 4), dtype=np.uint8)
    px[:5, :, 3] = 0; px[-5:, :, 3] = 255
    im = Image.fromarray(px, "RGBA")
    im = im.convert(mode) if mode != "P" else im.convert("RGB").quantize(64)
    path = _proj / f"art_{mode}.png"
    im.save(path)
    same(C.art(str(path)), ref_art(path))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_frame_quantize(dtype, monkeypatch):
    x = (np.random.default_rng(9).random((H, W, 3)) * 1.4 - 0.2).astype(dtype)
    x[0, :4] = [126.5 / 255, 0.5, 1 / 255 * 0.5]  # ties and near-ties
    monkeypatch.setattr(R, "_shots", type("shots", (), {"frame": staticmethod(lambda i: x)}))
    same(R.frame(0), (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8))
