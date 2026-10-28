"""The critic's measurements, for the director to run on its own cut first. They work on any mp4, no project needed.

    python -m reel.qa measure VIDEO    cut rhythm (scene cuts > 0.3, flash pairs merged), how many cuts land on audio
                                       onsets, frame-to-frame change and the near-static share
    python -m reel.qa motion VIDEO     optical-flow split per frame: static / camera-only / local motion
    python -m reel.qa sheets VIDEO OUT 1 fps contact sheets, timestamped, 30 frames per sheet -> OUT_NN.jpg

"Motion comic of AI stills" is the critics' standing complaint; `motion` is how they measure it. local% is frames
where something moves on its own, beyond what a single camera move explains.
"""
import argparse, re, statistics as st, subprocess, sys

import numpy as np

from . import music


def duration(fn):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", fn],
                                capture_output=True, text=True, check=True).stdout)


def scene_cuts(fn, thr=0.3, merge=0.07):
    txt = subprocess.run(["ffmpeg", "-hide_banner", "-i", fn, "-vf", f"select='gt(scene,{thr})',showinfo", "-an",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    cuts = []
    for x in (float(v) for v in re.findall(r"pts_time:([\d.]+)", txt)):
        if not cuts or x - cuts[-1] > merge:
            cuts.append(x)
    return cuts


def measure(fn):
    dur = duration(fn)
    cuts = scene_cuts(fn)
    edges = [0.0] + cuts + [dur]
    shots = [b - a for a, b in zip(edges, edges[1:])]
    print(f"cuts={len(cuts)} shots={len(shots)} mean={st.mean(shots):.2f}s median={st.median(shots):.2f}s "
          f">3s={sum(s > 3 for s in shots)}")
    print("cuts per 15 s:", [sum(1 for c in cuts if a <= c < a + 15) for a in range(0, int(dur), 15)])
    print("held over 3 s:", [(round(a, 1), round(b - a, 1)) for a, b in zip(edges, edges[1:]) if b - a > 3])
    t, f = music.flux(fn)
    on, _ = music.peaks(t, f)
    if cuts and len(on):
        off = np.array([on[np.abs(on - c).argmin()] - c for c in cuts])
        near = np.abs(off) <= 0.05
        print(f"cuts on an onset (within 50 ms): {near.sum()}/{len(cuts)} ({near.mean() * 100:.0f}%); "
              f"median offset of those {np.median(off[near]) * 1000 if near.any() else float('nan'):+.0f} ms "
              f"(negative = cut lands after the hit)")
        late = [(round(c, 2), round(o * 1000)) for c, o in zip(cuts, off) if not abs(o) <= 0.05]
        print("off-onset cuts (time, ms to nearest onset):", late[:40])
    txt = subprocess.run(["ffmpeg", "-hide_banner", "-i", fn, "-vf", "scale=640:-2,tblend=all_mode=difference,signalstats,"
                          "metadata=print", "-an", "-f", "null", "-"], capture_output=True, text=True).stderr
    v = [float(x) for x in re.findall(r"YAVG=([\d.]+)", txt)]
    print(f"frame change: median YAVG {st.median(v):.2f}, near-static (<0.5) {100 * sum(x < 0.5 for x in v) / len(v):.0f}%")


def motion(fn):
    import cv2
    cap = cv2.VideoCapture(fn)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps:
        raise ValueError(f"{fn}: cannot read its frame rate")
    near_cut = set()
    for c in scene_cuts(fn, merge=0):
        k = int(round(c * fps))
        near_cut.update((k - 1, k, k + 1))
    W, H = 320, 180
    ys, xs = np.mgrid[0:H:4, 0:W:4]
    A = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], 1).astype(np.float32)
    prev, rows, i = None, [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            mad = float(np.mean(cv2.absdiff(g, prev)))
            if mad < 0.25:
                rows.append((i, mad, 0.0, 0.0))
            else:
                fl = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                u, v = fl[::4, ::4, 0].ravel(), fl[::4, ::4, 1].ravel()
                cu = np.linalg.lstsq(A, u, rcond=None)[0]; cv_ = np.linalg.lstsq(A, v, rcond=None)[0]
                rows.append((i, mad, float(np.mean(np.hypot(A @ cu, A @ cv_))), float(np.mean(np.hypot(u - A @ cu, v - A @ cv_)))))
        prev = g; i += 1
    if not rows:
        raise ValueError(f"{fn}: no frames decoded")
    a = np.array(rows)
    b = a[[int(k) not in near_cut for k in a[:, 0]]]
    mad, glob, res = b[:, 1], b[:, 2], b[:, 3]
    static = mad < 0.25
    local = ~static & (res >= 0.25)
    cam = ~static & (glob > 0.3) & (res < 0.25)
    print(f"{fn}: frames={len(b)} static={static.mean() * 100:.0f}% camera-only={cam.mean() * 100:.0f}% "
          f"local={local.mean() * 100:.0f}% subtle={(~static & ~cam & ~local).mean() * 100:.0f}%  median residual={np.median(res):.3f}")
    fr = b[:, 0].astype(int)
    print(f"   local on even frames {local[fr % 2 == 0].mean() * 100:.0f}%, odd {local[fr % 2 == 1].mean() * 100:.0f}% "
          f"(a big gap means animation on twos)")
    sec = (fr / fps).astype(int)
    worst = sorted(((local[sec == s].mean(), s) for s in np.unique(sec)))[:8]
    print("   stillest seconds (local %, second):", [(round(m * 100), int(s)) for m, s in worst])
    np.save(fn + ".motion.npy", a)


def sheets(fn, prefix):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", fn, "-vf", "fps=1,scale=384:-2,drawtext=text='%{pts\\:hms}':x=4:y=4:"
                    "fontsize=16:fontcolor=yellow:box=1:boxcolor=black@0.6,tile=6x5", "-q:v", "3", f"{prefix}_%02d.jpg"],
                   check=True)
    print(f"sheets: {prefix}_NN.jpg")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("measure").add_argument("video")
    sub.add_parser("motion").add_argument("video")
    s = sub.add_parser("sheets"); s.add_argument("video"); s.add_argument("prefix")
    a = ap.parse_args(argv)
    {"measure": lambda: measure(a.video), "motion": lambda: motion(a.video), "sheets": lambda: sheets(a.video, a.prefix)}[a.cmd]()


if __name__ == "__main__":
    sys.exit(main())
