"""Render the project's cut. The shots module (video.toml `shots`, default mv/shots.py) defines frame(i) -> float canvas
and, optionally, shot_of(i) -> shot index for the sheet labels. Its directory is put on sys.path.

    python -m reel.render out PATH [--chunks 3,4]   full master (or just those chunks re-rendered), muxed with the audio
        [--draft]                                   the same frames encoded half size on a fast preset (check renders)
    python -m reel.render stills 1.2,40.5 [--dir D] full-size jpgs (default work/stills/)
    python -m reel.render sheet DIR [--step 0.5]    contact sheets, 30 labelled frames each
    python -m reel.render clip T0 T1 PATH           a short mp4 of [T0, T1) with audio

Chunks are CHUNK frames each, rendered in PROCS processes (env PROCS, default 4) into work/chunks/ (drafts:
work/chunks-draft/), slowest first by the times of the last render there (times.json).
"""
import argparse, importlib.util, json, os, subprocess, sys, time
import multiprocessing as mp
from functools import partial
from multiprocessing.util import Finalize

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import core as C
from .project import project

CHUNK = 60
PROCS = int(os.environ.get("PROCS", "4"))  # one per core: renders are memory-bound, and 5 or 6 were no faster
_shots = None


def shots():
    global _shots
    if _shots is None:
        path = project().shots
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location(path.stem, path)
        _shots = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = _shots
        spec.loader.exec_module(_shots)
    return _shots


def frame(i):
    f = shots().frame(i)
    out = np.empty(f.shape, np.uint8)
    for s in C._bands(len(f)):
        b = np.clip(f[s], 0, 1)
        b *= 255
        b += 0.5
        out[s] = b
    return out


def chunk_dir(draft=False):
    d = project().work / ("chunks-draft" if draft else "chunks")
    d.mkdir(parents=True, exist_ok=True)
    return d


def nchunks():
    return (C.NF + CHUNK - 1) // CHUNK


def writer(path, draft=False):
    # x264's default 1.5 threads per core cost 1.6x the CPU of 2 threads; chunks already run in parallel
    enc = (["-vf", "scale=trunc(iw/4)*2:trunc(ih/4)*2:flags=area", "-preset", "veryfast", "-crf", "20", "-threads", "1"]
           if draft else ["-preset", "medium", "-crf", "16", "-threads", "2"])
    return subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{C.W}x{C.H}",
                             "-r", str(C.FPS), "-i", "-", "-c:v", "libx264", *enc, "-pix_fmt", "yuv420p",
                             "-g", str(C.FPS), "-bf", "2", str(path)], stdin=subprocess.PIPE)


def _close(p, what):
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {what} (exit {p.returncode})")


_draining = []  # this worker's last chunk (encoder, tmp, path): x264 is still encoding its lookahead


def _drain():
    while _draining:
        p, tmp, path = _draining.pop()
        if p.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {tmp} (exit {p.returncode})")
        os.replace(tmp, path)


def _worker_init():
    Finalize(None, _drain, exitpriority=10)  # runs when the pool is closed and joined, not on terminate()


def render_chunk(k, draft=False):
    """Feed chunk k's frames to its encoder and return without waiting for x264 to drain its lookahead: that finishes
    while this worker renders its next chunk."""
    a, b = k * CHUNK, min(C.NF, (k + 1) * CHUNK)
    path = chunk_dir(draft) / f"c{k:03d}.mp4"
    path.unlink(missing_ok=True)  # so a failed encode can't leave the last render's chunk behind
    tmp = chunk_dir(draft) / f"c{k:03d}.tmp.mp4"
    t0 = time.time()
    p = writer(tmp, draft)
    for i in range(a, b):
        p.stdin.write(frame(i))
    p.stdin.close()
    _drain()
    _draining.append((p, tmp, path))
    return k, time.time() - t0


def run_chunks(ks, procs=PROCS, draft=False):
    shots()  # import once here so forked workers inherit it
    tf = chunk_dir(draft) / "times.json"
    times = {int(k): v for k, v in json.loads(tf.read_text()).items()} if tf.exists() else {}
    # slowest first, so the last chunks to finish are short ones; chunks with no time yet keep their order, in front
    ks = sorted(ks, key=lambda k: -times.get(k, float("inf")))
    t0 = time.time()
    pool = mp.Pool(procs, initializer=_worker_init)
    try:
        for k, dt in pool.imap_unordered(partial(render_chunk, draft=draft), ks):
            times[k] = round(dt, 2)
            print(f"chunk {k} {dt:.0f}s  ({time.time() - t0:.0f}s)", flush=True)
    except BaseException:
        pool.terminate()
        raise
    pool.close()
    pool.join()  # each worker drains its last encoder as it exits
    tf.write_text(json.dumps({str(k): times[k] for k in sorted(times)}))


def concat_mux(out, draft=False):
    d = chunk_dir(draft)
    missing = [k for k in range(nchunks()) if not (d / f"c{k:03d}.mp4").exists()]
    if missing:
        raise FileNotFoundError(f"chunks never rendered: {missing}")
    (d / "list.txt").write_text("".join(f"file 'c{k:03d}.mp4'\n" for k in range(nchunks())))
    vid = d / "video.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", d / "list.txt", "-c", "copy", vid],
                   check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", vid, "-i", project().audio, "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-t", f"{C.DUR:.3f}", "-movflags", "+faststart", out],
                   check=True)
    print("wrote", out, flush=True)


def stills(ts, outdir=None):
    outdir = outdir or project().work / "stills"
    os.makedirs(outdir, exist_ok=True)
    for t in ts:
        i = min(C.NF - 1, int(round(t * C.FPS)))
        Image.fromarray(frame(i)).save(f"{outdir}/s{t:07.2f}.jpg", quality=92)
    print("stills", len(ts), outdir, flush=True)


def _indexed(i):
    return i, frame(i)


def sheet(outdir, step=0.5, t0=0.0, t1=None, procs=PROCS):
    os.makedirs(outdir, exist_ok=True)
    t1 = C.NF / C.FPS if t1 is None else t1
    idx = [int(round(t * C.FPS)) for t in np.arange(t0, t1, step)]
    shot_of = getattr(shots(), "shot_of", None)
    with mp.Pool(procs) as pool:
        res = dict(pool.imap_unordered(_indexed, idx))
    F = ImageFont.truetype(str(C.FONTS["mono"]), 15)
    tw = 300
    th = round(tw * C.H / C.W)
    cols, rows = 6, 5
    per = cols * rows
    for p in range(0, len(idx), per):
        sh = Image.new("RGB", (tw * cols, th * rows))
        for n, i in enumerate(idx[p:p + per]):
            im = Image.fromarray(res[i]).resize((tw, th))
            label = f"{i / C.FPS:.2f}" + (f" #{shot_of(i)}" if shot_of else "")
            ImageDraw.Draw(im).text((4, 2), label, fill=(255, 255, 0), font=F, stroke_width=2, stroke_fill=(0, 0, 0))
            sh.paste(im, ((n % cols) * tw, (n // cols) * th))
        sh.save(f"{outdir}/sheet_{p // per:02d}.jpg", quality=85)
    print("sheets in", outdir, flush=True)


def clip(t0, t1, out):
    a, b = int(round(t0 * C.FPS)), int(round(t1 * C.FPS))
    tmp = f"{out}.v.mp4"
    p = writer(tmp)
    for i in range(a, b):
        p.stdin.write(frame(i))
    _close(p, tmp)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-ss", f"{t0:.3f}", "-t", f"{t1 - t0:.3f}",
                    "-i", project().audio, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-shortest", out],
                   check=True)
    os.remove(tmp)
    print("clip", out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("out"); o.add_argument("path"); o.add_argument("--chunks")
    o.add_argument("--draft", action="store_true")
    s = sub.add_parser("stills"); s.add_argument("times"); s.add_argument("--dir")
    h = sub.add_parser("sheet"); h.add_argument("dir"); h.add_argument("--step", type=float, default=0.5)
    c = sub.add_parser("clip"); c.add_argument("t0", type=float); c.add_argument("t1", type=float); c.add_argument("path")
    a = ap.parse_args(argv)
    if a.cmd == "out":
        run_chunks([int(x) for x in a.chunks.split(",")] if a.chunks else list(range(nchunks())), draft=a.draft)
        concat_mux(a.path, a.draft)
    elif a.cmd == "stills":
        stills([float(x) for x in a.times.split(",")], a.dir)
    elif a.cmd == "sheet":
        sheet(a.dir, a.step)
    else:
        clip(a.t0, a.t1, a.path)


if __name__ == "__main__":
    main()
