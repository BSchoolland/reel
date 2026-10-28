"""Share encode: two-pass H.264 + AAC that fits under a size limit (social uploads, Discord attachments).

    python -m reel.deliver MASTER OUT [--mb 9.5] [--size 1080] [--audio-kbps 96]

The video bitrate is whatever the limit leaves after the audio, with 4% headroom for the container. If that falls
under 300 kbps at the given size it refuses: pass a smaller --size (720 usually) rather than ship mush.
"""
import argparse, os, subprocess, sys, tempfile

from .qa import duration

X264 = "aq-mode=3:aq-strength=0.9:psy-rd=1.0,0.0:deblock=-1,-1:ref=5:bframes=5:qcomp=0.7:rc-lookahead=60:keyint=300"
MIN_KBPS = 300


def deliver(master, out, mb=9.5, size=1080, audio_kbps=96):
    dur = duration(master)
    vk = int(mb * 8 * 1024 * 0.96 / dur - audio_kbps)
    if vk < MIN_KBPS:
        raise ValueError(f"{mb} MB over {dur:.1f}s leaves {vk} kbps for video; try --size 720 or a bigger --mb")
    vf = f"scale={size}:-2:flags=lanczos"
    common = ["-vf", vf, "-c:v", "libx264", "-preset", "slower", "-b:v", f"{vk}k", "-x264-params", X264, "-pix_fmt", "yuv420p"]
    with tempfile.TemporaryDirectory() as d:
        pl = os.path.join(d, "pl")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", master, *common, "-pass", "1", "-passlogfile", pl,
                        "-an", "-f", "mp4", os.devnull], check=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", master, *common, "-pass", "2", "-passlogfile", pl,
                        "-c:a", "aac", "-b:a", f"{audio_kbps}k", "-movflags", "+faststart", out], check=True)
    got = os.path.getsize(out) / 1024 / 1024
    print(f"{out}: {got:.2f} MB at {vk} kbps video, {size}px")
    if got > mb:
        raise RuntimeError(f"{out} came out {got:.2f} MB, over the {mb} MB limit")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("master"); ap.add_argument("out")
    ap.add_argument("--mb", type=float, default=9.5)
    ap.add_argument("--size", type=int, default=1080)
    ap.add_argument("--audio-kbps", type=int, default=96)
    a = ap.parse_args(argv)
    deliver(a.master, a.out, a.mb, a.size, a.audio_kbps)


if __name__ == "__main__":
    sys.exit(main())
