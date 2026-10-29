"""Image generation through codex (gpt-6.1-sol's image tool by default), with the guards every video needed.

    python -m reel.art gen NAME [--refs a.png,b.png]   art/prompts/NAME.txt -> art/NAME.png (+ art/orig/NAME.png)
    python -m reel.art queue FILE [--par 3]            one "NAME [refs]" per line (# comments), PAR at once

Guards:
  - if REEL_ART_CAP is set, at most that many launches per cycle; the cycle is REEL_CYCLE (default 1), counted in
    art/launched-c<cycle>.txt
  - waits while the Codex 5-hour window is at or above 85%, until it resets
  - after a 401 or a usage-limit 429 every later call refuses (exit 3) until that line is removed from art/gen.log:
    re-login or wait, then clear it
Every outcome is one line in art/gen.log. Exit 0 = image written.
"""
import argparse, json, os, re, shutil, subprocess, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL = os.environ.get("REEL_IMAGE_MODEL", "gpt-6.1-sol")
CAP = int(os.environ["REEL_ART_CAP"]) if "REEL_ART_CAP" in os.environ else None
CYCLE = os.environ.get("REEL_CYCLE", "1")
METER_MAX = 85
METER_TRIES = 5
CODEX_HOME = Path.home() / ".codex"

EXIT_OK, EXIT_FAIL, EXIT_CAP, EXIT_BLOCKED, EXIT_METER = 0, 1, 2, 3, 4
_launch = threading.Lock()


def art_dir():
    from .project import project
    return project().art


def log(line):
    with open(art_dir() / "gen.log", "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {line}\n")


def meter():
    """(5h used %, weekly used %, seconds until the 5h window resets) from the Codex usage endpoint."""
    tok = json.loads((CODEX_HOME / "auth.json").read_text())["tokens"]
    req = urllib.request.Request("https://chatgpt.com/backend-api/wham/usage", headers={
        "Authorization": "Bearer " + tok["access_token"], "ChatGPT-Account-Id": tok.get("account_id", ""),
        "User-Agent": "codex-cli"})
    r = json.load(urllib.request.urlopen(req, timeout=20))["rate_limit"]
    return r["primary_window"]["used_percent"], r["secondary_window"]["used_percent"], r["primary_window"]["reset_after_seconds"]


def wait_for_meter(name):
    for attempt in range(1, METER_TRIES + 1):
        try:
            p, w, reset = meter()
        except Exception as e:
            log(f"{name} meter unreadable ({attempt}/{METER_TRIES}): {e}")
            if attempt == METER_TRIES:
                return False
            time.sleep(60)
            continue
        log(f"{name} meter 5h={p}% week={w}%")
        if p < METER_MAX:
            return True
        log(f"{name} sleeping {reset}s for the 5h reset")
        time.sleep(reset + 60)
    return False


def blocked():
    g = art_dir() / "gen.log"
    return g.exists() and re.search(r" (AUTH|CAP429) ", g.read_text())


def gen(name, refs=None):
    d = art_dir()
    prompt = d / "prompts" / f"{name}.txt"
    if not prompt.exists():
        log(f"{name} FAIL no prompt at {prompt}")
        return EXIT_FAIL
    if blocked():
        log(f"{name} REFUSED: an earlier AUTH/CAP429 line in gen.log")
        return EXIT_BLOCKED
    if not wait_for_meter(name):
        log(f"{name} METER: the Codex usage meter could not be read")
        return EXIT_METER
    launched = d / f"launched-c{CYCLE}.txt"
    with _launch:
        launched.touch()
        if CAP is not None and len(launched.read_text().splitlines()) >= CAP:
            log(f"{name} REFUSED: cycle {CYCLE} already launched {CAP} images")
            return EXIT_CAP
        with open(launched, "a") as f:
            f.write(f"{name} {time.strftime('%H:%M:%S')}\n")
    (d / "logs").mkdir(exist_ok=True)
    (d / "orig").mkdir(exist_ok=True)
    out = d / "logs" / f"{name}.log"
    cmd = ["codex", "exec", *(["-i", refs] if refs else []), "--skip-git-repo-check", "-m", MODEL,
           "-c", "model_reasoning_effort=low", "-"]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, input=prompt.read_text(), capture_output=True, text=True, timeout=900, cwd=d)
        text, rc = r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        text, rc = f"{e.stdout or ''}{e.stderr or ''}\nTIMEOUT after 900s", "timeout"
    out.write_text(text)
    m = re.search(r"session id:\s*(\S+)", text)
    imgs = sorted((CODEX_HOME / "generated_images" / m.group(1)).glob("*.png"), key=os.path.getmtime) if m else []
    if imgs:
        shutil.copy(imgs[-1], d / f"{name}.png")
        shutil.copy(imgs[-1], d / "orig" / f"{name}.png")
        log(f"{name} ok in {time.time() - t0:.0f}s")
        return EXIT_OK
    if re.search(r"401|unauthorized", text, re.I):
        log(f"{name} AUTH rc={rc}: codex is logged out")
        return EXIT_BLOCKED
    if re.search(r"429|usage_limit", text, re.I):
        log(f"{name} CAP429 rc={rc}")
        return EXIT_BLOCKED
    log(f"{name} FAIL rc={rc}: no image in the session (see logs/{name}.log)")
    return EXIT_FAIL


def queue(path, par):
    jobs = []
    for raw in Path(path).read_text().splitlines():
        s = raw.split("#", 1)[0].split()
        if s:
            jobs.append((s[0], s[1] if len(s) > 1 else None))
    with ThreadPoolExecutor(par) as ex:
        codes = list(ex.map(lambda j: gen(*j), jobs))
    bad = [(n, c) for (n, _), c in zip(jobs, codes) if c]
    print(f"queue {path}: {len(jobs) - len(bad)}/{len(jobs)} ok" + (f"; failed {bad} (see art/gen.log)" if bad else ""))
    return EXIT_FAIL if bad else EXIT_OK


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen"); g.add_argument("name"); g.add_argument("--refs")
    q = sub.add_parser("queue"); q.add_argument("file"); q.add_argument("--par", type=int, default=3)
    a = ap.parse_args(argv)
    if a.cmd == "gen":
        code = gen(a.name, a.refs)
        print(f"{a.name}: exit {code} (see art/gen.log)")
        return code
    return queue(a.file, a.par)


if __name__ == "__main__":
    sys.exit(main())
