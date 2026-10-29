"""Image generation through codex (gpt-6.1-sol's image tool by default).

    python -m reel.art gen NAME [--refs a.png,b.png]   art/prompts/NAME.txt -> art/NAME.png (+ art/orig/NAME.png)
    python -m reel.art queue FILE [--par 3]            one "NAME [refs]" per line (# comments), PAR at once

Every outcome is one line in art/gen.log. Exit 0 = image written, 3 = codex is logged out or out of usage.
"""
import argparse, os, re, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL = os.environ.get("REEL_IMAGE_MODEL", "gpt-6.1-sol")
CODEX_HOME = Path.home() / ".codex"

EXIT_OK, EXIT_FAIL, EXIT_BLOCKED = 0, 1, 3


def art_dir():
    from .project import project
    return project().art


def log(line):
    with open(art_dir() / "gen.log", "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {line}\n")


def gen(name, refs=None):
    d = art_dir()
    prompt = d / "prompts" / f"{name}.txt"
    if not prompt.exists():
        log(f"{name} FAIL no prompt at {prompt}")
        return EXIT_FAIL
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
    why = failure(text)
    if why == "moderation":
        log(f"{name} FAIL rc={rc}: blocked by the image model's safety filter; reword the prompt and retry")
        return EXIT_FAIL
    if why == "auth":
        log(f"{name} AUTH rc={rc}: codex is logged out")
        return EXIT_BLOCKED
    if why == "limit":
        log(f"{name} LIMIT rc={rc}: codex is out of usage")
        return EXIT_BLOCKED
    log(f"{name} FAIL rc={rc}: no image in the session (see logs/{name}.log)")
    return EXIT_FAIL


def failure(text):
    """Why a codex session made no image: moderation | auth | limit | None. Status codes match only as whole numbers
    (request ids are hex and often contain "401" or "429")."""
    if re.search(r"moderation_blocked|rejected by the safety system", text, re.I):
        return "moderation"
    if re.search(r"(?<![0-9a-f-])401(?![0-9a-f-])|unauthorized", text, re.I):
        return "auth"
    if re.search(r"(?<![0-9a-f-])429(?![0-9a-f-])|usage_limit", text, re.I):
        return "limit"
    return None


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
