# Goal: make reel renders as fast as they can be, without changing how directors use it

You are working in ~/Projects/reel-opt-opus, a git worktree of ~/Projects/reel on branch `optimize-opus`. Nobody is watching
and nobody answers questions: make every call yourself, and keep going until you've genuinely run out of
worthwhile wins. Commit as you go on this branch (end messages with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`).
Never touch ~/Projects/reel itself: live videos import it right now.

## The constraint that matters
Directors (Opus agents) write per-video Python against reel's API, guided by GUIDE.md. Their experience must not
change much: the same functions with the same signatures and meanings (`reel.core`, `reel.fx`, `reel.act`, `reel.music`,
`reel.sfx`), the same `python -m reel.render out|stills|sheet|clip` commands, same project layout. New optional things
are fine (e.g. a `--draft` flag, a cache decorator directors *may* use, an env var). If something genuinely needs a
director-facing change, keep it small, and document it in GUIDE.md.

## Correctness bar
Output must be pixel-identical to today's, or differ by at most 1/255 on rare pixels (float rounding), at full
quality. Prove it on two real projects, rendered with today's reel and yours:
- ~/reel-port/roadster-hop (PYTHONHASHSEED=0: it seeds one blink with hash())
- ~/videoinator-runs/jobs/9057278c0a5a (copy it somewhere first; never write into it)
Compare ~30 frames spread across each video (render them through the project's shots.frame with each reel on
PYTHONPATH; the worktree's reel must come first on PYTHONPATH, since the venv has ~/Projects/reel installed editable).
`.venv/bin/python -m pytest tests` must pass, and add tests for anything new.

## What to measure
Wall time of a full `reel.render out` of both projects, before and after, on a quiet machine. Other renders run on
this box (videoinator jobs): before any benchmark, check `pgrep -af "reel.render|render.py"` and wait until it's
quiet. Profile first (cProfile / py-spy on a sample of frames from both projects) and go where the time actually is.

## Ideas, not a checklist
Profile before believing any of these: per-frame work that's actually constant (static plates, blurs of things that
don't move, text/caption images, glows) cached across frames; vectorising hot Python loops (stars, puffs, polygons);
cheaper glow/bloom (separable or downsampled blur at identical output, or precomputed kernels); float32 everywhere and
in-place ops; process count and chunk size for 8 cores / 23 GB (one render must stay under ~12 GB); the encoder pipe
(preset, threads, raw frame transfer); a `--draft` mode (half resolution, faster encode) for directors' check
renders, and teaching GUIDE.md to use it, which changes nothing for the final master. GPU is allowed only if output
stays identical and it's not fragile (there's a GTX 1060, 6 GB); a 2D GPU port of what directors write is out of scope.

## Done
Write RESULTS.md: before/after timings for both projects (full render, per frame, and the draft mode if you add it),
where the time went (profile tables), each change with its measured win, the identity check numbers, and anything
you tried that didn't pay off. Leave the branch committed and unmerged; Ben decides about merging.

## You're not alone
Another agent is doing this same goal in parallel in ~/Projects/reel-opt (don't read or touch it). Its renders will
show up in `pgrep` too, so don't wait for a perfectly quiet box forever: benchmark when it's quiet, and if it hasn't
been for 10 minutes, benchmark anyway and say so in RESULTS.md. The final head-to-head timings get re-run separately
on a quiet machine, so correctness and real wins matter more than your own stopwatch.
