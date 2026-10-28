# reel render optimization: results (Opus run, branch `optimize-opus`)

## Headline

| | before (6247899) | after (eb85bd6) | speedup |
|---|---|---|---|
| roadster-hop, full `reel.render out` (2685 frames) | 340.4 s | **195.2 s** (cold: 202.8 s) | **1.74x** (1.68x) |
| roadster-hop, wall per frame | 126.8 ms | 72.7 ms (75.5 ms) | |
| 9057278c0a5a, full `reel.render out` (2694 frames) | 752.7 s | **326.2 s** (warm: 329.3 s) | **2.31x** |
| 9057278c0a5a, wall per frame | 279.4 ms | 121.1 ms | |
| roadster-hop, `out --draft` (vs the old master) | 340.4 s | 165.5 s (722 CPU-s) | 2.06x |
| 9057278c0a5a, `out --draft` (vs the old master) | 752.7 s | 284.4 s (1194 CPU-s) | 2.65x |
| CPU for the whole render (user+sys), roadster / 9057278 | 1961 / 3124 CPU-s | 1140 / 1643 CPU-s | 1.7x / 1.9x less |
| peak memory (PSS of the render tree), roadster / 9057278 | 7.06 / 4.42 GB | 5.36 / 4.26 GB | |

"Warm" means work/chunks/times.json from an earlier render exists, so the slowest chunks go first. Directors
re-render all the time, so that's the usual case. "Cold" is a project's first render. For 9057278 the two are the same
within noise: its slow chunks sit mid-video, so order matters less. Both use the new default PROCS=4. The baseline used
PROCS=5, its default. At PROCS=5 the new code takes 198.4 s and 324.9 s. Every timing here is from a quiet box
(method below).

Identity: 32 frames spread over each video, rendered through the project's `shots.frame` with each reel first on
PYTHONPATH: **max difference 1/255, on 425 of 37.3 M pixel-channels (0.0011%) for roadster-hop and 378 (0.0010%) for
9057278c0a5a.** Every one of those comes from the gamma `pow` (below); with `np.power` the same frames are
bit-identical (0 differing pixels, measured at commit 2812f3e). `.venv/bin/python -m pytest tests`: 146 passed.

The master encode changed only its x264 thread count. PSNR of the decoded master against those 32 exact frames is
38.81 dB before and 38.81 dB after for roadster-hop (min 34.94 / 34.93 dB, 53.1 / 52.9 MB), and 36.58 / 36.58 dB for
9057278c0a5a (min 31.94 / 31.96 dB, 67.6 / 67.4 MB).

Nothing changed for directors: same functions, signatures and meanings, same commands, same layout. New and optional:
`render out --draft`. GUIDE.md says to use it for check renders, and that PROCS is now 4.

## How it was measured

- Box: i7-7700HQ, 4 cores / 8 threads, 23 GB, GTX 1060 (unused). Same venv for both reels (~/Projects/reel/.venv,
  numpy 2.2.6, opencv 5.0.0), PYTHONHASHSEED=0, `python -m reel.render out` from the project directory.
- "Before" is `git archive 6247899 reel` on PYTHONPATH (verified per run via `reel.core.__file__`). "After" is a
  `git archive` snapshot of the commit. Projects: copies of ~/reel-port/roadster-hop (minus its out*.mp4 and old chunks) and
  of ~/videoinator-runs/jobs/9057278c0a5a (video.toml, BRIEF.md, mv, art, assets, fonts, work/audio.json, work/mix.wav).
  Nothing was written into either original.
- Quiet machine: the parallel agent and I took turns under a shared `flock /tmp/reel-render-benchmark.lock`, and
  every timed step first waited for 30 s with no `-m reel.render`, `reel.deliver` or libx264 process anywhere else
  (videoinator jobs). One of my batches overlapped a videoinator master render that started mid-run. I discarded it
  and ran it again. Chrome was idling at ~0.5 core throughout.
- Wall time comes from `/usr/bin/time` around the whole command, mux included. Memory is the summed PSS of the render
  processes and their ffmpegs, sampled every second.
- Profiles: cProfile over 64 frames spread over each video, in one process, on the quiet box.

## Where the time went

**In one process** (cProfile, 64 frames, quiet box). The "reel" share counts the numpy arithmetic run inside reel's functions.

| roadster-hop | before: 275 ms/frame | after: 159 ms/frame |
|---|---|---|
| share: reel / C calls (cv2, PIL) / project mv/ | 70% / 19% / 11% | 38% / 45% / 17% |
| top | `_comp` 19.2%, `finish` 14.6%, `poly` 4.9%, PNG decode 4.3%, `art` 4.3%, `fx.puff` 4.3%, warpAffine 4.0%, `car.view` 3.6%, `np.clip` 3.0%, `canvas` 2.5%, `vignette` 2.4%, `bloom` 2.2% | warpAffine 7.4%, PNG decode 7.4%, `_comp` 5.6%, `car.view` 5.6%, `cv2.pow` 4.3%, `np.clip` 4.1%, cv2.LUT 3.8%, cv2.merge 3.6%, `traffic.keyed` 3.6%, `_mask_comp` 3.4%, `np.dstack` 3.2%, `vgrad` 2.6% |

| 9057278c0a5a | before: 376 ms/frame | after: 247 ms/frame |
|---|---|---|
| share: reel / C calls / project mv/ | 54% / 22% / 24% | 25% / 43% / 32% |
| top | `benbot.draw` 17.0%, `finish` 10.6%, `_comp` 9.8%, `poly` 8.1%, GaussianBlur 7.4%, `glow_ellipse` 6.7%, `np.mgrid` 3.0%, warpAffine 2.4%, `np.clip` 2.3%, PNG decode 2.2% | `benbot.draw` 23.8%, GaussianBlur 14.8%, warpAffine 4.8%, PNG decode 3.2%, `glow_ellipse` 3.1%, `_mask_comp` 2.9%, `_comp` 2.9%, resize 2.8%, `np.clip` 2.7%, `cv2.pow` 2.6% |

**In the real render** most of the cost never shows up in a one-process profile:

- *OpenCV's thread pool.* Every cv2 call (warpAffine, resize, GaussianBlur, LUT) woke 8 worker threads that then spun
  on `sched_yield`. In one process: 32 frames of 9057278 took 27.3 CPU-s with OpenCV's default 8 threads and 12.5
  CPU-s with 1, at the same wall time (11.6 s vs 11.2 s); `strace -c` counted 392,577 `sched_yield` calls in 8 frames.
  With 5 render processes, that burned CPU was taken from the other renders, and the profile above can't see it.
- *Page faults.* glibc unmapped each frame-sized temporary and faulted it back in on the next allocation: sys time was
  258 s (roadster) and 507 s (9057278) of the baseline renders.
- *x264's threads.* The default (1.5 per core, for each of 5 encoders) cost 146 ms of CPU per frame, 20% of the
  roadster render. 2 threads cost 91 ms.
- *Memory bandwidth.* With a process per core every frame-sized pass competes for the same DRAM. finish + quantize,
  as whole-frame passes, ran 2.6-2.7x slower per process with 4 processes running than with one, and 5 → 8 processes
  gained nothing.
- *Scheduling.* Chunks went out in order, so a slow chunk near the end held up the finish. The x264 lookahead drain
  (1.3 s per chunk on 2 threads) left its worker idle.

## Changes, each with its measured win

"cpu32" is one process rendering the 32 identity frames per project, timed with `/usr/bin/time`, startup included.

| commit | cpu32 roadster: CPU-s (wall) | cpu32 9057278: CPU-s (wall) | full render, roadster |
|---|---|---|---|
| baseline 6247899 | 19.4 (11.2 s) | 28.5 (13.5 s) | 340.4 s, P5 |
| 2812f3e exact compositing, slowest chunks first | 17.9 (8.5 s) | 27.7 (11.5 s) | |
| 4ed215b one OpenCV thread, malloc, x264 on 2 threads, cv2.pow | 9.1 (7.8 s) | 11.1 (9.8 s) | 224.0 s, P5, no chunk history |
| 885cba2 banded compositing, encoder drain overlapped | 8.0 (6.9 s) | 10.9 (9.7 s) | 198.4 s, P5, with history |
| eb85bd6 PROCS = 4 | 8.2 (7.2 s) | 11.3 (9.9 s) | 195.2 s, P4, with history (202.8 s without) |

1. **Exact compositing without numpy's 3-wide inner loops** (2812f3e). numpy runs `(h, w, 3) op (h, w, 1)` and
   `(h, w, 3) op (3,)` as millions of 3-element inner loops. `draw`/`_comp`, `poly`, `fx.puff`, `glow_ellipse`,
   `vignette`, `canvas`, `vgrad` and `art`'s premultiply now build matching contiguous operands (cv2.merge,
   coverage lookup tables via cv2.LUT, tiled colour rows), with the same values and dtypes, so the output is
   bit-identical, including float64 scalars (NEP 50 promotes them). Full frame, one process: poly blend
   23.4 → 10.7 ms, `col * a` 11.2 → 4.3 ms, `reg *= 1 - a` 6.7 → 5.8 ms, `reg += rgb` 7.9 → 5.7 ms, canvas
   6.3 → 1.0 ms, vignette 6.3 → 3.4 ms. `tests/test_exact.py` pins each one to the code it replaced.
2. **Slowest chunks first** (2812f3e). `run_chunks` orders chunks by `times.json`, written at the end of the previous
   render; chunks with no time yet go first. Replaying the baseline logs: roadster 335 → 321 s, 9057278's own
   draft log 590 → 560 s. Measured on the final code: roadster 202.8 s cold → 195.2 s warm; 9057278 unchanged.
3. **One OpenCV thread** (4ed215b): 27.3 → 12.5 CPU-s on the 9057278 sample at equal wall time (see above). Frames
   are identical.
4. **glibc keeps freed memory** (4ed215b): `mallopt` M_MMAP_THRESHOLD 32 MB, M_TRIM_THRESHOLD 1 GB, M_TOP_PAD 64 MB,
   set when reel.core is imported. On top of (3): 12.5 → 11.8 CPU-s, minor faults 115k → 49k. Full renders' sys
   time: 258 → 59-81 s (roadster), 507 → 35-58 s (9057278), together with (3).
5. **x264 on 2 threads** (4ed215b). Re-encoding one 60-frame chunk: default threads 146 ms CPU/frame, 4 threads
   129, 3 threads 95, 2 threads 91, 1 thread 88. But 1 thread encodes at half the speed (60 frames in 5.1 s vs 2.8 s),
   so the drain at the end of each chunk takes longer. Preset, CRF, GOP and B-frames are unchanged; master quality
   is under the headline.
6. **Gamma by `cv2.pow`** (4ed215b): 37 → 7.4 ms per frame. It is the only change that isn't bit-exact: within a few
   float32 ulps of `np.power`, which moves ~1 pixel-channel in 10^5 by 1/255. The float64 exp/log route took 81 ms
   and float32 exp/log took 16 ms.
7. **`render out --draft`** (8b40c4b): the same frames, encoded half size on `veryfast` (CRF 20, 1 thread), into
   `work/chunks-draft/` so a master's chunks are never mixed with a draft's. `reel.qa` reads a draft like the
   master: on 9057278, measure gave 56 cuts, 53/56 on onsets, median shot 1.50 s, near-static 3% vs 2%; motion gave
   local 73% vs 74%. The draft saves the encode only, since the frames cost the same. A real half-resolution render
   isn't possible: director code hardcodes 1080-px coordinates and slices canvases (`cv[:420]`) directly.
8. **Banded compositing** (885cba2). `_comp`, `_mask_comp`, `glow_ellipse`, `finish` and the uint8 quantize in
   `render.frame` work 32 rows at a time, so a band's temporaries stay in cache across every pass over it. Same
   arithmetic per pixel. finish + quantize per frame: 1 process 35.0 → 27.7 ms, **4 processes 91 → 53 ms each**
   (64-row bands: 63 ms). Full render: roadster 217.2 → 198.4 s.
9. **Encoder drain overlapped** (885cba2). x264 encodes the last ~40 frames of a chunk only after stdin closes (1.31 s
   on 2 threads vs 7.95 s for the chunk's 60 frames). `render_chunk` now returns as soon as the frames are in.
   The worker finalises that chunk (wait, rename) before handing back its next one, and pool workers drain the last
   one as they exit (`multiprocessing.util.Finalize`; the pool is closed and joined, not terminated). A chunk's mp4
   is deleted before its re-render, so a failed encode can't leave a stale chunk for `concat_mux`. Measured with (2):
   roadster P5 224.0 s (no history) → 217.2 s.
10. **PROCS 4** (eb85bd6). Banded, with history: roadster P4 194.8 s / P5 198.4 / P6 198.3; 9057278 P4 326.3 / P5
    324.9. P4 uses 1116 vs 1266 CPU-s and 5.45 vs 6.95 GB on roadster, which goes back to the other jobs on the box.

The end-to-end tests now put the checkout on the subprocesses' PYTHONPATH. Run from the test project's cwd, they had
been importing whatever reel the venv had installed (here: ~/Projects/reel), not the code under test.

## Tried, and didn't pay off

- **More processes.** 8 processes at commit 4ed215b: 226.5 s against 224.0 at 5, and 9.4 GB. With banding, 6 = 5
  ≥ 4. The render is memory-bound on 4 cores, so hyperthreads add nothing.
- **Decoded art in a shared, memory-mapped cache** (work/cache/art/*.npy, copy-on-write). Peak memory 6.58 → 5.52 GB at
  P5, but the first render pays the writes (242.9 s vs 224.0), and warps from a mapped array were 6% slower
  (10.7 vs 11.4 ms) because file mappings use 4 KB pages where numpy's arrays get huge pages. Memory isn't the limit
  at P4, so I dropped it. PNG decoding looks big in the 64-frame profiles (3-7%), but that's one pass over nearly
  every image, which is what each worker does once per render: ~1 s per worker, under 1% of a full render. Not
  worth a cache.
- `np.maximum` in place of `np.clip`: same speed (4.35 vs 4.38 ms). `vgrad` through `cv2.repeat` / nearest
  `cv2.resize`: 1.5-3 ms a call, ~1% of a frame, not worth the code.
- **Band-wise warps.** `warpAffine` into a band with a shifted matrix rounds its fixed-point coordinates differently,
  so it isn't provably exact. Left whole.
- **Pre-encoding the audio** while chunks render: the whole serial tail (concat + mux) is ~3 s.
- **GPU.** Not attempted. NVENC would change the master's encoder (not the same output, and consumer cards cap
  concurrent sessions). Moving the compositing itself to the GPU is the 2-D port the brief rules out.
- **numba** (installed with librosa) for fused compositing loops. Not needed: numpy + cv2 rewrites reached exactness
  without a JIT dependency or compile time per process.
- **Caching static per-frame work** (plates, glows) inside reel. reel receives arrays it can't safely key by identity
  (they're mutable), and these directors already `lru_cache` their static layers. Nothing to gain transparently.
- **Half-frame-rate drafts.** Would make `qa motion` read every other frame as static and shift cuts by a frame.

## What's left

Director code now dominates: in 9057278, `benbot.draw` (24%) and the cv2.GaussianBlur calls it and `kit` make (15%)
are the two biggest items. reel can't change those without changing director code. Of reel's own time, warpAffine
(sprite and camera warps, 5-7%) is the largest; the x264 encode is ~90 ms of CPU a frame, about a fifth of the
render, at the master's unchanged preset.
