# Making a video with reel

You're building a music video in code, frame by frame: Python draws every frame from generated art and vector shapes,
cut to a song. reel holds what every earlier video had to rebuild. Your project only holds the shots.

## The project

```
<project>/
  video.toml       song = "assets/song.mp3"   (plus size, fps, duration, shots, [fonts]: see reel/project.py)
  BRIEF.md         what to make; the critic judges against it
  assets/          the song, references, anything the brief supplied
  art/             generated images (art/prompts/NAME.txt -> art/NAME.png)
  mv/shots.py      frame(i) -> float32 RGB canvas in [0, 1]; optionally shot_of(i); split sections into mv/sec_*.py
  work/            audio.json, mix.wav, chunks, stills, sheets
  out/             masters and share encodes
```

Python is `~/Projects/reel/.venv/bin/python` (numpy, opencv, Pillow, librosa, reel). Run every command from the
project directory, or set `REEL_PROJECT`.

## The library

| | |
|---|---|
| `reel.core` (`import reel.core as C`) | `W H FPS DUR NF`, `canvas`, easing (`seg smooth ease_out back_out spring noise1`), `art`/`art_crop`/`get`/`put`, `M_place` + `draw` (affine, over/add/screen, tint), `draw_persp`, `cover` (background plates), `poly circle line rrect_pts ellipse_pts`, `glow vgrad`, `text text_w wrap` (fonts: sans = Inter variable, dm, mono, monom, spacemono), `shake apply_cam`, `bloom vignette finish`, `seed(name)` |
| `reel.fx` | `plume` (thruster/rocket), `puff` and `Puffs` (smoke, steam, dust clouds), `stars` |
| `reel.act` | character acting, pure in t: `bob` on the beat, `blink`, `look` (eye-darts), `talk` tilt, `hop` arcs, `land_squash`, `pose` (all of it at once) |
| `python -m reel.music [--bpm N]` | onsets + fitted beat grid -> work/audio.json. In code: `B = reel.music.beats()`, `B.snap(t)`, `B.beat(k)` |
| `reel.sfx` | synthesized hits (`blast thud clang pop whoosh click spool hiss …`) and `sfx.mix(cues, mutes)` -> work/mix.wav |
| `python -m reel.art gen NAME` / `queue FILE` | codex image generation (art/gen.log logs every outcome) |
| `python -m reel.render out/stills/sheet/clip` | chunked parallel render muxed with the audio (`out --draft`: the same frames, half size, fast encode); stills; contact sheets; clips |
| `python -m reel.qa measure/motion/sheets VIDEO` | the critic's own numbers: cut rhythm, cuts on onsets, near-static frames, local motion |
| `python -m reel.deliver MASTER OUT` | the share encode under 9.5 MB |

Read a module's docstring before using it; they're short. If you need something every video would need, add it to
reel with a test rather than to your project.

## What the critics keep saying

The same complaints come back on every video, so design against them from the start:

- **Cuts that ignore the music.** Snap every cut to a measured onset (`B.snap`); use the grid only where there are no
  onsets. A grid 78 ms late made cuts miss the music for four versions of one video.
- **One gag held for the whole video gets old.** Each section gets its own idea.
- **The muted first three seconds.** Feeds autoplay muted. Something must read and move from frame 0.
- **Captions on screen long enough to read twice**, legible on a phone, and inside their box: text that spills out of
  its window or off the frame edge is the most common bug critics find. Check every caption and label at full size.
- **A climax the music doesn't lift.** If the song has no drop where the payoff lands, build the hit on top of the
  music: stacked impacts (`blast` + `thud` + `clang`), a riser into it, a camera slam.
- **Never cut the music out.** No muted beats or dropouts, however dramatic they look on paper: viewers hear them as
  glitches. The song plays straight through; `sfx.mix` mutes are for trimming a song's own silence, nothing else.

## Working

- To keep a character on-model in generated art, give the image model its reference: `python -m reel.art gen NAME
  --refs ref.png`.
- Check your own cut before you call it done: `render sheet`, full-size `stills` at the key moments, `qa measure`,
  `qa motion`. The critic runs the same numbers. Check renders are `render out work/draftN.mp4 --draft` (qa reads
  them the same); render the master without `--draft`.
- Randomness comes from fixed seeds (`np.random.RandomState(k)`, `C.seed("name")`). Never `hash()`: it changes every
  process, so chunks rendered in different processes disagree.
- Every frame must render on its own from t alone. No state carried between frames; chunks render in parallel.
- Memory: the box has 23 GB. A full render is 4 processes (PROCS).
