# The listening page

The page of examples that goes out with each build: what an engine does to test sounds it has not seen, as audio,
with the scores next to it. It is published as a private page, not from this repo; the audio is not kept here.

- `build.py` puts the page together: the head and the first table (`intro.html`, `table.html`), six reverb examples
  picked by a fixed rule (`rendered.json`), then one section per build (`tone.html`, `echo.html`, `level.html`,
  `new044.html`, `cut045_head.html` + `cut045_panels.html`, `fader046_head.html` + `fader046_panels.html`,
  `new047_head.html` + `new047_panels.html`, `move050_head.html` + `move050_panels.html`, `four051_head.html` +
  `four051_panels.html`) and `outro.html`. It writes `index.html` next to an `audio/` folder of MP3s.
- Each section has a script that renders its examples with the app's own processing (`Tests/render.cpp` built as
  `drrender`) and writes the MP3s and the section's HTML: `tone_section.py`, `echo_section.py`, `level_section.py`,
  `new044_section.py`, `cut045_section.py`, `fader046_section.py`, `new047_section.py`, `move050_section.py`,
  `four051_section.py`. The heads with the numbers in them come from the result files of the scoreboard:
  `level_numbers.py`, `new044_head.py`, `cut045_head.py`, `fader046_head.py`, `new047_head.py`, `four051_head.py`
  (which also checks, with asserts, every count its sentences state against those files); `move050_head.py` reads
  the result files of the pitch reader's tests instead.
- The section of build 0.5.1 (the fixed filter, distortion and saturation) says next to each example what the real
  effect did, measured for the page: a filter's curve from the effect's answer to a click, a bend's squeeze from
  the sound before and after it (`true_curve`, `true_top` in `four051_section.py`).
- The section of build 0.5.0 (the MOVEMENT faders) uses the owner's own field recordings, picked by fixed rules from
  what the app reads in each of them. Its script is here; the panels it writes (`move050_panels.html`, with the
  pitch curves read from those recordings) and its numbers (`move050_info.json`) are not, like the audio.
- Examples are picked by rules written down in the scripts (a place in a ranking, the middle of a group), never by
  ear.
- The scripts expect the working folders of the research code (`Research/room`, the corpus, the scoreboard sets)
  where they were on the machine that made them; the paths are at the top of each file.
- `shot_cut045.js`, `shot_fader046.js`, `shot_new047.js`, `shot_move050.js` and `shot_four051.js` take pictures of a section with a
  headless browser, to check the layout in light, dark and phone width.
