# The listening page

The page of examples that goes out with each build: what an engine does to test sounds it has not seen, as audio,
with the scores next to it. It is published as a private page, not from this repo; the audio is not kept here.

- `build.py` puts the page together: the head and the first table (`intro.html`, `table.html`), six reverb examples
  picked by a fixed rule (`rendered.json`), then one section per build (`tone.html`, `echo.html`, `level.html`,
  `new044.html`, `cut045_head.html` + `cut045_panels.html`) and `outro.html`. It writes `index.html` next to an
  `audio/` folder of MP3s.
- Each section has a script that renders its examples with the app's own processing (`Tests/render.cpp` built as
  `drrender`) and writes the MP3s and the section's HTML: `tone_section.py`, `echo_section.py`, `level_section.py`,
  `new044_section.py`, `cut045_section.py`. The heads with the numbers in them come from the result files of the
  scoreboard: `level_numbers.py`, `new044_head.py`, `cut045_head.py`.
- Examples are picked by rules written down in the scripts (a place in a ranking, the middle of a group), never by
  ear.
- The scripts expect the working folders of the research code (`Research/room`, the corpus, the scoreboard sets)
  where they were on the machine that made them; the paths are at the top of each file.
- `shot_cut045.js` takes pictures of a section with a headless browser, to check the layout in light, dark and
  phone width.
