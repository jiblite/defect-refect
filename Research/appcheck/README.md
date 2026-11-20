# The app itself, checked

The memory check (`Research/moving/memcheck.py`) and the scoreboard run the app's processing without its window
(`Tests/render.cpp`). These scripts check what that leaves out: the window's own code (`Source/App/Main.cpp`: the
loader, the worker and its threads, the faders, Swap, the pictures, Export) and whether the app gives the same sound
as the render test. First run for build 0.4.7. They are a record of how the check was run: paths point to the
working folders it ran in, and no audio is in the repo.

## How
- `mkbuild.py` writes `build.sh`: the app with all of its own code under the address and the undefined-behaviour
  sanitizer, linked against the JUCE objects of the local release build (4 minutes). It also writes
  `syntax_main.sh`: `Main.cpp` through clang and through mingw-w64 (the number sizes of Windows), syntax only.
- `scen.sh display tag scenario first.wav [second.wav]` starts that app on a display of its own (Xvfb) and drives it
  with the mouse (xdotool), waits for it to come to rest after every step, takes a picture, and asks the window to
  close the way a close button does (`wmdel.c`). The scenarios: `two` (every fader, Swap twice, one sound alone,
  Export), `quit`, `clear` and `swap` (asked to close, the first sound taken away, Swap pressed while the first
  analysis runs), `monkey` (random clicks, drags, double clicks and wheel turns on the controls), `exp` (fader rows
  pulled to one end, then Export to a file).
- `exp_all.py` and `exp_44.py` run `exp` on a list of cases and compare each exported file with the render test's
  output for the same sounds and fader positions (faders only at an end of their travel or in the middle, where
  their value is exact).
- `lagcheck.py export.wav reference.wav` says how many samples an export comes after the render test's output and
  how equal the two are once lined up.
- `valgrind.sh` (build 0.5.1) runs the two new engines' own test programs under valgrind, which sees a read of a
  value that was never set.

Without a window manager the atoms of the close protocol do not exist and JUCE only looks them up: `wmdel intern`
makes them before the app starts. The fader rows' places depend on which groups are shown and how many faders each
has (a crusher without a hold has no Sample Rate fader): look at the first picture before trusting a row.

## What it found (build 0.4.7)
Sanitizer: nothing, in any run (two sounds through every fader, Swap and Export; closing 10 and 24 s after the
start, clearing, swapping four times, all while an analysis ran; 150 random actions; every export run; every
picture of the picture test hook). The app closed by itself within 4 s every time, exit code 0.

Three faults, none of them one the sanitizer could see, all mended in 0.4.7:
1. With STUTTER Amount at far left the picture was an empty box under a caption that still spoke of repeats.
2. Export said "Saved" whenever a writer could be made, also when the file could not be opened or written.
3. The loader's change of sample rate. The exports at 48 kHz were equal to the render test's (below), the ones at
   44.1 kHz were not: the app's sound came 109 samples after the render test's (the delay of the resampler it
   used), and a 96 kHz file came down without a low-pass: a 30 kHz tone at -12 dB came out as an 18 kHz tone at
   -12.1 dB. The app now uses the render test's conversion (`Source/Core/Resample.h`): in place to the sample, the
   tone at -84.7 dB. The delay measured once more on two noise files (3 s, nothing above 16 kHz, faders in the
   middle), with the app as it was before the mend and after: 109 samples (2.27 ms) at 44.1 kHz and 50 samples
   (1.04 ms) at 96 kHz before, which is the 100 samples of the file that the old resampler states as its delay;
   0 samples after, and the 96 kHz export then equal to the render test's (-118.0 dB).

The app's exports against the render test's (24-bit files, so -120 dB is the last bit; 22 comparisons, 12 of them
with files at 44.1 or 96 kHz):

| sounds | faders | difference |
|---|---|---|
| crusher and stutter on one sound, alone | both Amounts far left | equal |
| the same onto a mono second sound | middle | equal |
| mixing crusher and stutter, alone | Amount, Bit Depth, Slice far left | -119.3 dB |
| the same onto a second sound | the same three far right | -121.1 dB |
| Haas delay, alone / onto a second sound | far left / middle | -118.4 / -120.4 dB |
| ring modulation, alone / onto a second sound | both far left / both far right | -117.2 / -124.8 dB |
| widener, alone / onto a second sound | far left / middle | -122.2 / -119.5 dB |
| 44.1 kHz: crusher and stutter, alone | middle; stutter far left; stutter far right | -120.6, -120.4, -120.9 dB |
| 44.1 kHz: the same onto a 44.1 kHz second sound | middle | equal |
| 96 kHz: stutter, alone | Amount far right; Slice far left | -119.9, -119.6 dB |
| 96 kHz: crusher onto a 44.1 kHz second sound | all three far right | -121.4 dB |
| 96 kHz test tones; a 44.1 kHz sound with nothing found | middle | -128.7, -119.1 dB |
| 96 kHz: noise up to 16 kHz | middle | -118.0 dB |
| 44.1 and 96 kHz: a mixing crusher that is found (nine tenths crushed, a hold of three samples), alone | Amount far left | -119.3, -118.9 dB |

## Build 0.5.0: the MOVEMENT group
New in the app's own code: the MOVEMENT group in the fader panel (always there, in the left column), its pictures,
and the pitch reader's thread (`PitchReader` in `Main.cpp`). That thread reads one sound after the other beside the
worker, and its answer comes in while the user may be doing something else, so the runs were aimed at it.

- `scen.sh` knows MOVEMENT's rows (`MROWS`, `MEND` in `exp`; its faders in `two` and `monkey`) and can wait for the
  reader: with `WAIT=reading` (`quit`, `clear`, `swap`) or `EARLY=reading` (`exp`) a scenario goes on once the Pitch
  fader's note says "reading", so the close, the clear, the Swap or the pull of the faders happens while the
  reader's thread runs. It sees the note by comparing its place on the screen with a picture of the word
  (`reading_tpl.png`, cut out of a `watch` run, which takes a picture every 2 s:
  `convert watch_w21.png -crop 60x16+326+414 +repage reading_tpl.png`).
- `exp_all.py` has the cases m1 .. m10, `exp_44.py` y10 .. y12. The two cases with the mixing crusher (x2, x4)
  listed two rows that are not there (that crusher has no Sample Rate fader, so STUTTER sits one row higher);
  their lists are right now, the figures are those of 0.4.7.

Sanitizer: nothing, in any run, and the app closed by itself within 4 s with exit code 0 every time:
asked to close 1 s and 9 s after "reading" appeared (two sounds; the first sound's pitch, then the second's) and
with one sound of zaps; the first sound taken away while its pitch was read; Swap pressed while reading and then
three times in a row; `two` (29 steps: every found fader, MOVEMENT's five with two sounds and three with one, the
group switched off and on, Swap twice, Reset, Tone Match, Export); `monkey` twice (150 actions on a crushed and
stuttered sound with a revving engine as the second sound, 120 on a vibrato with zaps as the second); the picture
hook (75 pictures, 15 of them MOVEMENT's); 32 export runs.

The app's exports against the render test's. The ten cases of 0.4.7 at 48 kHz and the nine at other rates, now
with MOVEMENT in the window and its faders in the middle, give the figures of 0.4.7 case by case (the 96 kHz test
tones are a new file: -121.3 dB). With MOVEMENT's faders moved:

| sounds | MOVEMENT's faders | difference |
|---|---|---|
| a vibrato, alone | Pitch far left | -123.2 dB |
| a revving engine, alone | all five far right | -119.6 dB |
| the engine as the second sound under the vibrato | all five far right | -119.6 dB |
| crusher and stutter on one sound, alone, both Amounts far left | Loudness and Tone far left | -120.3 dB |
| the same onto a mono second sound | Loudness and Spectral far right | equal |
| Haas delay, alone | Stereo and Spectral far left | -118.4 dB |
| zaps as the second sound under the engine | all five far left | -119.5 dB |
| the engine, alone | Pitch pulled far left while it said "reading" | -119.5 dB |
| the engine under the vibrato | all five pulled far right while Pitch said "reading" | -119.6 dB |
| a second sound of ten samples under the crushed sound | all five far right | equal |
| 44.1 kHz: crusher and stutter, alone | Loudness, Tone and Spectral far left | -120.1 dB |
| 96 kHz: stutter onto a 44.1 kHz second sound | all five far right | -116.5 dB |
| 44.1 kHz: crusher and stutter, alone | Pitch far right | -120.6 dB |

Two of these say less than they seem to. The ten samples come out as silence from both (the first sound is silent
at that time, and Loudness at far right brings its level). On the 44.1 kHz sound the reader follows nothing (the
crusher's folded lines move against the sound's own), so the fader says "no pitch" and the export is the one with
the fader in the middle, to the sample; on the same sound made at 48 kHz with another crusher it follows 47 %.
The two cases pulled while "reading" come out like the render test's files for those fader positions, and the
second is, to the sample, the file of the same pull made after the reading: the result waits for the answer. The
release build of the app, run on two of the cases (the engine under the vibrato, and the engine alone pulled while
"reading"), gave the same files as the sanitizer build, to the sample.

The memory check ran its two new stages for this build (`Research/moving/memcheck.py move` and `movefx`: 80 runs,
522 renders, every one clean). Its older stages keep MOVEMENT's faders in the middle, where the render test measures
no movement and runs the code of 0.4.7, and gives the same files bit for bit (`Research/movement/same.py`, 38 cases);
they were not run again.

## Build 0.5.1: FILTER, DISTORTION and SATURATION
New in the app's own code: two more analysis jobs beside the worker (`filterJob`, `bendJob` in `Main.cpp`; a file at
another rate gets its bend read off its own samples), the two groups in the fader panel (FILTER with two or three
faders, DISTORTION or SATURATION with one), their pictures, and no EXCITER group.

- `scen.sh` has the scenario `four` for a sound that shows SATURATION and a FILTER group, with a second sound:
  every fader of both groups to its ends, MOVEMENT's Loudness and Tone, Result and Play, Swap twice, Reset, Tone
  Match, the second sound away and the faders on the sound itself, each group switched off and on, Sureness to
  both ends, Export. `monkey` takes the rows of the sound's own faders (`FROWS`, and `HY2` for the second group's
  title).
- `exp_051.py` has the cases f1 .. f5, b1 .. b5, c1 .. c3 and y21 .. y23; the made sounds come from
  `Research/filter/mkcheck.py`.

Sanitizer: nothing, in any run, and the app ended by itself with exit code 0 every time: asked to close 3, 9 and
25 s after the start (two sounds, and a long distorted one), the first sound taken away and Swap pressed 6 s
after the start, all while the first analysis ran; `four` (23 steps); `monkey` twice (150 actions on the sound
with both groups and a revving engine as the second sound, 120 on a distorted file at 44.1 kHz with a filtered
mono file as the second); the picture hook (93 pictures, 21 of them the new groups'); 16 export runs. The app was
gone within 4 to 8 s of being asked, with one exception: asked 3 s after the start it took 92 s, at a moment when
the machine's four cores were shared with seven other jobs at full load and a second copy of the app that had just
started. The same run again with fewer of them took 8 s. The app waits for its analysis threads, and each of them
stops only where it looks whether it should.

One fault in the scenario itself, none in the app: two pulls of `four` started 18 points beside a fader that
stood at its far left and moved nothing, which showed in the pictures (both Amounts still at 0 %). They start on
the fader now, and the run was made again: both at 200 %.

The app's exports against the render test's. Every difference is one unit of the 24-bit file at the most (143
to 147 dB under full scale); the figure is that difference against the level of the file, so a quiet file reads
higher:

| sounds | faders | difference |
|---|---|---|
| a low-pass with a peak, alone | Amount far left | -121.7 dB |
| the same | Cutoff and Resonance far right | -121.7 dB |
| the same onto a second sound | middle | -117.0 dB |
| a high-pass onto a second sound | Amount and Cutoff far right (a file at -48 dB) | -95.3 dB |
| a high-pass that is read as a band-pass, alone | Amount and Cutoff far left | -128.3 dB |
| a distortion, alone | Amount far left | -118.7 dB |
| another distortion, alone | Amount far right | -138.9 dB |
| a distortion onto a second sound | middle | -127.3 dB |
| a saturation, alone | Amount far left | -121.6 dB |
| the same onto a second sound | Amount far right | -130.1 dB |
| a low-pass and a saturation on one sound, alone | both Amounts far left | -125.3 dB |
| the same onto a mono second sound | Cutoff far right; MOVEMENT's Loudness and Tone far right | -124.2 dB |
| the same, alone | the filter's Amount and Cutoff far right; MOVEMENT's Pitch far left | -129.9 dB |
| 44.1 kHz: a low-pass, alone | Amount far left | -121.9 dB |
| 44.1 kHz: a distortion, alone | Amount far left | -129.9 dB |
| 44.1 kHz: the same onto a 44.1 kHz second sound | Amount far right | -131.0 dB |

The memory check ran every one of its stages again for this build, because the two new engines measure every sound:
the new stage (`Research/moving/memcheck.py four`: 34 runs, 1,632 renders) and the nine older ones, 271 runs and 4,899
renders in all, every one clean.

The sanitizer does not see a value that is read before it was ever set. For that the two new engines' own test
programs (`Tests/filter.cpp`, `Tests/bend.cpp`) ran under valgrind (`valgrind.sh`): each of the 15 sounds made for
the memory check's new stage alone, with the faders at both ends and in between, and four of them onto four second
sounds (a mono one, one at 44.1 kHz, two short ones) with the faders in the middle and far right, and each engine
once made to work on a sound in which it finds nothing: 156 runs, no report. The whole render test ran under it
once as well (a sound with a low-pass and a saturation, onto a mono second sound with both groups' faders far
right, and alone with them far left; 15 minutes): no report, and all memory given back at the end.

## Build 0.5.2: the declipper
New in the app's own code: `bendJob` in `Main.cpp` hands the found bend to the declipper (`bend::addFlat`: the tops
that were cut or pressed flat are rebuilt, in a file at another rate from the file's own samples); the distortion
picture draws a rebuilt half-wave's own curve and says in words what was done; the Amount fader's tip.

- `scen.sh` has the scenario `flat` for a sound whose tops are rebuilt (one group, one fader), with a second sound:
  the fader to both ends and part of the way back, MOVEMENT's Loudness and Tone, Result and Play, Swap twice, Reset,
  Tone Match, the second sound away and the fader on the sound itself (far left: the rebuilt tops), the group
  switched off and on, Sureness to both ends, Export. 20 steps.
- `exp_052.py` has the cases d1 .. d14, c1, c2 and y31 .. y35; the made sounds come from `Research/declip/mkcheck.py`.

Sanitizer: nothing, in any run, and the app ended by itself with exit code 0 every time: asked to close 3, 9, 25 and
40 s after the start (a cut and a soft flat top with a second sound, a minute of soft flat tops, a long cut: gone
within 4 to 8 s), the first sound taken away and Swap pressed 6 s after the start, all while the tops were being
rebuilt; `flat` three times (a cut with a mono second sound, a soft flat top with a revving engine, a dense sound cut
on one side only with a mono second sound); `monkey` twice (150 actions on a cut with the engine as the second sound,
120 on a soft flat top at 44.1 kHz with a filtered mono file); the picture hook (93 pictures, the same as build
0.5.1's: its made-up finds have no flat tops); 21 export runs.

One flaw, in the picture and not in the sound: a plain cut on a dense sound is not rebuilt (the declipper tries its
rebuilding one step down first and refuses where that fails), and left of the middle the engine leaves such a
half-wave as it is. The picture drew the straight line there all the same, as if the tops were back, next to words
that said "left of the middle nothing changes". `bendPlot` now draws a half-wave that stays as found; where only
one side is cut, the other side is walked back the meter's way and drawn so. Seen in the window on a noise with three
tenths of its values cut and on one cut on one side only (the new sound `cutNoiseUp`).

Two of the made sounds (`softMono`, `soft96`) are read with 41 % and so are not shown at the default Sureness: their
exports prove only that a group that is not shown does nothing. `softMonoP` and `soft96P` are the same kind of sound
read with 100 %, and are rebuilt.

The app's exports against the render test's:

| sounds | faders | difference |
|---|---|---|
| a plain cut, alone | Amount far left (the tops rebuilt) | -127.6 dB |
| a soft flat top, alone | Amount far left | -129.1 dB |
| a plain cut onto a mono second sound | middle | -124.9 dB |
| a soft flat top onto a mono second sound | Amount far right | -134.4 dB |
| a plain cut, alone | Amount far right (cut as deep again) | -138.0 dB |
| a dense sound with a cut that is not rebuilt, alone | Amount far left (the export is the sound itself) | -141.1 dB |
| a dense sound cut on one side only, not rebuilt, alone (a file at -26 dB) | Amount far left | -116.2 dB |
| a 16 bit file cut into full scale, alone | Amount far left | -128.4 dB |
| a mono file with a cut onto a 44.1 kHz second sound | Amount far right | -128.0 dB |
| another level of cut on each channel, alone | Amount far left | -123.0 dB |
| a minute of soft flat tops, alone | Amount far left | -127.3 dB |
| a plain cut, alone | Amount far left; MOVEMENT's Loudness and Tone far left | -127.3 dB |
| a mono soft flat top, alone | Amount far left; MOVEMENT's Pitch far right | -129.2 dB |
| a mono soft flat top that is not shown, alone | MOVEMENT's Pitch far right | -142.2 dB |
| a low-pass and a saturation on one sound, alone | both Amounts far left | -125.3 dB |
| the same onto a mono second sound | Cutoff far right; MOVEMENT's Loudness and Tone far right | -124.2 dB |
| 44.1 kHz: a plain cut, alone | Amount far left | -126.8 dB |
| 44.1 kHz, 16 bit: a plain cut onto a 44.1 kHz second sound | Amount far right | -132.9 dB |
| 22.05 kHz: a plain cut onto a mono second sound | middle | -124.1 dB |
| 96 kHz: soft flat tops, alone | Amount far left | -128.8 dB |
| 96 kHz: soft flat tops that are not shown, alone | none | -130.6 dB |

The memory check ran every one of its stages again for this build, because the declipper looks at every sound: the
new stage (`Research/moving/memcheck.py flat`: 56 runs, 1,626 renders, on the 43 sounds of
`Research/declip/mkcheck.py`) and the ten older ones, 327 runs and 6,525 renders in all, every one clean. (The
machine restarted during the last two stages; the long test sounds were made again and the two stages ran from
their start.)

The sanitizer does not see a value that is read before it was ever set. For that the declipper's own test program
(`Tests/bend.cpp`) ran under valgrind on each of the 43 sounds with the fader far left (`Research/declip/check.sh`):
no report. The same script ran it under the address sanitizer (220 runs: every sound far left, some at other fader
places and onto second sounds, some with the first pass forced onto parts of a short sound) and under the thread
sanitizer (16 runs: the sounds whose rebuilding runs on several threads): no report. The whole render test ran under
valgrind as well (`Research/declip/vg_render.sh`: a cut onto a mono second sound with the fader far right and alone
with it far left and half way, a second of soft flat tops alone and onto a second sound of ten samples): no report,
and all memory given back at the end.

## What it does not cover
Dropping a file onto the window and the Open dialog (the sounds come in on the command line), playback (no sound
device here), the Export IR dialogs, and anything that only Windows or the Mac does differently. The JUCE objects
are not instrumented, so a fault inside the library's own code would show only if it reached the app's memory.

The last two rows of the table closed a gap: the sounds with a mixing crusher that the memory check had (seven
tenths crushed, a hold that is not a whole number) are ones the engine does not find, or finds without the hold, so
the undoing of such a crusher in a file at another rate (the sound before the crusher is brought to 48 kHz like the
sound itself) had run nowhere. On the two new sounds the crusher is found with its hold, far left gives the dry sound
back to -55 dB (44.1 kHz) and -69 dB (96 kHz) where the crushed sound is at -16 and -23 dB, the app's export is
equal to the render test's, and `memcheck.py soft` runs them and a long one under the sanitizer.
