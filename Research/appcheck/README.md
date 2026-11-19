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
   tone at -84.7 dB.

The app's exports against the render test's (24-bit files, so -120 dB is the last bit):

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

## What it does not cover
Dropping a file onto the window and the Open dialog (the sounds come in on the command line), playback (no sound
device here), the Export IR dialogs, and anything that only Windows or the Mac does differently. The JUCE objects
are not instrumented, so a fault inside the library's own code would show only if it reached the app's memory.
