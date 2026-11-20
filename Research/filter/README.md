# The fixed-filter engine: FILTER (build 0.5.1)

A fixed filter is a low-pass, a high-pass or a band-pass that stands still while the sound plays, with or without
a resonance at its corner. (A filter that moves is the moving-tone engine's FILTER SWEEP.) The engine reads the
filter's curve off the sound, and the group works on that curve: LOW-PASS FILTER, HIGH-PASS FILTER or BAND-PASS
FILTER, with Amount, Cutoff and, where the curve has a peak, Resonance.

Prototype here, the app's code in `Source/Core/Filter.h` and `Source/Core/Reader.h`, compared by
`Tools/filter_parity.py` through `Tests/filter.cpp`: the curves come out within 0.0001 dB of the prototype's and
the sounds within -119 dB (ten pairs).

## The difficulty

A filter leaves no mark that a sound could not have by itself: a sound without highs may be a filtered sound or a
dull one. What can be read is that sounds are not built the way filters are. A filter's skirt falls by the same
number of dB in every frame, loud or quiet, and it falls on a straight line per octave down to the noise; a
sound's own spectrum moves from frame to frame and has no such edge. So the reader does not look at the average
spectrum alone.

## What is read (`ffeat.py`)

The sound's energy in 104 bands a twelfth of an octave wide, from 50 Hz to 19 kHz (frames of 8192 samples every
2048, both sides together). Per band six curves over the frames that count: the average, the quietest frame, the
10th, 50th and 90th in a hundred, and the loudest, all against the sound's own level. A long sound is looked at in
at most 280 frames.

## The reader (`fcurve.py`, `fgen.py`, `fgen2.py`, `ftrain.py`)

A small net of one-dimensional convolutions along the bands (48,000 weights). In: the six curves and each band's
place. Out: the filter's curve band by band, in dB with 0 in the pass band, and whether a filter is there.

It is taught on made examples: a training clip's band energies (sometimes two clips mixed, a stretch of it, with a
mild colour of its own) through a curve of the family or through none, sometimes with a noise floor added
afterwards. The family (`fcurve.py`): low-passes and high-passes of several builds (one two-pole section with any
resonance, two of them in a row, flat ones of one to six poles with or without a peak at the corner, a four-pole
ladder with feedback) and band-passes (one or two two-pole sections, or a high-pass and a low-pass together), half
of them bent the way a filter made at 48 kHz bends its curve towards half the rate. The second make of examples
(`fgen2.py`) adds what is not a fixed filter and was taken for one by the first reader: filters that move (sweeps,
phasers, flangers), a level that moves, a room's tail, a source that was cut off at a low sample rate, and sources
with a noise floor of their own, which a filter then shapes.

Two such nets are in the app, taught from different starts: one on 390,000 examples for 24 rounds, one on 780,000
(those and as many again) for 12. On its own held-back examples each reads the curve of a filtered sound 3.0 dB
off on average, and 1.2 dB into a sound without a filter.

## From the reader's answer to the engine (`fengine.py`, `filt.py`)

- **Two readers, and the least of what they read.** Band by band the engine takes the reading that is nearer to
  zero, and nothing where the two do not agree on the sign: what only one of them reads is not taken. A cut that
  is read too deep does harm when it is undone (a lift where nothing was cut); one that is read too shallow
  leaves some of the filter in. On the outside cases (table under Results) their mean reads the curves best and
  the least of them does the least harm.
- **Two looks.** The read curve is taken off the band energies and the rest is read again. The reader was taught
  curves held at -40 dB, and a real skirt runs on under that; the second look adds to the first only where the
  first read a deep cut (nothing under 24 dB of cut, all of it from 36 dB on).
- **Where nothing is left, the skirt runs on.** Where a filter has cut everything (the band's level is within 6 dB
  of the lowest the readers are given, 110 dB under the sound's own) there is nothing to read, and the answer
  drifts back towards no cut at all: a low-pass at 1 kHz came out as a dip at 3 kHz with the top octaves open. In
  such a band the curve keeps the lowest point it reached on the way out from the pass band, once that is 12 dB
  down. The same rule in every band, a curve that may never rise again, was tried and does harm: a dip read into
  a part where the first sound is only quiet becomes a cut for a second sound that has everything there.
- **What counts as found.** The readers' verdict (the mean of their two), a cut of at least 6 dB somewhere the
  clean sound could be heard, and a curve of one of the three kinds: 6 dB down or more at the lower end of the
  range (a high-pass), at the upper end (a low-pass) or at both (a band-pass). The group is shown where the
  Sureness slider stands at first from a verdict of 2.5 (a logit), and kept for a lower slider from 0. The curve's
  corners are where it passes -3 dB; a peak of 1 dB or more above the pass band is a resonance.
- **The filter itself** is the curve as a minimum-phase filter, so taking it off also undoes the filter's phase.
  The curve is 104 numbers, not one of a few textbook shapes: a ladder filter, a state-variable one and a pair of
  two-pole sections are each read as what they did.
- **Taking it out lifts what was cut.** The lift ends at 40 dB, and at 12 dB in a band that holds still while the
  rest of the sound moves: such a band is noise that came after the filter, and lifting it would bring up hiss.
- **Cutoff** moves the curve along the frequency axis (two octaves each way), **Resonance** scales its peak.
- **A filter that moves is the moving-tone engine's** (`both.py`), and the two groups never show side by side: both would
  work on the same filter. A sweep that is sure enough to show where the Sureness slider stands at first wins, and
  so does a fainter one that is at least as sure as the fixed filter. Otherwise the fixed filter wins and the faint
  sweep is not shown at all. On 4 of the 32 outside cases the moving-tone engine has such a faint suspicion of a
  sweep (0.07 to 0.54 sure, where the filter engine is 0.90 to 0.98 sure); on the scoreboard's eight sweeps it is
  0.68 to 1.00 sure, and the filter engine reads a fixed filter into three of them (0.59 to 0.90 sure). Until this
  rule any sweep won, and on those four sounds neither group was shown: through the whole app 18 of the 32 cases
  at the default, with the rule 22, as the filter engine alone.

## Results

Outside cases, blind (`fgrid.py`): 32 cases, 64 sounds, from four outside effects (8 cases of the scoreboard and
24 fresh ones; none of their code is in the training family). The read curve is 6.3 dB off on average where the
clean sound can be heard (median 5.4). Scores: 100 = the truth, 0 = nothing gained.

| | at the default | with the slider down |
|---|---|---|
| sounds shown (64) | 51 | 60 |
| cases shown (32) | 22 | 28 |
| taken out, mean / median over the shown | 50 / 58 | 46 / 52 |
| taken out: cases worse than nothing | 0 | 0 |
| onto a second sound, mean / median | 63 / 78 | 60 / 76 |
| onto a second sound: cases worse than nothing | 1 | 1 |
| clean corpus (198): a filter shown | 0 | 3 |
| sounds through the other effects (624): a filter shown | 16 | 59 |
| ... of them truly cut by 6 dB or more at an end of the range | 7 | 13 |

The one case that goes onto a second sound worse than nothing (-73) is a high-pass at 190 Hz that is read at
290 Hz, 0.6 octave too high. The true filter changes that second sound by a part 19 dB under it, so a small
fault counts for much.

The readers alone and together, at the default (a verdict of 2.5):

| | cases shown | taken out, mean | worse than nothing | onto, mean | worse | filters shown on the 624 other sounds | curve off, dB |
|---|---|---|---|---|---|---|---|
| the first reader alone | 24 | 48 | 2 | 55 | 2 | 23 | 6.2 |
| the second reader alone | 21 | 42 | 2 | 57 | 1 | 18 | 6.2 |
| both, their mean | 22 | 48 | 2 | 60 | 1 | 17 | 5.8 |
| both, the least | 22 | 50 | 0 | 63 | 1 | 15 | 6.5 |
| both, the least, the skirt running on where nothing is left (the app) | 22 | 50 | 0 | 63 | 1 | 16 | 6.3 |

With the slider down the difference is larger: one reader alone or the mean of both makes 3 of the cases it
then shows worse when the filter is taken out, the least of both none.

BOARD_PLACEHOLDER

## What it does not do

- **A filter on one side only.** Both sides are read together.
- **A shelf or a bell of an equaliser, a band-stop or a notch.** A curve that is not 6 dB down at an end of the
  range is not shown. A change of colour is left to the MOVEMENT group's Spectral fader and to TONE MATCH.
- **A sound that is dull or thin by itself** is not told from a filtered one beyond a point, and that goes both
  ways (`fothers.py`). A filter that is not there: where the slider stands at first on none of the 198 clean
  clips and on 9 of the 624 scoreboard sounds of the other effects (5 of the 240 with a reverb, 2 of the 128
  with an equaliser, a chorus, a delay); with the slider down on 3 and on 46. Most are a high-pass read into a
  sound that has little low end, some a low-pass read into a dull one. (The other seven shown at the default are
  right: three filter sweeps, where the moving-tone engine's group wins when it finds them too, and four speaker
  cabinets, which do cut the lows and the highs.) And a filter that is there is missed, or its corner is read
  too high or too low, where the sound has little near the corner: 10 of the 32 cases are not shown at the
  default, 4 not at all.
- **What the filter left nothing of.** Beyond the point where the first sound is gone the curve is not read but
  carried on, and a second sound gets that guess.
- **A sound under a second** (fewer than 20 frames that count) is not read.

## Files

`ffeat.py` what is read, `fcurve.py` the family of curves, `fgen.py` and `fgen2.py` the training examples,
`ftrain.py` the net, `reader.py` the nets read from the app's model file or the trainer's files, `filt.py` a curve
as a filter, `fengine.py` the engine, `fgrid.py` the outside cases, the limits and the readers alone and
together, `fothers.py` the filters shown on other effects' sounds, `both.py` the filter engine and the moving-tone
engine on the same sounds (through the app's processing), `feval.py`, `ftest.py`, `fscore.py` and
`fclean.py` earlier evaluations (with `common.py`), `mkcheck.py` the made sounds of the memory check. Model: `Tools/export_reader_model.py` writes `Resources/filter.bin`. The scripts expect the
working folders where they were on the machine that made them.
