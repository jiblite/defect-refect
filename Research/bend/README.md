# The bend engine: DISTORTION and SATURATION (build 0.5.1)

A distortion and a saturation are the same thing at two strengths: a fixed curve that every value of the sound is
pushed through. Small values pass in a line, large ones are squeezed. The engine reads that curve off the sound,
and the group works on what was read. It is called DISTORTION when the loudest values were squeezed by 10 dB or
more, SATURATION under that.

Prototype here, the app's code in `Source/Core/Bend.h`, compared by `Tools/bend_parity.py` through
`Tests/bend.cpp`: what the meter sees and its answer come out equal, the sounds within -299 dB (six pairs).

## What is read (`bfeat.py`)

One half-wave at a time, the values above zero and those below: a tube or a tape does not bend the two alike.
The sound is cut into frames of 2048 samples, and the frames are sorted into ten classes by how high their peak
stands against the half-wave's own. Per class: where the frame's values lie under its largest (eleven places in
their order, from three quarters of the way up), how steep the wave is at six heights against the frame's own
steepness, and how many frames the class holds; and how much of the half-wave sits within 0.1, 0.5, 1 and 2 dB of
its peak. 184 numbers per half-wave.

Why these: a curve leaves the quiet frames as they were and presses the loud ones together. So the same sound
shows its own shape in the quiet frames and the bent shape in the loud ones, and the difference between the
classes is the curve. A tone that was bent is also flat where it is high, which the steepness shows; a level that
was held down by a moving gain (a limiter) piles its values up near the top too, but keeps the wave's shape.

## The answer (`bfeat.answer`)

Per half-wave, by how many dB the curve squeezed each of eleven levels of the sound as it was, from 30 dB under
its peak up to the peak, against the curve's plain gain for small values. For a training example the box and the
sound that went in are known, and the answer is measured: one rising curve is fitted through all pairs of values
in and out, both halves at once so that it cannot fold at zero.

The curve is therefore not one of a few textbook shapes. It is the eleven numbers per half-wave, joined by a
smooth rising line (`bengine.smooth`).

## The meter (`bgen.py`, `bgen_tone.py`, `btrain.py`)

Four small nets (both half-waves' numbers in, the squeeze at eleven levels and one verdict out; each half-wave is
read with the other one as its background, and the two verdicts are added). They are taught alike from different
starts, and the meter's answer is the mean of theirs: on a real bend the four agree, a chance find is one net's.

90,000 training examples, 7,500 more held back:

- 78,000 from the training clips (`bgen.py`): a stretch of a clip, sometimes two mixed, sometimes dull or thin by
  itself. A third stay as they are. Nearly a fifth are things that are not a bend: a limiter, the opposite of a
  curve (values spread apart), a sound cut down to few steps. The rest go through a box of the family (`bchain.py`):
  a drive, one of seven curve shapes, even or uneven for the two halves, with or without plain sound mixed in,
  and often what such boxes have behind the curve, an offset blocker and a mild low-pass. Where the box changed
  the sound too little to be heard, the example counts as "no bend there".
- 12,000 with made steady tones (`bgen_tone.py`): sines, harmonic sounds of any strengths and phases, chords,
  pairs that beat, tones that waver, struck sounds that die away, each alone or mixed with a clip. A plain tone's
  values pile up near its top much as a bent sound's do (a sine spends a third of its time within 1 dB of its
  peak), the training clips hold few sounds of that kind, and the first meter took 12 of 12 made sines for
  saturated. Tones by themselves and in a mix are in the set as they are (no bend); the mix, and a tone played as
  notes of many levels, also through a box. A steady tone alone through a box is left out on purpose: what comes
  out is another steady tone, and nothing in it tells that a box was there.

More examples of the same kind do not help: a meter taught on 36,000 and one on 78,000 read the held-back
examples equally well. Taught for long a net learns its examples by heart, and then flags different clean sounds
each time; so each net sees every input a little off in training (`JIT`), and four are read together.

## Taking it out, pushing it further, putting it on (`bengine.py`)

- **Onto a second sound:** that sound is scaled so that its loudest values reach as far into the curve as the
  first sound's did, pushed through, and brought back to its own peak. Amount: how far into the curve.
- **Right of the middle on the sound itself:** the bend is walked back and the sound pushed through the same curve
  again, further in.
- **Left of the middle:** the way back. Values are spread apart again by the curve read the other way round.

The way back has two limits, and both are in the engine on purpose.

*Flat tops stay flat.* Where the curve pressed many levels into one, nothing in the sound says which was which.
The way back spreads values at most six times as far apart as they lie (`SPREAD`); beyond that it would only
blow up whatever ripples sit on the flat tops.

*A reading that is too high does harm.* The meter is off by a dB or two. Read too low, some of the bend stays
in. Read too high, values are spread apart that were never squeezed: a new fault, and a worse one than the bend.
So the way back takes three quarters of the read squeeze, less 1 dB (`careful`).

Compared on the 48 outside cases, over the 33 the app shows (`bback.py`, `bback2.py`): the plain way back scores
15 on average and makes 9 cases worse than leaving them alone, three of them badly (under -20); the true answer,
were it known, scores 23 and makes 11 worse, four badly; three quarters less 1 dB scores 22.5 and makes 7 worse,
one badly. Seven tenths of the reading scores 21, half of it 19.5. A lower cap on the spreading does worse than
K 6 (21 with K 3, 18 with K 2).

## How sure (`bengine.sure`)

The meter gives two things, a verdict (is a bend there to be heard) and the squeeze. The verdict alone is cautious:
it was taught to say no where a box changed the sound too little to be heard, and it says no to many a soft bend
that can be heard. The squeeze alone takes any flat-topped sound. The sureness is the weaker of the two: it is at
the Sureness slider's first position (65 %) when the verdict's logit is -2 and the squeeze at the peak 5 dB, and a
bend is kept at all (shown when the slider is turned down) from a logit of -6 and 3 dB. Before that rule the app
showed 23 of the 48 cases and none of the scoreboard's eight saturations; with it 33. The ten more go onto a
second sound at 24 to 85, and are taken out at 8 to 75 but for one at -1 (`bgrid.py`, `bgrid2.py`, `brule.py`).

## Results

Held-back examples (7,500), the four nets together (`bens.py`): the squeeze of a heard bend is off by 1.7 dB over
the levels and 2.7 dB at the peak (each net alone: 1.8 and 2.85). Shown where the slider stands at first: 97.5 %
of the heard bends of 10 dB and more at the peak, 76 % of those of 5 to 10 dB, 14 % of the softer ones; 0.2 % of
the clean clips, 3 % of the limited ones, 2 % of the made tones, 4 % of tone and clip mixed.

Made steady tones, twelve of each (`look11.py`), shown at the default and, in brackets, with the slider down:

| | shown |
|---|---|
| a sine, a wavering sine, a triangle, a chord of three sines, noise alone | 0 (0) each |
| harmonic sounds (two kinds) | 0 (1) each |
| a sawtooth | 0 (3) |
| a sine in noise, 20 dB under it | 1 (2) |
| two close sines that beat | 2 (6) |
| a pulse wave | 8 (9) |
| a square wave | 12 (12), as DISTORTION of 21 dB |

A square wave is what a sine pushed hard through a curve looks like, and nothing in the sound tells them apart.

Outside cases, blind (`beval.py`): 48 cases, 96 sounds, from eight outside effects, four that distort and four
that saturate (8 distortion and 8 saturation cases of the scoreboard, 16 and 16 fresh ones; none of their code is
in the training family).

| | shown at the default | with the slider down |
|---|---|---|
| distorted sounds (48) | 45 | 47 |
| saturated sounds, 6 dB and more at the peak (25) | 21 | 25 |
| saturated sounds, under 6 dB at the peak (23) | 3 | 8 |
| clean corpus (198) | 4 | 12 |
| sounds through the other effects (608) | 12 | 33 |

The read squeeze is off by 1.8 dB rms on average (1.2 in the median), and DISTORTION or SATURATION is the right
name on 85 of the 96 sounds. The four clean sounds shown at the default have flat tops themselves: three are the
same steady sound, a quarter of whose values lie in one half-dB band under its peak, the fourth has a hundredth of
its values at full scale.

Scores over the cases the app shows at the default (33 of 48; 100 = the truth, 0 = nothing gained):

| shown as | cases | onto a second sound, mean / median | worse than nothing | taken out, mean / median | worse than nothing |
|---|---|---|---|---|---|
| DISTORTION | 24 | 34 / 53 | 5 | 20 / 17 | 6 |
| SATURATION | 9 | 54 / 51 | 0 | 29 / 16 | 1 |

Through the whole app on the scoreboard's 16 cases: DISTORTION shown on 7 of 8, onto a second sound 49 on average
and 56 in the median, taken out 9 on average; SATURATION shown on 3 of 8 (onto a second sound 45, 85 and 61, taken
out 15, 12 and 52). The detector of build 0.5.0 showed 4 of the 8 distortions and none of the saturations, put a
made-up curve on the second sound (10 on average) and could not take anything out.

## What it does not do

- **Undo a hard bend.** Taking out scores about 20 where the group is shown, and that is the level relations put
  right, not the flat tops filled in. A real declipper, which makes up what the tops were from the sound around
  them, is another engine and is not in this build.
- **Boxes that are more than a curve** (`bplain.py`). One of the eight outside effects is a curve and nothing
  else: what it puts out is a rising curve of what went in, to the last bit. In the other seven such a curve
  leaves a part of the output unexplained, from 6 to 46 dB under it: they filter before or behind their curve, or
  mix sound in that is not in step. The two that are least like one curve (it leaves 10 and 15 dB under their
  output in the median, against 23 to 29 for the other five) are read worst: the squeeze at the peak is off by
  3.8 and 4.4 dB on average on them, by 1.1 to 2.8 on the others, and on single sounds by up to 12 dB. Of the
  five shown cases that go onto a second sound worse than nothing, two are of one such effect and read 9 dB too
  high, one is a case in which a curve explains least of all (6 dB under the output), and one is bent by 27 dB;
  for the fifth no reason was found.
- **A soft saturation** (under 6 dB at the peak) is mostly not shown: 3 of 23 sounds. It changes too little of the
  wave's shape to be told from the sound itself.
- **A steady tone through a box** reads as a tone. A square or a pulse wave reads as a distorted sound.
- **A limiter or a compressor** is not a bend (a level held down by a moving gain leaves the wave's shape alone)
  and is shown on 3 % of the limited test examples.
- **A bend with something behind it** (`bunder.py`, the 96 outside sounds; as they are, 45 of the 48 distorted
  ones and 21 of the 25 saturated by 6 dB and more are shown). Behind a low-pass at 8 or 4 kHz it is found as
  before; behind one at 2 kHz, 45 and 18 are shown and the squeeze reads 2.4 dB lower than without; at 1 kHz, 37
  and 13, and 4.5 dB lower. Under a room 18 dB under the sound, 42 and 21; 12 dB under, 42 and 17, read 3.4 dB
  lower; under a room 6 dB under the sound, 30 and 6, and the squeeze reads 8 dB too low. The group then does too
  little by that much: what comes after the curve fills the flat tops in.

## Files

`bfeat.py` what is read and the measured answer, `bchain.py` the family of boxes, `bgen.py` and `bgen_tone.py`
the training examples, `btrain.py` the nets, `bmeter.py` the meter read from the app's model file or the trainer's
files, `bengine.py` the engine. Evaluations: `bens.py` the held-back examples, `beval.py` the outside cases,
`bback.py` and `bback2.py` the ways back, `bgrid.py`, `bgrid2.py` and `brule.py` what is shown, `look10.py` and
`look12.py` the clean sounds that are shown, `look11.py` made steady tones, `bplain.py` how far an outside box is
one curve, `bunder.py` a bend behind a filter or under a room (with `common.py`). Model:
`Tools/export_bend_model.py` writes `Resources/bend.bin`. The scripts expect the working folders where they were
on the machine that made them.
