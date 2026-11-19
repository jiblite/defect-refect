# The crush engine: Python side

`crush.py` is the prototype that `Source/Core/Crush.h` is a port of. `Tools/crush_parity.py` compares the two on
whole sounds (through `drcrush`, built from `Tests/crush.cpp`): on 13 pairs (the 8 scoreboard pairs of the
BITCRUSH family, three fresh cases, two clean sounds) the same crusher is read (steps within a hundred-thousandth,
the hold the same), a bare crusher's sounds are the same sample for sample, the sounds of a crusher that mixes
plain sound in agree to -215 dB (-136 dB where the sound before the crusher is kept as 32-bit floats), and
Amount in the middle gives the sound back bit for bit. The other scripts are
a record of how the engine was built and judged, as they were run: paths point to the working folder they ran in,
and no audio is in the repo.

## What is looked for
A sound whose samples were rounded to a grid of values (fewer bits) and held for several samples each (a lower
sample rate):

    wet[t] = Q (dry[b(t)]),    Q(v) = offset + step * round ((v - offset) / step),    b(t) = the last hold boundary at or before t

and the crusher that mixes plain sound back in, with soft steps:

    v = m hold(x) + (1 - m) x,    wet = (1 - m) v + m Q(v)

Read off the sound itself, without the dry sound: each side's step and offset, the hold (a whole number of
samples or not), and for the second kind the share m and how far the steps are rounded off.

## The method
1. **The grid.** A sound on a grid of values has an amplitude distribution made of spikes a step apart, and the
   Fourier transform of that distribution has a line at one over the step: a narrow one, as wide as one over the
   range of the values, standing out of its surroundings. The step and the offset are then made exact by a
   straight line through the values themselves, with windows that narrow. A grid counts if 99.5 % of the sounding
   samples lie within a fiftieth of a step of it and it is 13 bits or coarser (a finer one is an ordinary file).
2. **The hold**: the places where the sound changes at all. With a hold of k samples they all lie k apart or a
   multiple of it; with a hold that is not a whole number (a rate that does not divide the file's) they lie on
   the pattern ceil (j P + c), whose slope and start are fitted. A hold counts from 2 samples, with 99 % of at
   least 50 changes on the pattern.
3. **Plain sound mixed in.** The values then lie around the grid values, not on them: each grid value has a
   stretch of width (1 - m)(1 - aa) of a step filled evenly, with sharp ends (aa: the share of a step over which
   one level blends into the next). Read off: the step at which the values folded onto one step show the
   sharpest ends, where those ends lie and how fast the values thin out beyond them, the hold from where the
   large changes fall, and with a hold the blend from the trace a wrong one leaves at the hold's boundaries.
   Because of the plain share nothing is lost: the map is one to one, and the sound before the crusher can be
   worked out exactly. It counts if the values lie 8 times denser around the grid values than between them.
4. Values that sit on a grid itself but only some of them: something came after the crusher (a filter). The grid
   is read and put onto a second sound as a bare crusher; what came after is not read.

A long sound: the settings are read from its loudest 30 s.

**The faders.** Amount onto a second sound: 0 = none, 1 = through the same crusher (the same grid of values in
absolute terms, the same hold), 2 = what the crusher changes twice over. Bit Depth: the steps a sixteenth to
sixteen times as large (4 bits either way). Sample Rate: the hold a quarter to four times as long. Each of the
two is shown when the crusher has that part. On the sound itself, a crusher that mixed plain sound in: Amount 0 =
the sound before the crusher, and from there every setting can be had. A bare crusher: what the rounding and the
hold threw away is gone, so left of the middle nothing happens; right of it the sound goes through steps twice
as large and a hold twice as long.

**In the chain.** A crusher is read only when it was the last thing the sound went through (anything after it
takes the values off the grid). So onto a second sound it goes on after the other found effects, and on the
sound itself it is worked first, on the sound as it was found.

## Results
All with the app's own code unless said otherwise.

**Fresh cases** (`bf7.py`: 14 draws for each of four outside crushers and of a plain one, on any clip of the
corpus, 70 cases):

| effect | found | step off by at most | hold | onto a second sound (median, least) | taken out (median, least) |
|---|---|---|---|---|---|
| plain | 14 of 14 | 0 | right on 14 | 100.0, 100.0 | cannot be |
| outside 1 (steps only) | 14 of 14 | 0 | right on 14 | 97.8, 73.6 | cannot be |
| outside 2 (steps and hold) | 14 of 14 | 0 | right on 14 | 80.1, 15.7 | cannot be |
| outside 3 (plain sound mixed in) | 14 of 14 | 0.0001 bits | right on 14 | 99.8, 96.1 | 99.7, 91.6 |
| outside 4 (a filter after it) | 1 of 14 | | | 75.5 | cannot be |

Outside 2 rounds by a rule of its own and holds another sample of the dry sound than the one at the boundary,
which the crushed sound does not tell: grid and hold are exact and the result still differs.

**Wrong finds** (`nf2.py`): none on the 198 clean clips, none on the 384 scoreboard sounds of the other
families; 14 of the 16 sounds of the BITCRUSH family found (the two missing: the crusher with a filter after it).

**Ordinary files** (`pcm16.py`): every clean clip saved as a 16-bit file (plain rounding, with dither, and 30 dB
quieter) and as a 24-bit file, four forms of 198 clips: no crusher on any of them. A file's own word length is
a grid of 16 bits or finer against full scale, and a grid counts from 13 bits.

**Scoreboard** (8 pairs, through the app's render test): found 7 of 8. Onto a second sound 91 in the middle case
(99.8 and 99.4, 86.0 and 100.0, 88.4 and 93.1 for outside 3, 1 and 2; 47.0 and not found for outside 4). Taken
out: 99.6 and 92.7 for outside 3, the others cannot be. Before the engine: found 7 of 8 by the detector, nothing
taken out, onto a second sound 10 in the middle case (-0.9 and 1.0 for outside 3).

**Other sample rates** (`r44t.py`: 16 clips each at 44.1 and 96 kHz, a bare crusher made at that rate): grid and
hold right on 16 of 16 at both. The app reads such a file at its own rate (after its change to 48 kHz the values
are off the grid) and carries the grid and the hold's length in time over: one case through the render test at
44.1 kHz gives 90.5 onto a second sound.

**Speed**: 0.2 to 0.4 s for a 6 s sound; with plain sound mixed in 1.2 s in the middle case, 1.9 s at most.

## What it cannot do
- A **crusher with a filter after it** is not read (found on 1 of 14 fresh cases, and then only its grid). The
  way there: undo the filter first, then read the grid.
- **Which dry sample a held value was**, and the crusher's **rounding rule** (nearest, down, towards zero): the
  sound does not tell. The nearest value and the sample at the boundary are taken.
- A **bare crusher cannot be taken out**.
- A crush under another effect is not found: the values are off the grid.
- A **16-bit file made louder by 18 dB or more** and kept as it is (no dither, nothing else done to it) sits on
  a grid of 13 bits or coarser against full scale, and is read as a light crusher.

## Files
`crush.py` the prototype. `bf7.py` fresh cases (made by `fc7.py`), `bc2.py` the scoreboard pairs (prototype),
`nf1.py` / `nf2.py` wrong finds (prototype / the app's code), `pcm16.py` ordinary 16-bit and 24-bit files,
`r44t.py` other sample rates; `fc7.py`, `nf1.py`, `nf2.py`, `pcm16.py` and `r44t.py` are in `Research/stutter` (they
serve both engines), `st_common.py` in `Research/sides`.
