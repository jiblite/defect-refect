# The exciter study (build 0.5.1): tested, and withdrawn

An exciter takes the sound above a corner, drives it through a curve and mixes the result back in. What comes out
has more highs than went in: partly the old highs made louder, partly new partials an octave and more above what
was driven. Up to build 0.5.0 the app's detector could show an EXCITER group, and the old engine behind it put a
made-up exciter onto the second sound. On the scoreboard that group was found on 1 of 8 cases, and the one find
scored far below doing nothing. The question for 0.5.1 was whether an engine can do for the exciter what the other
engines do for their effects: read it off the sound and work on what was read.

It cannot, with what a single file holds. The EXCITER group is gone in 0.5.1.

## What the outside exciters do (`exc1.py`)

20 cases from three outside tools (8 on the scoreboard, 12 fresh ones), each with two sounds, 40 excited sounds in
all, each with the sound as it was before. Measured on the pairs: the exciters lift the bands above 2 to 4 kHz,
one tool by 1 to 4 dB, one by 6 to 17 dB, one by 7 to 35 dB, and nearly by the same amount in quiet and in loud
frames. So in the long-term spectrum an exciter is a high shelf.

## Two ways to tell it from the sound alone

- **New partials that hang on the old ones (`exc0.py`).** If the highs were made from the upper middle by a curve,
  they can be foretold from the upper middle, squared and cubed. Fitted on half of each sound and tested on the
  other half, on the scoreboard's eight cases and with the corner at 1.5, 2.5, 4 and 6 kHz: the share of the highs
  foretold this way is 4 to 12 % in the median on the excited sounds and 5 to 10 % on the same sounds before the
  exciter. It is higher after the exciter on 2 of the 8 cases and lower on one of the two heaviest (32 % after,
  70 % before): a sound's highs hang on its upper middle by themselves, and the tools filter before and
  after their curve, so what a single curve would leave behind is not there to be read.
- **The lift itself (`egen.py`, `eeval.py`).** A reader of the same kind as the fixed filter's (band energies of
  the whole sound into a small net, `Research/filter/`), taught on 260,000 made examples of exciters over the
  training clips, with plain bright sounds, equalisers and filters as the things it must not take for one. On its
  own held-out examples it is right on 79 % of the excited sounds and takes 14 % of the plain ones for excited.
  On the outside cases (`eeval.py`):

  | sure above | excited sounds found (of 40) | the same sounds before the exciter (of 40) | clean corpus |
  |---|---|---|---|
  | 0 | 13 | 4 | 11.6 % |
  | 1 | 12 | 2 | 6.1 % |
  | 2 | 10 | 2 | 3.0 % |
  | 3 | 8 | 2 | 1.5 % |

  The eight it finds are the heavy-handed ones (lifts of 15 dB and more). The lift it reads is off by 3.6 dB rms
  against the true one; reading no lift at all is off by 4.4 dB.

A sound with 8 dB more highs than it had is, to anything that looks at that sound alone, a bright sound. The
reader can only know what sounds are usually like, and sounds differ in their highs by more than most exciters
add.

## What the app does instead

- `dropExciter` in `Source/Core/Estimate.h` takes the group out of what the detector may show. Nothing else of
  the detector changed.
- With two sounds, TONE MATCH carries the first sound's long-term highs over to the second, which is what an
  exciter leaves in a spectrum.
- By itself, a sound with added highs is shown as it is: no group for them. The fixed filter's group is for cuts
  (a low-pass, a high-pass, a band-pass) and does not take a lift for one.

## Files

`exc0.py` the first look, `exc1.py` the cases as they are, `egen.py` the training examples, `eeval.py` the reader
on the cases, the other effects' sounds and the clean corpus. The reader itself is trained with the fixed
filter's script (`Research/filter/`, pattern `e`). The scripts expect the working folders where they were on the
machine that made them.
