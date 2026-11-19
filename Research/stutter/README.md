# The stutter engine: Python side

`stutter.py` is the prototype that `Source/Core/Stutter.h` is a port of. `Tools/stutter_parity.py` compares the
two on whole sounds (through `drstutter`, built from `Tests/stutter.cpp`): on 14 pairs (the 8 scoreboard pairs of
the STUTTER family, four clips with a stutter of our own, two clean sounds) the events, the stretches and
everything left out are the same, every rendered sound is the same sample for sample, and Amount in the middle
gives the sound back bit for bit. The other scripts are a record of how the engine was built and judged, as they
were run: paths point to the working folder they ran in, and no audio is in the repo.

## What is looked for
A slice of the sound played several times over, in place of what came after it:

    wet[s0 + j L + i] = wet[s0 + i],    j = 1, 2, ..  up to the event's end

One event is a start s0, a slice length L (20 ms to 2 s) and an end. Read off the sound itself, without the dry
sound: every event, and whether each play comes up from nothing and goes back to it (a fade of 2 ms).

## The method
1. **The table.** Every window of 64 samples goes into a table by its samples (rounded to 2^-16), so two windows
   with the same samples land in the same place. For each window the nearest earlier place with the same samples
   is its distance; windows of one value over and over (silence) say nothing. Runs of windows with one distance
   are the candidates.
2. **Stretches.** Each run is walked out from its middle, sample by sample, to the first and the last sample that
   is what the sample L before it was (within 2^-18). A stretch counts if it is at least half a slice long (one
   and a half plays), 20 ms or more, and holds enough samples that are neither silent nor standing still. One
   that is mostly silent counts only beside one of the same slice length that is not.
3. **Events.** A slice starts where its stretch starts, or where the event before it ends when the two follow
   straight on each other. It ends after the fewest whole plays that hold the stretch. Samples that are silent or
   equal to their neighbour say nothing at either end: they are equal whatever was played. Another event that
   starts a whole number of plays on ends this one, even where the sound goes on being equal for a while (both
   slices begin alike).

**What is not an event although it repeats exactly:**
- silence, and a sound that stands still (a steady offset);
- a sound that repeats of its own accord, a loop: one stretch from the first 5 % to the last 10 % of the sound,
  or, for a slice over 0.3 s, a tenth of the sound repeating at that distance outside the stretches;
- a steady wave, a held note of a machine: a slice of it "played again" shows nothing where one play ends and
  the next begins. The wave's period is looked for only past the point where its likeness to itself has first
  fallen under a half (a slow wave is like itself a few samples on, which is smoothness and not a period);
- the same note struck again with silence between (every play stands by itself in silence);
- a sound that goes on the same way before the slice or after the last play, at another loudness: a note that is
  itself a short loop fading in or out, a loop through an effect that has to settle.

**The faders.** Onto a second sound: its own slices are played again at the same times; Amount 0 = none, 1 = as
found, 2 = each slice twice as many extra times; Slice: the slices half to twice as long, over the same stretch.
On the sound itself: right of the middle each slice is played more often, over what follows it. A stutter cannot
be taken out, since what the repeats covered is gone. Left of the middle each slice is played fewer times and
the sound closes up, so it ends early (silence at the end): the dry sound with the covered parts missing. Slice
on the sound itself: only shorter (a longer slice is not in the sound).

**In the chain.** A stutter is read only when it was the last thing the sound went through (anything after it
makes the repeats unequal). So onto a second sound it goes on last, after every other found effect and after
Tone Match. On the sound itself it is worked last too: the other engines' answers belong to the sound's own
times, and closing up moves them.

## Results
All with the app's own code unless said otherwise.

**Every clip of the corpus with a stutter put on** (`sk2.py`: 198 clips, slices of 25 to 400 ms, 2 to 8 plays,
with or without fades, 1259 repeats made, 1127 of them on a slice that sounds and moves): found on 198 of 198;
1152 repeats read, 1138 right to the sample, 12 with the right start and slice but another end, 2 read that
were not made, 2 missed. Fades told right on 198 of 198. The events put onto another clip against that clip
through the same stutter: 100.0 of 100 in the middle case, lower quarter 99.7, least 13.7.

**Fresh cases** (`sf7.py`: 16 draws for each of two outside tools and of a stutter of our own, 48 cases):

| effect | found | events to be seen | right to the sample | not made | missed | onto a second sound (median, least) |
|---|---|---|---|---|---|---|
| outside tool 1 | 16 of 16 | 209 | 217 | 0 | 0 | 99.8, 36.4 |
| outside tool 2 | 16 of 16 | 226 | 225 | 0 | 2 | 99.8, 43.1 |
| plain | 16 of 16 | 81 | 79, and 1 within 8 samples | 1 | 2 | 100.0, 19.3 |

(More can be right than are "to be seen": that count leaves out slices under -60 dB, some of which are read all
the same. The low scores are sounds where an event on a silent slice was missed, see below.)

**Far left** (`so1.py`, the same 48): the found sound with Amount at 0 against the dry sound with the covered
parts cut out: 98.0, 98.2 and 99.4 of 100 in the middle case for the three effects. Sample for sample the same on
the cases without fades where every event is right; with fades the 2 ms at each slice's edges stay (-21 dB in the
middle case). Where an event was missed everything after it lies later than in that dry sound (least -42.5).

**Wrong finds** (`nf2.py`): none on the 198 clean clips, none on the 384 scoreboard sounds of the other
families; all 16 of the STUTTER family found. Nor on the 198 clean clips saved as 16-bit files (plain rounding,
with dither, 30 dB quieter) and as 24-bit files (`pcm16.py`).

**Scoreboard** (8 pairs, through the app's render test): found 8 of 8; onto a second sound 90 in the middle case
(65.5 to 99.8). Before the engine: found 8 of 8 by the detector, and what it put onto the second sound was worse
than leaving it off on all 8 (a stutter with made-up places). The scoreboard's "taken out" column does not apply:
it compares with the dry sound at the same times.

**Other sample rates** (`r44t.py`: 16 clips each at 44.1 and 96 kHz, a stutter made at that rate): found on 16 of
16 at both; 143 of 144 and 133 of 133 events on a sounding slice right to the sample. The app reads such a file
at its own rate (its change to 48 kHz would make the repeats unequal) and moves the events to 48 kHz: one case
through the render test at 44.1 kHz gives 99.4 onto a second sound and 99.9 at the far left.

**Speed**: 0.02 s for a 6 s sound (the prototype: 0.1 to 3 s).

## What it cannot do
- A **slice that is silent or stands still** is not seen, and where a cut lies inside silence cannot be known.
- **Two events one after the other whose slices hold the same samples** (an exactly periodic machine sound) are
  read as one.
- A **repeat that is not exact** is not a repeat here: a stutter under another effect, after a change of the
  sample rate, or after a change of the word length with dither is not found.
- A stutter **cannot be undone**. The far left is the dry sound with the covered parts missing, shorter by what
  was covered.
- With fades, the 2 ms at each slice's edges are left as they are.

## Files
`stutter.py` the prototype. `sk2.py` every clip, `sf7.py` fresh cases (made by `fc7.py`), `so1.py` the far left,
`su1.py` the scoreboard pairs (prototype), `nf1.py` / `nf2.py` wrong finds (prototype / the app's code),
`pcm16.py` ordinary 16-bit and 24-bit files, `r44t.py` other sample rates; `st_common.py` (in `Research/sides`) is
what they share.
