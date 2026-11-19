# The sides engine: Python side

`sides.py` is the prototype that `Source/Core/Sides.h` is a port of. `Tools/sides_parity.py` compares the two
on whole sounds (through `drsides`, built from `Tests/sides.cpp`): on the 16 scoreboard pairs of the HAAS and
WIDTH families the verdicts, the taps and every number of the rule are the same, the rendered sounds agree to
-151 dB or better, and Amount in the middle gives the sound back bit for bit. The other scripts are a record of
how the engine was built and judged, as they were run: paths point to the working folder they ran in (there
`sides.py` was called `stereo.py`), and no audio is in the repo.

## What is looked for
Two things that are done between the two sides of a sound:

- **HAAS**: one side arrives later than the other (1.25 to 40 ms).
- **WIDTH**: a sound from the middle is made wide. Every widener of the test set but one does it in one of two
  ways: the sound reaches the two sides through two short filters that differ (a copy a few milliseconds later,
  added on one side and taken away on the other), or a filtered copy of the middle is put at the sides.

Both are read off the sound itself, without the dry sound.

## The method
**(A) The pair.** One sound x reaches the sides through two sparse filters, left = a * x, right = b * x, so
left * b = right * a.

1. Both sides are brought to one level and sent through one filter (linear prediction, 40 ms long) that
   flattens their average spectrum. Without it a steady note explains itself from its own past as well as the
   other side explains it.
2. The taps (0 to 40 ms) are looked for with a penalty on their number (a lasso on the exact correlations of the
   whole sound). A side's own last half millisecond may not be used: two filters that differ only there are a
   difference in tone, not one of the effects looked for.
3. The taps found are fitted exactly in pieces of 5 ms. The pieces that fit worst are left out (a click, a
   plugin's glitch, a part of the sound that has no effect on it: up to half of the sound), and the taps that
   explain next to nothing go.
4. This is done both ways round (the right side told from the left one and from its own past, and the other
   way).

**(B) The copy.** side = h * middle, h at -5 to +5 ms, fitted on signals whitened by the middle's own spectrum.

**The check.** With the filters known, the one sound behind both sides is x = (A* L + B* R) / (|A|^2 + |B|^2).
What the two sides hold beyond a * x and b * x is measured per piece of 5 ms. That says where along the sound
the effect is, how much of the sound it explains, and which way round is the better one.

**The end of the file.** That formula takes the silence after the last sample for something both sides said.
Where a tap reaches beyond the end (the sound through a late tap is never heard there) it gets x too small: with
one side 15 ms late, the last 15 ms at about half. Those pieces then did not obey, the Amount fader left them
alone, and the sound taken out ended on 15 ms with the late side still in place (on one scoreboard case the room
engine read a room into that). So x over the last samples is the least-squares fit to what is in the file, the
samples before them kept (conjugate gradients; one tap a side: two steps), and a side that is moved in time is
made from x itself. On that case the sides of the sound taken out now differ by -107 dB (before: -57 dB, all of
it in the last 15 ms).

**The rule.** A pair counts when it has at most 8 taps, reaches at least 1.25 ms, makes a side worth speaking of
(-20 dB or more of white noise), is obeyed by half of the sound's energy or more, leaves at most -20 dB
unexplained there, and explains at least 10 dB more than one sound at two levels would (a sound simply panned
somewhere obeys any pair close to "no filters at all"). Otherwise the copy counts when it explains 90 % of the
side's power beyond a plain balance, the side is at -15 dB or more, and half the sound obeys. HAAS: the two
sides' main taps are 1.25 ms or more apart; else WIDTH.

**The Amount fader.** The sound is the rest plus the pair's part, and only the pair's part is changed, so the
middle of the fader is the sound itself. One tap on each side (a plain delay): the late side moves in time and
the levels come together, so no comb is heard on the way from the middle to the left. Anything else: a straight
line between the plain sound and the pair; to the right everything but each side's direct sound grows. Onto a
second sound each side goes through its own filter.

## Results
`runall.sh` runs every check (about 15 minutes on four cores), `st_report.py` prints them.

**Fresh cases** (`st7.py`: 14 draws for each of 8 outside effects on any clip of the corpus, 112 cases). "Best
possible" is the dry sound folded to mono: several of the tools fold the sound before they work, and what was
at its sides before is then gone for good.

| effect | found | taken out (median) | best possible |
|---|---|---|---|
| haas, four tools | 56 of 56 | 100.0, 99.3, 56.7, 59.5 | 100.0, 99.4, 56.7, 60.0 |
| width, a copy later on the sides | 14 of 14 | 53.3 | 53.3 |
| width, a comb between the sides | 13 of 14 | 87.1 | 100.0 |
| width, the middle through all-pass filters | 10 of 14 | 88.9 | 97.4 |
| width, the side simply made louder | 0 of 14 | | |

The last one cannot be read off a sound: nothing ties a side that was made louder to the middle. Margins of the
rule on the found pairs: taps at most 6 (limit 8), reach at least 110 samples (60), made at least -13.4 dB
(-20), obeyed by at least 0.96 (0.5), unexplained at most -22.7 dB (-20), beyond a panned sound at most -15.8 dB
(-10). Found copies: share at least 0.92 (0.9), side at least -12.2 dB (-15).

**False finds** (`st5.py clean`, `st5.py fams`): on the 198 clean clips (124 with anything at their sides) no
HAAS and no WIDTH. One clip fails a single criterion, and that by far: its pair reaches 2 samples where the rule
asks for 60 (a difference in tone between the sides); every other clip fails two or more. On the 400 sounds
of the scoreboard's 18 families: none outside the two stereo families (384 sounds), all 16 HAAS sounds as HAAS,
12 of the 16 WIDTH sounds as WIDTH (the 4 others are the side simply made louder).

**Scoreboard with the app's code** (`bench/famboard.py`, 8 cases a family; before = build 0.4.6):

| family | found | taken out, median | onto a second sound, median | worse than doing nothing |
|---|---|---|---|---|
| haas | 8 of 8 (before 6) | 98 (before 33) | 31 (before 17) | 0 of 8 |
| width | 6 of 8 (before 5) | 77 (before 11) | 45 (before -17) | 0 of 8 (before 5) |

"Onto a second sound" is scored against the outside tool's own output, and most of these tools fold the second
sound to mono first. The app does not: each side goes through its own filter and a sound that is wide already
keeps what it has at its sides. Where the second sound is mono the score is 100; where it is wide the two
results differ by the tool's fold, not by the effect.

**Long sounds** (`drsides find` on 120 s made of 20 clips): a HAAS on all of it is found in 5 s (the taps to 8
digits); with a loud minute in the middle and the rest 20 dB down the same; a HAAS on the first 80 s and plain
stereo clips after it is found and its stretches end at 80 s; plain clips only: nothing. A sound of more than 4
minutes: the pair is looked for in the 4 minutes with the most energy and checked on all of it.

## What it does not do
- A side that was simply made louder (or quieter) is not found; see above.
- An effect on less than half of the sound is not found (the rule's "obeyed by half of the sound").
- The dry sound's own stereo, where a tool folded it away, cannot come back: removal gives the one sound behind
  both sides, in the middle.
- A pair holds for the whole sound: a delay that changes along the sound is not followed.
- In a file cut from the middle of something longer, the late side begins with sound from before the cut, which
  nothing in the file explains: those first milliseconds (the delay's length) stay as they are.
