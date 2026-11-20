# The declipper: tops that were cut flat, rebuilt (build 0.5.2)

Build 0.5.1 walks a distortion back along the curve it has read, and stops where the curve is flat: where many
levels were pressed into one, nothing in a value says how high it stood. The sound around it does. A real sound is
made of few partials that last a while, and the cut-off tops are what is missing for them to go on smoothly. This
engine rebuilds the tops from that, and it sits behind the far left of DISTORTION and SATURATION.

There was a second reason. Build 0.5.1 does harm to a plain hard cut: the meter reads a soft, blurred curve where
there is a sharp cut, and walking that curve back spreads values apart that were never squeezed. On twelve sounds
through outside hard clippers its far left scores -69 on average, down to -283 (100 = the sound before the
clipper, 0 = left alone).

Prototype here (`dengine.py`), the app's code in `Source/Core/Declip.h` and the flat-top part of
`Source/Core/Bend.h`, compared by `Tools/declip_parity.py` through `Tests/bend.cpp`: on nine cases that take every
path the two reach the same verdicts, and the sounds agree to -102 dB or closer.

## Three kinds of top (`stats`, `regimes`)

Each half-wave is looked at by itself, the values above zero and those below.

- **A plain cut.** Many values sit at exactly one level, the top (16 or more, and 0.2 % of the half-wave's live
  values, within 0.01 dB), the values just under them are no denser than those 1 to 2 dB further down (at most
  1.3 times as dense), and the next value down holds under a third as many as the top. Nothing under the cut is
  touched. This is looked for on every sound, whatever the meter says.
- **A soft flat top.** The meter has found a bend, 1 % or more of the values lie within a quarter dB of the peak,
  and they lie denser there than 1 to 2 dB under it: a pile under the top with a knee below it.
- **Neither.** The curve as the meter read it, walked back the careful way. Nothing here changes: such a sound
  comes out of build 0.5.2 as out of 0.5.1.

A file at another rate than 48 kHz is read at its own rate: its tops are flat only in its own samples.

## The rebuilding (`solve`, `rebuild`)

Some values are known. The others are only known to lie at a bound or beyond it. The answer is looked for among
sounds that agree with the known values, stay beyond the bounds, and need few cells of a time-frequency picture
(windows of 2048 samples every 1024, a sine window, so that the way to the picture and back gives the sound again;
4096 at 96 kHz). A cell counts together with its two neighbours in time, which is what makes partials that last
win over clicks: social sparsity with a persistent empirical Wiener shrinkage (Siedenburg, Kowalski and Doerfler,
2014). The cells' threshold comes down in steps, from 0.3 to 0.0003 of the level that 999 of 1000 of the picture's
cells stay under, a few rounds per step, each round a step on from the last in the direction it moved.

A sound is rebuilt stretch by stretch, so a long file never has its whole picture in memory: unknown values less
than six hops apart belong together, a long run is cut into parts of at most 192 hops, and each part is solved
with three hops of its known surroundings, on up to eight threads. Window, hop and the size of the neighbourhood
matter little (tried on plain cuts); the rounds do: a plain cut gets 8 steps of 12 rounds, a soft top's first pass
16 of 25.

## A plain cut

The values at the cut are unknown at or above it, all others are known as they are. With the level of the cut
known exactly, about half the damage comes back on sounds cut by 3 to 18 dB (scores of 40 to 51, a study with a
cut made here). A dense, noise-like sound comes out worse than it went in (with 63 % of its values cut: -24): a
cut hardly changes such a sound's picture, and nothing in it says what the tops were.

## A soft flat top: two passes

*First pass.* The meter's reading is walked back up to a first height, the level from which the curve turns flat
(the bent value moves less than half a dB per dB of the value before the bend). Everything above that height is
unknown, at or above the height's own value before the bend, and is rebuilt.

*What the rebuilt values tell.* Per level of the bent sound, take the middle one of the rebuilt values, and make
the row rising: that is the curve above the first height, read off the sound with no model of the box. On the six
cases of the one outside effect that is a curve and nothing else, tried with the curve under the first height
given, it matches the true curve within 8 % far up.

*Second pass.* The lower four fifths of the zone above the first height are walked back along this curve and
count as known. Only the very top is rebuilt once more (8 steps of 12 rounds). In that trial the six cases score
55 to 94 this way, where the first pass alone gave 8 to 85.

## What keeps it from doing harm

All of it is read off the sound itself.

1. **The rebuilding is tried one step down first** (`selftest`). The known values are cut again 3 dB lower,
   rebuilt, and held against what was there, on the three seconds of the sound with the most values above the
   test's cut. 100 = the rebuilt band matches, 0 = no better than the cut. Under 20 nothing is rebuilt: a plain cut then
   stays as it is, and a soft top goes the careful way of build 0.5.1. Dense, noise-like sounds fail it.
2. **Nearly a square wave is left alone.** A soft half-wave with three quarters of its live values at the very top
   has nothing left to go on.
3. **One wave, one way.** A side that reads as a plain cut next to a soft side, and fits a soft top too, is taken
   as soft. (A symmetric sound whose two sides fell either side of the 1.3 above came out at 29; taken one way, 63.)
4. **A first pass that contradicts itself is refused.** How far its values lie from the curve at their level of
   the bent sound, on average: over 3.5 dB nothing is rebuilt. One case of all that were tried stands there, at
   4.5; the others are at 3.2 or under.
5. **A second pass has to earn its place.** It is thrown away, and the first pass is the answer, when it leaves
   the tops more than three times as rough as the first did (the summed squares of the second differences inside
   tops of five values or more), or when the first pass does not bear the curve out (higher bent values did not get
   higher values: a rank correlation in the upper half of the pairs at or under zero). Such flat tops were not a
   curve's work: a clipper that works at two or four times the rate and comes back down leaves ripple on its tops,
   and the ripple is not the level. On the trial below the second passes that did harm are 3.4 to 11.4 times as
   rough and the good ones 2.7 at most, and the two harmful ones that are smooth have a negative correlation.

## Long sounds, other rates

A soft top's first pass costs the most rounds, and its use is to tell the curve. On a sound whose unknown values
span more than 900 hops per channel (19 s at 48 kHz) it works on 900 hops in parts of 48, spread evenly over the
sound, and the curve they tell serves the whole sound; the second pass goes over all of it. On four sounds of 48 s
that costs 1.3 points on average against a first pass on everything (58.4 against 59.7, with 600 hops 57.7, with
1200 hops 59.3) and takes 12 s in place of 20 on four threads; a plain cut of that length takes 5 s. (Working on
the loudest parts in place of evenly spread ones lost half the gain.) A sound of more than 20 minutes is not
rebuilt.

Files at 44.1 and 96 kHz and mono files, three sounds each with a plain cut and a soft one made here
(`mk_rates.py`, `rates.py`), score as the same sounds at 48 kHz do: a plain cut 89 / 89 / 89 / 89, 48 / 49 / 47 /
49 and 59 / 58 / 57 / 58 (44.1, 48, 96, mono), a soft top 74 / 61 / 59 / 61, 81 / 83 / 84 / 76 and 72 / 75 / 76 /
74.

## The fader, a second sound, what is shown

- **The sound itself.** With x the sound before the bend (rebuilt where its tops were flat) and F a half-wave's
  curve, the fader gives F(g x) + w (y - F(x)): g is the drive the fader stands for, and w = g squared up to 1.
  What the curve does not explain of the sound in hand comes back as the fader nears its middle, so the middle is
  the sound itself to the last bit, far left is the rebuilt sound, and right of the middle the rebuilt sound is
  pushed into the curve again, further in.
- **The level.** The result keeps the sound's peak (its value without the largest 1 in 10000) where it was, as in
  build 0.5.1, and besides that no value comes out higher than the sound's highest was. Rebuilt tops stand above
  that peak, here by 3 dB. Most clipped files are cut at full scale, and there such tops came out over full
  scale, where the app's limiter pressed them down again: with the test file brought to full scale a score of 44
  fell to 29. With the second condition it stays at 43 (`fullscale.py`).
- **Onto a second sound.** A soft flat top goes on as the meter read it, as in build 0.5.1. A plain cut whose tops
  were rebuilt goes on as a cut, the second sound's peak as far past it as the first sound's rebuilt peak stands.
  On the six such cases among the twelve outside hard clippers the rebuilt peak stands 6.9, 11.2, 17.7, 8.9, 10.3
  and 7.8 dB above the cut where the real one stood 6.6, 11.2, 16.6, 11.5, 14.1 and 6.8, and the second sound
  scores 87, 100, 80, 73, 32 and 62: 72 on average, where 0.5.1 scored 13 (four of the six it did not show;
  `second.py`). A cut
  whose tops are not rebuilt goes on as the meter read it, exactly as in 0.5.1: how hard the sound was driven is
  not known there (see the limits).
- **Shown.** A plain cut whose rebuilt peaks stand half a dB or more above the cut shows as DISTORTION by itself
  (sureness 90 %), whatever the meter said.

## Results

Scores are for Amount at far left: 100 = the sound before the effect, 0 = left alone, under 0 = made worse. All
numbers are from the app's code as the app uses it (`trial.py`; it acts only on a group it shows).

**84 outside cases** (the 48 distortion and saturation cases of `Research/bend`, and 36 through outside clippers:
`clip_fx.py`, `mk_clip.py`):

| | cases | build 0.5.1 | build 0.5.2 |
|---|---|---|---|
| a soft flat top | 15 | -1 (8 worse than left alone) | 58 (1 worse) |
| a plain cut | 13 | -64 (8 worse) | 22 (none worse) |
| neither | 56 | 20 (6 worse) | 20, the same to the last bit |
| all | 84 | 3 (22 worse) | 27 (7 worse) |

No case comes out worse than from 0.5.1. Of the 13 plain cuts seven are rebuilt (six of them by enough to be
shown: 15, 31, 44, 49, 55 and 89, where 0.5.1 gave -132, 0, -68, -83, 0 and 0) and six fail the self-test and are
left alone (0.5.1: -283, -142, -79, -40, -4, 0). Of the 15 soft tops 13 keep their second pass.

**162 more soft clippings** (`mk_many.py`: eight soft curves of an outside clipper, each at its own rate and
working at 2 and 4 times the rate, six of each, driven by 8 to 28 dB; and 18 through the outside distortion that
is a curve, driven by 14 to 34 dB; clean sounds from the whole corpus). 63 are rebuilt, 99 are not (81 have no
pile at the top, three are nearly square waves, 14 fail the self-test, one is refused by rule 4):

| | cases | rebuilt | of these: better by 5 or more | worse by 5 or more | mean gain over 0.5.1 |
|---|---|---|---|---|---|
| a curve at the file's own rate | 48 | 27 | 23 | 3 | +50 |
| working at 2 times the rate | 48 | 15 | 14 | 0 | +78 |
| working at 4 times the rate | 48 | 10 | 6 | 2 | +32 |
| the outside distortion | 18 | 11 | 11 | 0 | +44 |
| all | 162 | 63 | 54 | 5 | +53 |

Over the 63 the mean goes from -10 to 43 and the median from 4 to 54; 30 of them came out of 0.5.1 worse than
left alone, 7 still do. Over all 162 the mean goes from 4 to 25. The safeguards were set on this trial: the first
engine, with one rule for the second pass (how densely its result fills the picture), gained 40 points on the
cases it rebuilt and lost by 5 or more on ten.

The five that still lose: three lightly bent sounds (the bent sound is 0.5 to 0.8 dB from the clean one on the
score's scale, so 0.15 to 0.4 dB of difference is 16 to 79 points), one clipping at 4 times the rate (34 to 17),
and one sine-shaped clipping with 70 % of its values at the top (10 to -5). A sixth case loses 12 points without
being rebuilt: one side reads as a plain cut, the self-test refuses, and that side is then left as it is where
0.5.1 walked it back.

**Sounds without a cut.** Of the 198 clean clips four are shown as bent where the Sureness slider stands at
first, the same four as by 0.5.1 (`clean.py`); one of them, a loop that the meter takes for saturated, has a pile
at its top and is now rebuilt at far left. On the scoreboard's other 120 cases a first run showed DISTORTION on
one sound that a crusher had cut down to 3 bits: the top step of such a sound holds many equal values, as a cut
does. But its next step down held a hundred times as many, and under a real cut there is next to nothing: under
the cuts of the outside hard clippers the top holds 150 times as many values as the next value down, or more. (The
lowest of all sounds that the engine takes as plain cuts, a soft clipping at four times the rate, has 3.4 times
as many; a sine cut down to steps has at most 2.4 times as many.) Hence the third condition for a plain cut, at
3 times.

**Through the whole app** (`Research/room/bench/famboard.py`, 136 cases of 17 effect types): two of the 16
distortion and saturation cases have a flat top, and taken out they score 80 and 54 (0.5.1: -10 and 29). Every
other case comes out as from 0.5.1, groups and scores (one distortion moves by 0.07 of a point: its file sat at
full scale, and no value comes out higher than the file's highest was now). The scoreboard has no hard clipper;
for a cut on a second sound see `second.py`.

## What was tried and dropped

- *Why a first pass lands too high.* The worst case of the trial came back 6 to 8 dB too tall. It is not the
  meter: with the true curve in place of the reading the first pass is still 6.4 dB over. Its waves were
  flat-topped and steep-flanked before the clipping (on its longer tops the true peak stands 7 dB over the top's
  edge, the first pass makes arcs of 21 dB), and a sparse model takes a steep flank for the foot of a big arc. The
  meter is fooled the same way (it reads 21 dB where 14 are true). Rule 4 catches this case, not the kind.
- *Making the self-test see a rebuild that lands too high:* scores on the test's band alone, a test cut 6 dB down,
  and how the test's rebuild agrees with the first pass. None separates good cases from bad.
- *The careful reading for the first pass's walk* (worse), *a cap by the meter's top* (the meter is as wrong), *a
  firmer verdict of the meter before a soft top is rebuilt* (thin evidence: one of the best cases sits right at
  the line).
- *How densely the second pass fills its picture against the first* as the test for rule 5: it let four of nine
  harmful second passes through and threw good ones away.
- *A cut put on a second sound by its share:* the second sound driven so far that as many of its values are cut
  as of the first sound's. It was checked on a cut put onto the first sound's own clean version only, where it
  scores 100 by construction. The same drive cuts very different shares of two sounds (6 % of one and 1.5 % of
  the other, 51 % and 14 %): on the six cases with rebuilt tops it scored -9 on average, and on the twelve it was
  worse than 0.5.1 on six.

## What it does not do

- **A sound without a pile at its top** stays as in build 0.5.1: 56 of the 84 cases, most of them through
  effects that are more than a curve (they filter before or behind it, or mix sound in) or that bend softly.
- **Dense, noise-like sounds** are left alone, and a long file is judged by the three seconds with the most cut
  values.
- **A sound clipped almost to a square wave** is not rebuilt.
- **Waves that were flat-topped before the clipping** can come back too tall. One such case is caught; the kind
  cannot be told from the sound.
- **A clipper that works at a higher rate inside** mostly gets the first pass only, and is rebuilt less often (10
  of 48 at 4 times the rate).
- **A lightly bent sound** can come out a touch worse than by the old way.
- **Cut levels that differ between the channels** are not handled (one level per half-wave for both channels).
- **A file with a few clipped peaks** (rebuilt by under half a dB) does not show DISTORTION by itself.
- **A cut whose tops are not rebuilt, on a second sound.** How hard the first sound was driven is then not
  known. The second sound goes through the curve as the meter read it, as in 0.5.1, and on a dense sound the
  meter reads a plain cut 6 to 9 dB too deep (23, 15 and 9 dB where 14.5, 6 and 3 are true). On three of the six
  such cases the second sound comes out far worse than untouched (-175, -137, -698), as it did in 0.5.1.
- **Time.** A soft flat top costs half to nearly all of a short sound's length in waiting on four threads (few
  stretches to share out, and twice that at 96 kHz), about a quarter of a long one's; a plain cut about a tenth.
- The thresholds rest on about 250 cases.

## Files

`dengine.py` the engine, `clip_fx.py` and `mk_clip.py` the outside clippers and their 36 cases, `mk_many.py` the
162 soft clippings, `mk_long.py` long cases, `mk_rates.py` files at other rates and mono, `trial.py` the app's
code against build 0.5.1 on a folder of cases, `rates.py` the same on the files of `mk_rates.py`, `clean.py` the
clean clips, `second.py` a found cut on a second sound (the whole app against build 0.5.1's, twelve cases),
`fullscale.py` clipped files brought to full scale. Before a build: `mkcheck.py` makes 40 sounds that take every path of the engine or could trip it
(nothing known, one side or one channel cut, a cut that holds for a second, files at 22 to 192 kHz), `check.sh`
runs the engine's own test program on them under the address and thread sanitizers and under valgrind, and the
stage `flat` of `Research/moving/memcheck.py` runs them through the whole app. The scripts expect the working
folders where they were on the machine that made them.
