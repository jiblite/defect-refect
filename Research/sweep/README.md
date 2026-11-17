# The flanger engine: Python side

`sweep.py` is the prototype that the single-sound part of `Source/Core/Sweep.h` is a port of.
`Tools/sweep_parity.py` compares the two on whole sounds: the same answer to "is it there" on 8 of 8 sounds,
the numbers behind the decision agree to 5e-11, the delay line to 2e-12 samples, and the rendered sounds to
-252 dB or better. What `Sweep.h` does with long sounds (pieces, the joined line) and its test of where along a
sound the copy is were written in C++ only and are tested from here through `drsweep` (`flong.py`). The scripts are a record of how the engine was built and
judged, as they were run: paths point to the working folder they ran in, and the audio is not in the repo.

## The method
A flanger is the sound plus a copy of itself whose delay moves (0.2 to 16 ms here). In the cepstrum of a short
stretch of the sound that copy is a peak at its delay, as high as its gain. Over the whole sound the peak draws a
smooth line.

1. `picture` the cepstrum of one channel frame by frame (one frame every 5.3 ms), short delays from short
   windows and long delays from long ones (four window lengths, 512 to 4096 samples; delays of 8 to 768 samples).
2. `track` the smooth line through the picture that collects the most (dynamic programming; the delay may move
   10 samples per frame, moving costs), once for a copy that is added and once for one that is subtracted, through
   the whole picture and through its moving part (the picture minus what stays put for 165 ms: held notes).
3. `candidate` follows a line up: the delay to a fraction of a sample (`refine`), a periodic curve (sine or
   triangle, 0.06 to 4 Hz) where one fits (`lfo_fit`), and the copy's gain and feedback, corrected in rounds until
   taking the copy out leaves nothing along the line.
4. The copy is taken out by the exact inverse of the moving filter (`_run`, inverse) and put onto another sound by
   the same moving filter at the same times. Speed, Sweep and Feedback faders change the movement (`movement`).

## Is it a flanger? (the rule, `accept`)
A melody draws lines into the cepstrum too (a note's pitch period is a peak), and so do formants. A line counts
as a moving copy when all of this holds, per channel:
- strong enough (gain 0.2 or more), seen in at least 15 % of the frames, moving by at least 12 samples, and
  reaching a delay of 1.3 ms (below that the picture shows the sound's own moving colour);
- not ringing like a pitch (feedback at most 0.75, and the first peak no higher than a copy as loud as the sound);
- moving like a sweep: at most 30 % of its way in jumps, standing still for at most 60 % of the time (80 % when
  it follows a periodic curve);
- gone after removal: what is left along the line is under 30 %;
- the level test: above `2 * SR / move` Hz the sweep carries every partial of the sound across at least two
  teeth of the comb, so a copy of gain g adds `10 log10(1 + g^2 / (1 - fb^2))` dB there whatever the sound is.
  Taking the copy out must take away between 0.35 and 2.2 times that. A comb pressed onto a sound that has no
  copy makes it louder; a note's own harmonics lose much more.

The last test is what made the rule safe (`stats.py`, `rule.py`, `levelb.py`, `final_stats.py`):

| sounds of 6 s                                      | first rule (no level test above 2 SR / move) | final rule |
|----------------------------------------------------|----------------------------------------------|------------|
| clean corpus sounds, dev split: falsely reported   | 9 of 115                                     | 0 of 115   |
| clean corpus sounds, test split                    | 5 of 83                                      | 0 of 83    |
| clean training clips                               | 4 of 133                                     | 1 of 133   |
| synthetic flangers found, dev / test               | 27 of 39 / 16 of 27                          | 28 of 39 / 17 of 27 |
| taken out where found (median), dev / test         | 71 / 70                                      | 82 / 75    |
| found ones made worse than doing nothing           | 4                                            | 0          |

Of the 21 synthetic flangers the final rule does not find, 12 have no tracked line within 2.5 samples of the true
one, 4 pass on one channel only or with differing signs, 4 are followed up and refused, and 1 has no line strong
enough. By kind of sound: loops 12 of 13, recordings 9 of 14, drums 3 of 3, synthetic speech 0 of 5.

## Outside effects (the scoreboard's kind of case)
`ev.py` runs the prototype on the bench cases (four flangers from outside tools, two settings each, both sounds
of a case): found on 5 of 16 sounds, never worse than doing nothing. Where found: out 91, 89, 96 (three sounds
whose line is seen in 99 % of the frames) and 39, 29 (a light flanger, copy at 0.38, seen in 57 to 72 %); onto the
case's other sound 92, 91, 94 and 27, 61. Not found: the two thru-zero cases (the dry path is delayed as well, so
the copy's delay crosses zero), a setting whose line moves by 4 samples only, and a light copy (0.26) that the
level test refuses; taking that one out would have gained 1 to 14 of 100.

In the app (`Research/room/bench/famboard.py`, 8 cases): FLANGER shown on 4 of 8 (0.4.2: 0 of 8), out 91, 89,
96, 39, onto 91, 91, 94, 27.

What a perfect measurement could reach (`oracle.py`: the true moving comb, measured with the dry sound known;
`ev_or.py` uses it for both directions) and what a correction of the moving tone curve alone could reach
(`ceil_eq.py`: the true magnitude response per moment, no phase): the tone curve takes a flanger out at 53 in the
median and puts it on at -8 (12 of 16 worse than nothing), a chorus at 44 and -22. Putting a flanger or a chorus
onto another sound needs the copy itself, not its colour. `shift.py`: where "onto" scores low, the line is
wrong in places, not late (no time shift improves it).

The same engine on the chorus, phaser and filter-sweep cases finds nothing, 0 of 16 sounds each (`ev.py chorus`
and so on). Those three need engines of their own.

## Long sounds, and where the copy is (`Sweep.h` only)
The picture of a long sound takes a megabyte a second several times over, and the rule was set on sounds of 6 s.
So a sound of 24 s or more is measured in pieces of 12 to 24 s, each like a sound of its own (`findPiece`), and
the pieces that show the copy are put together (`join`). Taken out, every such piece loses the copy with its own
line, gain and feedback: one moving filter through the whole sound (`runVar`) whose gain is 0 in a piece that
shows nothing and whose settings run into each other over 20 ms at the borders (`settings`). When only one of
several sounding pieces shows the copy the group's sureness is 0.5 instead of 0.9, which the default Sureness
does not show.

A line is tracked through the whole sound, also through a stretch that has no flanger: there it holds on to
whatever the picture has, and the inverse filter would press a comb onto a sound that has none. So a line that
passes the rule is asked, half second by half second, what the rule asks of the whole sound (`presence`):
- is there a moving line? What the picture holds along the smoothed line (`smoothLine`: a parabola through the
  24 frames either side), less what stands still there, against the line's height on this sound: 40 % and more
  is a yes, 20 % cannot be told, nothing is a no. A line that stands still (the copy at a turn of its sweep, or a
  held note) cannot be told either;
- does taking the copy out lower the level above `2 * SR / move` by 0.2 to 2.8 times what a copy of that gain
  adds? If not, it is a no.

The answers that gather the most are chosen, where a change between "there" and "not there" costs as much as
three clear answers (a copy that is not seen for a second and a half is still there; several seconds without one
are left alone). The inverse filter then works only in the stretches that have the copy, faded in and out over
20 ms (`gateOf`), and a line with no such stretch is refused. Why the smoothed line: in a clean stretch the
tracked line wanders from one thing in the picture to the next (1 to 5 samples off its own smoothed course), a
copy's line is smooth (0.1 to 0.8 samples), so what a made-up line collects is lost when it is smoothed.

Put onto another sound, the lines of the stretches that have the copy run on as one line: through a piece that
shows nothing and through a stretch without a copy the neighbours' lines carry on, as their periodic curve or
back and forth (`joinedLine`), with the pieces' middle gain and feedback, over the WHOLE of the other sound. A
piece that shows nothing may still have the flanger (on long sounds it is often simply not found in a piece), so
"not found here" is no reason to leave the other sound dry there. The price: a flanger that really is on a part
of the first sound only goes onto all of the second.

Three short cuts make the analysis 2.4 times as fast and change no number (the results are the same bit for
bit): while gain and feedback are corrected the picture is only made where the line reads it (`picture` with a
line); the running median behind the moving part of the picture keeps its window sorted (`staticPart`); the two
channels of a piece are measured at the same time.

`flong.py` makes the test sounds (dry corpus clips laid end to end, 33 to 62 s, through the four outside
flangers of the scoreboard) and runs `drsweep both` on them; `fone.py` runs single cases and `fmk.py` writes a
case's sounds to disk.

Numbers at the default Sureness, 24 long sounds a set (33 to 62 s, six through each of the four flangers;
100 = the dry sound again, 0 = as if nothing had been done):

| set                     | FLANGER shown | found, but only at a lower Sureness | taken out, where shown | onto a second sound, where shown |
|-------------------------|---------------|-------------------------------------|------------------------|----------------------------------|
| clean                   | 0 of 24       | 0                                   |                        |                                  |
| flanged from end to end | 3 of 24       | 3 more (out 18, 33, 34)             | 93, 85, 30             | 88, 79, 17                       |
| flanged in one half     | 4 of 24       | 4 more (out 31, 20, -8, -8)         | 65, 75, 63, -5         | see below                        |

- None of the sounds flanged from end to end comes out worse than it went in, at any Sureness.
- On long sounds the flanger is found far less often than on sounds of 6 s (4 of 8 there): a piece of a sound
  flanged from end to end shows the copy in 16 % of the cases (33 % for the flanger with the loudest copy, 17 and
  16 % for the two lighter ones, none for the thru-zero one), and a find mostly rests on one or two pieces.
  Guiding a piece by its neighbours' lines is the next step here.
- Half-flanged: the clean half comes back as it went in on seven of the eight sounds with a find (the difference
  to the input is at -172 to -186 dB) and is at -32 dB on the eighth, where the gate opens inside the 50 ms over
  which the test sound fades its flanger in. Without the test of where the copy is, two of these sounds had their
  clean half changed (the difference at -10 and -21 dB) and one came out at -23.
- Onto a second sound the flanger goes over all of it (see above), so against a second sound that has the flanger
  on the same half only these four cannot score (-6, 19, -131, -13). Inside the flanged half the 6 s blocks of
  the two best are at 46 to 84 and 40 to 75.
- What the test costs on sounds flanged from end to end, where it should leave nothing out: the same six sounds
  are found with and without it; two lose 1 to 2 of 100 (86 to 85 and 32 to 30 taken out), one gains 5 (13 to
  18), three are the same.
- Analysis of a long clean sound takes 0.14 s for every second of sound (0.12 to 0.16 on eight sounds of 20 to
  62 s; four cores, an idle machine, both channels at once).

## Known limits
- Light flangers (copy under about 0.35 of the sound) on busy sounds are found on some sounds and not on others.
- Thru-zero flangers and flangers whose delay stays under 1.3 ms are not found.
- One copy per channel; a flanger with a wandering (not periodic) movement is taken out as tracked, and goes onto
  another sound as tracked, without a curve to continue it.
- Analysis takes 1.2 to 1.9 s for a 6 s sound that has a line to follow up, 0.75 to 1.1 s for one that has none
  (four cores, an idle machine; the eight flanged sounds of the scoreboard and four clean ones).

## Files
- `sweep.py`: the prototype. `lab.py`: helpers for loading cases and plotting (not part of the engine).
- `stats.py`, `rule.py`, `levelb.py`, `final_stats.py`, `recheck.py`: the rule's numbers on clean sounds and
  synthetic flangers.
- `ev.py`, `ev_or.py`, `oracle.py`, `ceil_eq.py`, `shift.py`: outside effects, the true moving comb, ceilings.
- `flong.py`, `fone.py`, `fmk.py`: long sounds through `drsweep`.
