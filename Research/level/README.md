# The level-movement engine: Python side

`level.py` is the prototype that the single-sound part of `Source/Core/Level.h` is a port of; `rule.json` holds
the fitted rule. `Tools/level_parity.py` compares the two on whole sounds (48 sounds, 36 of them
with a movement: the same answer on all, the measures agree to 1.4e-9, the scores to 1e-11, the shapes to 2e-10 dB
and the rendered sounds to -217 dB). What `Level.h` does with
long sounds (pieces, the clock, where the movement is) was written in C++ only and is tested from here through
`drlevel` (`long.py`, `win.py`). The scripts are a record of how the engine was built and judged, as they were
run: paths point to the working folder they ran in, and the audio is not in the repo.

## The method
Tremolo, auto pan and a gated rhythm are one thing: a gain per channel that moves over time and repeats. The engine
measures that gain on the one sound (period, shape over one cycle, depth, how left and right differ) and uses it
both ways: divided out of the sound it was found on, or multiplied onto another sound.

How a gain that was put on the sound is told from the sound's own rhythm: a gain moves ALL bands by the same
amount in dB at the same moment, and it repeats exactly. Notes and hits do neither.

1. `levels` power in 15 bands per channel, one frame every millisecond (5.3 ms windows).
2. Both sides together (tremolo, gate): `diff_lane` is what all bands do together over a short step (the median
   over bands of the level change in dB, so the sound's own level, which differs from band to band, drops out).
   `scan_together` asks for every period from 60 ms up how much of that lane repeats with it (epoch folding), and
   `peaks_of` keeps the periods that stand out.
3. `measure_together` everything about one candidate period: the shape over a cycle from level differences between
   cells of the cycle (`pairs_fold`), its depth, and whether all bands and all cycles carry it (`agreement`: how
   well each band follows the shape, the slope of each band against it, the size of each cycle, the two halves of
   the sound against each other), how steep it is, whether it looks like struck notes (`struck_numbers`).
4. Left against right (auto pan): `pan_lane` is left minus right in dB, where the sound itself cancels;
   `find_pan` measures its swing, period and shape, and `fit_law` which law the two sides follow (the gains add up
   to a constant, or the powers do). A pan slower than half the sound is fitted as one slow curve (`slow_fit`).
5. A gate: steep steps are looked at finely (`sharpen`: power in steps of 0.125 ms laid over all cycles; the
   period that makes the steps sharpest, each step's place and course), and steps the gate leaves out are found
   (`left_out`, `gate_steps`).
6. `apply` takes the gain out (never lifting by more than 40 dB, nor more than the bands say the gate really
   closes, `band_depths`) or puts it on, scaled in dB for everything between and beyond.

## Is it a movement? (the rule)
A steady rhythm repeats too. Two small linear rules decide (`rule.json`, written by `fit4.py` for the shared
movement and `fit5.py` for the pan; the same numbers are constants in `Level.h`):
- together: 33 measures of step 3 after a coarse first filter (`first_filter`) and never for a shape that jumps
  up and sinks back slowly, which is what struck and plucked notes do in every bar of music (`struck`);
- against: 9 measures (swing, how alike the bands swing, what is left after the shape, the steepness...).

Two things were added to the pan rule after the fit, from what the app's scoreboard showed (they are in `score_pan`
and in `Level.h` alike):
- the repeating shape must explain the lane: at least 0.2 over what chance explains (`adj`). Every true pan of the
  fit data explains 0.3 and more. A stereo delay whose echoes fall on alternating sides swings left against right
  by tens of dB and repeats nothing (-0.2), and that large swing alone had carried it over the cut;
- a movement from side to side that the sound shows once only (less than 1.25 cycles: no part of it is seen coming
  back) scores no higher than a sureness of one half, which the default Sureness does not show. Something that
  passes by moves from side to side too: 6 of the corpus's 40 recordings do, by 6 to 35 dB. On the dry recordings
  the bands do not swing alike and the rule refuses them, but an effect that makes the bands alike (a bitcrusher
  did) left nothing to tell the two apart.

The look in long windows (below) asks for the first of the two afterwards. A pan that is on a part of a window only
explains less of the whole window (0.13 for one on the first half of a minute), so what that look finds is a hint:
it stands when it explains 0.2 after all, or when the stretch that has it, measured by itself, passes the whole
rule; otherwise it is no find. Asked straight away, the least share had cost a slow pan on one half of a sound
(taken out at 95 before and after).

Fitted on 527 clean 6 s clips and the same clips with one synthetic movement put on (`synth.py`, `mkdata.py`,
`mkstage2.py`, `mkpan.py`), plus the 115 dry sounds of the scoreboard corpus's dev split (`mkdev.py`); the cut
is set on out-of-fold scores so that 0.4 % (together) and 0.2 % (pan) of clean clips have a candidate over it.
The score's zero is that cut, and the app's Sureness slider compares it: the default, 65 %, shows a movement from
score 0 up; lower settings offer movements down to score -2.

What the rule lets through (`final_rates.py`, `clean.py`):

| sounds of 6 s                                                    | reported |
|------------------------------------------------------------------|----------|
| clean train clips (shared movement / pan)                        | 0 of 527 / 0 of 529 |
| dry corpus sounds, dev / test split (test never used in the fit) | 0 of 115 / 0 of 83  |
| synthetic gates on train clips, found with the right period      | 123 of 174 (71 %)   |
| synthetic tremolos on train clips                                | 74 of 183 (40 %)    |
| synthetic pans on train clips (0.15 to 3 Hz)                     | 412 of 529 (78 %); 47 more, seen once only, at a Sureness of 50 % or lower |

The tremolos it misses are the shallow ones and the sawtooth that falls (0 of 28: a level that jumps up and sinks
is exactly what the `struck` test refuses). A wrong period was chosen for 4 of 357 movements.

## Outside effects (the scoreboard's kind of case)
`ev.py` runs the prototype on cases made with effects from outside tools (four tremolos, four pans, three gates;
two sounds per case, test split): found 11 of 16 tremolo sounds, 15 of 16 pans, 10 of 16 gates. Over all 16
sounds of a family (a sound that is not found counts 0) the movement comes out at 92 (tremolo) and 94 (pan) of
100 in the median and goes onto the case's other sound at 93 and 97; no result is worse than doing nothing.
Gates are the hard family: out 22, onto 79.

`others.py` asks the prototype on the sounds of the other families. Run before the two additions to the pan
rule, it reported a movement on 5 of 224 sounds with another effect (two with a ring modulator at 16 Hz, which is
a level movement; a stereo delay and a bitcrushed recording, which the additions now refuse; a filtered bass
loop) and on 2 of 248 sounds of the reverb and tone sets (one sound, dry and wet).

In the app (`Research/room/bench/famboard.py`, 8 cases per family, medians, a case that is not shown counts 0):

| 8 cases each | shown (0.4.2) | taken out, 0.4.2 to 0.4.3 | onto the other sound | taken out where shown      |
|--------------|---------------|---------------------------|----------------------|----------------------------|
| tremolo      | 5 (5)         | 41 to 85                  | 44 to 78             | 95, 84, 95, 96, 86         |
| auto pan     | 8 (8)         | 64 to 96                  | 71 to 98             | 99, 92, 95, 81, 88, 96, 97, 98 |
| gated rhythm | 6 (6)         | 0 to 60                   | -18 to 79            | 89, 92, 88, 45, 0, 76      |

No case comes out worse than it went in (0.4.2: gated rhythm put onto the other sound made all six worse). On the
198 dry sounds of the corpus the app shows none of the three groups. On the 112 cases of the other families it
shows one of them once: TREMOLO, at a sureness of 0.66, on a bass loop with a low shelf (taking it out changes
next to nothing). The detector of 0.4.2 had an AUTO PAN on a phaser there and, inside the family, a GATED RHYTHM
on a tremolo. One tremolo that 0.4.2 showed (taken out at 24) is under the default Sureness now.

## Long sounds (`Level.h` only)
The rule was fitted and tested on sounds of 6 to 9 s. A longer sound is cut into pieces of 6 to 9 s:
1. each piece is measured alone, as tested;
2. every piece is measured again at the periods the others found (same measures, same rule, the period given);
   the pieces that show the movement at its period carry it. The movement's score is the middle one of the
   carrying pieces' scores (with more pieces the best score of a clean sound only goes up), less 1 when just one
   of several pieces carries it;
3. a pan too slow for such pieces is looked for in windows of 36 to 72 s. What such a window finds is asked again at
   a half and a third of its period: a pan three times as fast looks like a slow one with a strong third harmonic,
   and three cycles laid over each other also hold the sound's own wander from side to side. A fraction is only
   asked when it still explains at least 0.4 of what the found period explains (folded at a wrong fraction of its
   period a movement all but disappears, and the rule's score means nothing for a period that does not explain
   the lane);
4. the clock: where in the surest piece's cycle every place of the sound stands. One steady period if that puts
   all carrying pieces within 0.03 cycles (an effect's own oscillator), else from piece to piece (a speed that
   wanders);
5. where the movement is: every stretch of half a second or more (a whole cycle; of a pan half a cycle) is asked
   whether the sound does there what would be taken out. What the stretch does is set against what the movement
   would do there and against no movement at all (sums of absolute differences): +1 = just what the movement
   says, -1 = none of it, 0 = cannot be told (silence, or the sound's own changes far larger). The answers that
   gather the most evidence are chosen, where every change between "there" and "not there" has a price, and each
   change is then placed to the millisecond from the same evidence frame by frame. Only the stretches that have
   the movement are touched. If one window found the movement and it turns out to be on a part of the window only,
   that part is measured again by itself: the movement is surer and truer there;
6. a gate's left-out steps: where eight or more cycles in a row are without the gate's steps and as loud as their
   neighbours' open stretch, the gate is not there (nothing to lift), and they count against the movement in 5. A
   pattern's own left-out steps stay closed.

Taken out, a piece that carries the movement by the rule loses its own measured movement, any other stretch that
has it the surest piece's movement on the clock. A carrying piece is asked with both and loses the surest piece's
movement instead of its own when it follows that clearly better, or, for a gate, when that takes clearly more of
the piece's level steps away (8 % and more): a gate's own measurement can be wrong in a way the stretches' answers
do not show, because steps it takes for left out predict nothing and so cost nothing. Put onto another sound, the surest piece's shape runs on the first
sound's clock at the times at which the first sound has the movement: an effect that comes in halfway comes in
halfway on the second sound too (past the first sound's end it goes on as at its end).

`long.py` makes the test sounds (dry corpus clips laid end to end, crossfaded, 20 to 62 s; the effects of
`../detect/zoo_fx.py`) and runs `drlevel` on them:
- `clean`: 48 long sounds without any effect;
- `wet`: 88 with one of 11 outside effects on the whole sound;
- `part`: 44 with the effect on one half only (the other half must come back untouched);
- `drift`: 16 with a speed that wanders; `slow`: 24 with slow pans;
- `SEED=1` draws a second set, kept for the final check.
`win.py` reads a result file: what single stretches say with and without the effect, how much of each is marked,
where the border was put, and what other prices would have chosen.

Numbers of the final code at the default Sureness (100 = the dry sound again, 0 = as if nothing had been done;
"median": a sound that is not shown counts 0):

| set                          | sounds | shown | taken out: median, where shown | onto a second sound | worse than nothing |
|------------------------------|--------|-------|--------------------------------|---------------------|--------------------|
| clean                        | 48     | 0     |                                |                     |                    |
| whole sound: tremolo         | 32     | 27    | 88, 89                         | 92, 93              | 1 (-4)             |
| whole sound: auto pan        | 32     | 31    | 94, 94                         | 97, 97              | 0                  |
| whole sound: gate            | 24     | 23    | 71, 71                         | 81, 82              | 0                  |
| one half: tremolo            | 16     | 9     | 78, 83                         | 82, 90              | 0                  |
| one half: auto pan           | 16     | 16    | 92, 92                         | 93, 93              | 0                  |
| one half: gate               | 12     | 11    | 67, 69                         | 75, 76              | 0                  |
| wandering speed: tremolo     | 8      | 5     | 37, 67                         | 4, 19               | 1 (onto)           |
| wandering speed: auto pan    | 8      | 8     | 89, 89                         | 84, 84              | 0                  |
| slow pans (3 to 23 s)        | 24     | 21    | 94, 94                         | 94, 95              | 0                  |

- Clean: two of the 48 are offered at the lowest Sureness (a "pan" of 3 dB once in 23 to 29 s).
- Whole sound, stretch by stretch: of the 6 s stretches of the sounds that are shown, 9 % (tremolo), 2 % (auto
  pan) and 19 % (gate) come out under 50.
- One half: the half without the effect comes back unchanged in 31 of the 36 sounds that are shown (the difference
  to the input is at -172 to -200 dB); in five the border sits a little off and it is at -73 to -35 dB.
- Slow pans: the three that are not shown pass 1.2, 1.2 and 0.9 times in their sound and are offered at a
  sureness of one half (taken out, they come out at 93, 87 and 89).
- What the late changes did, case by case against the run before them: the changes to the pan rule (the fraction
  rule of step 3, the least share) brought back a fast pan the long look had lost (0 to 92) and lifted six more
  pans (by 1 to 20); the gate choice lifted three gates (by 4 to 8); the pan seen once took the three slow pans
  above from shown to offered; every other case is the same.
TODO-S1
- Time: TODO-TIME

## Known limits
- A tremolo of a few dB on a busy sound, and any sawtooth that falls, is not found at the default Sureness.
- One movement per sound: a tremolo and a pan together give the surer one.
- Movements faster than 16.7 Hz are not looked for (up there a sound's own roughness and beating take over).
- A gate that is keyed by another sound has no fixed shape; what is found is its period and average shape.
- A movement from side to side that the sound shows less than 1.25 times is never shown at the default Sureness
  (a pan of 16 s on a sound of 20 s).
- A tremolo on a part of a long sound is found less often (9 of 16) than one on all of it (27 of 32).
- A speed that wanders inside a piece: the clock is steady within a piece, so such a tremolo is taken out less
  well (67 where shown) and goes onto another sound poorly (19).
- A slow pan on a stretch shorter than four of its cycles cannot be measured by itself, so on a part of a long
  sound it is not found.
- Echoes that fall on alternating sides in a slow, even rhythm can still read as a pan (1 of 15 synthetic
  ping-pong delays), and on the scoreboard a bass loop with a low shelf shows TREMOLO just over the default.

## Files
- `level.py`, `rule.json`: the prototype and its rule. `synth.py`: the synthetic movements of the fit data.
- `mkdata.py`, `mkstage2.py`, `mkpan.py`, `mkdev.py`: the fit data. `fit4.py`, `fit5.py`: the two rules
  (`fit1.py` to `fit3.py`, `fit4b.py`: earlier tries; `fit6.py`: a rule of hard limits tried against the fitted
  sum). `final_rates.py`, `clean.py`: what the final rule lets through.
- `ev.py`: outside effects on 6 s sounds. `others.py`: what the engine says on sounds with other effects.
- `long.py`, `mk_case.py`, `win.py`: long sounds through `drlevel`.
