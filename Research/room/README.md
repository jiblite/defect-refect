# The reverb engine: Python side

The scripts the V2 reverb engine was built with, as they were run. They are a record, not a package: paths point
to the working folder they ran in (`/tmp/claude-0/v2/...`), and the audio (test sets, dry clips, training cases) is
not in the repo. The C++ in `Source/Core/Room*.h` is a port of `train/engine.py` and agrees with it to about
-150 dB (`Tools/room_parity.py`).

## The chain
1. `train/common.py` what the model sees of a sound: 41 bands, per band and frame the level (LE) and how much of
   the power left and right do not share, at two averaging times (U64, U256).
2. `train/net.py` the model (a small U-Net, 231k weights): per band and frame, how far the cell is room (0 .. -30 dB).
   `train/predict.py` runs it; `'first+second'` means two looks: the second model also sees the room read from the
   first answer (`roomnum.second_look`).
3. `train/roomnum.py` the room in numbers from that answer: decay time and strength per band, with caps.
4. `train/engine.py` what the app does with it: `fader` (Amount on the sound itself), `one_sound` (Length,
   Brightness, Width reshape the sound's own reverb), `onto` (the found room on a second sound), `room_ir` (Export IR).

## Scoreboard (`bench/`, `proto/truth.py`)
`bench/mkbench.py` makes the test sets with effects that are not this project's code (`bench/zoo*.py`): 120 reverb
cases and 64 tone cases, each with two sounds. `proto/truth.py` gives the exact direct part of each wet sound.
`bench/score.py` scores a result against the truth: 100 = identical, 0 = no better than leaving the sound alone.
`train/final_eval.py` scores the whole engine (`ORACLE=1`: with the right room-or-sound answer).
`bench/famboard.py` runs the app itself (`drrender`) on the cases of the 17 other effect families and scores what it
takes out and what it puts on the second sound, for the effect's own group and for everything the app shows.

## Training (`train/`)
`mkclips.py` dry clips, `mkcases.py` training cases (a dry clip, a reverb, the exact direct part),
`mksb.py` the scoreboard sounds as the model sees them, `cnntrain.py` the training, `mkpass2.py` the data for the
second look. Model in the app in 0.4.0: `unet_b9` (first look) + `unet_p2` (second look).
Model in the app since 0.4.1: `unet_w` + `unet_wp2`, the same two fine-tuned with `cnntrain2.py` on the first
material plus `mkcases_w.py` (the whole sound narrowed or mono) and `mkcases_s.py` (wide sounds under the reverb,
a quarter of them without reverb); `mksb_w.py` makes the narrowed and mono scoreboard sounds. Against the first
pair (`cmp_final.py`, `sbx_eval.py`, `narrow_test.py`): reverb out of the 40 full test cases 23.2 / 22.8 (median /
mean) against 19.7 / 20.1, the same share made worse (12 %), room onto a second sound level (19.4 / 21.7 against
18.6 / 23.2); band score on the scoreboard sounds at 0.3 of their width 38 to 43 against 19 to 21, in mono 36 to
40 against 14 to 17; narrowed down to mono the room is still found on 9 of 10 sounds (first pair: about 7).

## "Is there a room at all"
`sure_stats.py` collects what the engine says on sounds with and without reverb, `sure_fit2.py` fits the rule in
`Estimate.h` (`roomSureness`), `sure_cpp.py` checks the app's own verdicts on sounds the rule was not fitted on,
`own_stats.py` and `narrow_test.py` show what happens when left and right share nearly everything.

`bench/cut_test.py` is the test behind the look with silence of 0.4.2 to 0.4.4 (`recheckRoomEnd`): the verdict on
sounds as files, cut off at 6 s, and with 3 s of silence after them. The models of that time had learned from
sounds that end inside the file. Cut off while still playing, a room showed on 54 of 120 dry sounds (on 18 with
silence after them) and on 90 of the corpus's 198 dry clips (29). The app looked a second time with silence after
the sound and the less sure verdict counted: 18 and 29. Sounds with a natural ending were unchanged; cut sounds
with a real reverb paid, 68 of 120 shown instead of 88. 0.4.5 trains the model on such sounds instead (next
section) and that look is gone.

## 0.4.5: files that stop while the sound plays, excerpts, other effects, long sounds
**What was wrong** (measured on build 0.4.4, `bench/look1.py`, `look2.py`, `train/cut_proxy.py`, `cut_eval.py`).
- Of the 19 scoreboard cases on which the app showed a REVERB that is not the case's effect, 12 show it on the
  clean clip under the effect as well (recorded loops, plucked-string and steel-drum samples, recordings). 7 were
  made by the effect. On 912 sounds with one of the other 17 effects alone (`bench/prep_nw.py`, `prep_sb.py`)
  REVERB showed on 14.7 %: 86 sit on a clip that shows it clean, 48 are made by the effect.
- The bigger fault: every training case ended inside the file. On a file that stops while the sound plays, and on
  an excerpt cut at both ends, the model's answer for the last second or two is wrong (4.5 dB away from its answer
  on the whole file), and the engine did as much harm as good:

| 40 test reverbs, the engine alone (`cut_eval.py`): reverb out, median / worse than untouched | 0.4.4 pair | 0.4.5 pair |
|---|---|---|
| rings out, whole file                         | 23.2 / 12 % | 25.2 / 8 %  |
| cut off at 6 s                                | 1.5 / 48 %  | 11.4 / 18 % |
| excerpt, 1.0 to 5.5 s                         | 9.5 / 45 %  | 11.3 / 15 % |
| cut off, Amount at far left                   | -18.8 / 68 % | 4.6 / 48 % |
| excerpt, Amount at far left                   | -12.6 / 65 % | 1.4 / 45 % |

**The repair.** Both looks are fine-tuned from the 0.4.1 pair (`train/cnntrain3.py`, which takes pictures of
several lengths; `run_x1.sh`, `run_x2.sh`; six epochs each, about 70 minutes on four cores) on a fixed 60 % of the
earlier material plus `train/mkcases_x.py`: 3615 renders, each stored whole (the clip and 3 s after it), cut off at
6 s, and as a 4.5 s excerpt from inside. 55 % of them have one of the 33 'dev' tools of the other effect families
on the sound first (the other 31 tools stay unseen), and 45 % have no reverb at all, where the right answer is "all
sound". `mksb_c.py` and `mksb_fx.py` make the scoreboard's cut, excerpt, clean and effect sets as the model sees
them. Model in the app since 0.4.5: `unet_x` + `unet_xp2`.

Band score (median, dev / unseen), 0.4.4 pair -> 0.4.5 pair: ringing 44.3 / 50.1 -> 44.6 / 54.2; cut off at 6 s,
first look 15.1 / 12.0 -> 24.9 / 23.9, both looks 29.6 / 24.7; excerpt, first look 2.5 / 4.6 -> 23.1 / 19.8, both
looks 22.9 / 23.4. The standard scoreboard (`final_eval.py 3`): reverb out 23.2 / 22.8 / 12 % (median / mean /
worse) -> 25.2 / 26.4 / 8 %; half way 15.8 -> 17.3; far left 13.8 / 1.3 / 38 % -> 11.0 / 7.0 / 35 %; doubled 21.5 ->
21.8; room onto a second sound 19.4 / 21.7 -> 18.3 / 24.7; strength off by 3.5 -> 2.8 dB (median); decay time
within 20 % on 64 % -> 74 % of the cases; light reverbs, mean 11.6 -> 18.9 and worse on 38 % -> 23 %.

**The app itself** (`train/cut_app.py`: `drrender` at the default Sureness on the same 40 cases; a case without a
REVERB group is left alone and counts as 0):

| REVERB shown / reverb out, median and mean / worse than untouched | 0.4.4 | 0.4.5 |
|---|---|---|
| rings out, whole file | 82 % / 23.2, 23.2 / 2 % | 90 % / 25.2, 26.6 / 2 % |
| cut off at 6 s        | 48 % / 0.0, 8.8 / 8 %   | 68 % / 11.4, 17.4 / 5 % |
| excerpt               | 45 % / 0.0, 6.1 / 8 %   | 60 % / 4.7, 13.7 / 5 %  |

0.4.4 kept the harm of the first table off the screen by not showing the group on half of the cut-off reverbs
(the look with silence); 0.4.5 shows it more often and what it then does is better (where shown, median 24 against
17 to 18). With Amount at far left a fifth to a quarter of the cases still come out worse than untouched, in both
builds.

**The rule** (`bench/rsure.cpp`, `fit_sure.py`). `roomSureness` is fitted again on 2470 looks: every reverb tool on a
clean clip and after another effect, ringing out and cut off at 6 s, against clean clips, the other effects alone
and tone effects alone; fitted on the tools and clips training used, checked on the others. Unseen half, at the
slider's 65 %: reverb shown on 95 % of the ringing sounds and 71 % of the cut-off ones (0.4.4: 93 % and 60 %), on
11 % of the sounds without (0.4.4: 14 %). What the number says holds (65 to 80 % sure: 74 % have a
reverb; 80 to 90: 83 %; over 90: 96 %). The look with silence after the sound adds nothing for this pair: on the
false cases it gives the same verdict as the first look, and as "the less sure counts" it costs the cut-off
reverbs 7 points for 1 point fewer false ones.

**The look without the other effects** (`Estimate.h`, `recheckRoomWithout`; `bench/recheck_eval.py`). The new
training lowered the false REVERB that sits on the clip itself (clean clips shown: 29 -> 20 of 198) but not the
kind an effect makes (48 -> 50 of 912). So a found room must still be there with the other found effects taken
out: echoes, level movement, flanger, phaser, filter sweep and chorus by their engines, a Haas delay by moving the
late side back (for echoes alone this existed since 0.4.2). On the 912 sounds with an effect alone that takes the
effect-made false REVERB from 50 to 26 (Haas 12 -> 4 shown, delay 15 -> 8, auto pan 6 -> 3, chorus 8 -> 6, flanger
9 -> 7); of 340 sounds with a real reverb that the engine shows, 78 are looked at again and one loses its group.
All sounds without reverb, unseen half: 11.4 % -> 8.7 % (effect alone 13.2 % -> 9.6 %). What is left is mostly on
the clip itself (67 of 93) and the effects the app cannot take out yet (widening, distortion, ring modulation).
On the 40 test reverbs of the table above the look changes nothing (one sureness moves from 0.704 to 0.695). It
adds about a tenth to the analysis time of a sound on which a room and one of those effects are both found. In the
app itself (under Xvfb): a delay with a reverb shows DELAY and REVERB, and the scoreboard's Haas case that showed
REVERB in 0.4.4 shows none.

**The scoreboard** (`bench/famboard.py`: the app on the 136 cases with one outside effect each; `bench/famsum.py`
prints these lines), 0.4.4 -> 0.4.5: the case's own effect is shown on 89 -> 89 and every own-group score is as it
was. REVERB shown where it is not the case's effect: 19 -> 13. The new pair alone gives 16 (nine gone, six new);
the look without the other effects then takes away three (a flanger and a band-pass sweep that the new pair had
added, and a Haas delay). Other false groups 7 -> 7; cases with no other group 112 -> 116; every shown group at far
left makes 15 % -> 9 % of the cases worse than untouched; everything shown put onto a second sound 27 % -> 24 %.
Of the 13 left, ten sit on a clip that shows REVERB with no effect on it. Three are made by the effect: a chorus
that comes out only in part (the room is still read at 0.75 after it), a delay whose echoes are taken out and the
room is still read (0.73), and a delay whose echo the echo engine does not find.

**Long sounds** (`Estimate.h`, `roomInPieces`, `roomVerdict`; `bench/prep_long.py`, `long_eval.py`, `long_eval2.py`,
`long_eval3.py`, `long_app.py`). The model has learned from sounds of 4.5 to 9 s and the rule is fitted on sounds of
6 and 9 s; most files are longer. A test set of 134 sounds of 18 to 48 s (a clean clip played three and six times
over; twelve rows of five to eight different clips, clean and with a sweep, a phaser or a chorus; a reverb tool on
such loops and rows, ringing out and cut off; 265 looks in all) shows what the look at the whole sound does there:
REVERB on 40 % of the 121 looks without a reverb (on 10 of the 12 clean rows of clips; build 0.4.4 shows it on 9 of
those 12), where the looped clips show it on 10 % when each is looked at alone. It is not stray evidence adding up
with length
(asking that a band's eight all-room moments come within 9 s of each other changes one verdict in 265): the model's own
answer changes. On a clean clip looped to 36 s it finds a room on 15 of 40, on the clip itself on 7. The second
look is told the room numbers of the whole sound, and with a room that is believed to be there somewhere it finds
some of it in every stretch.

Three repairs were measured (`long_eval3.py` prints this table, REVERB shown at the slider's 65 %; `long_eval.py`
and `long_eval2.py` have more ways of taking the stretches):

| | without reverb (121) | with reverb (144) | of them ringing out (53) |
|---|---|---|---|
| the rule on the whole sound (up to now) | 40 % | 76 % | 85 % |
| the whole sound's evidence by 6 s stretches: the average | 10 % | 66 % | 66 % |
| the same: the larger of the average and the last stretch | 22 % | 74 % | 81 % |
| pieces of 6 s, each looked at as a sound of its own: the average | 5 % | 69 % | 68 % |
| the same: the larger of the average and the last piece (**the app**) | 17 % | 74 % | 81 % |

A piece looked at alone is what the model learned and what the rule was fitted on, and it shows: a clean clip
looped gives the verdict of the clip itself on 38 of 40, and shows REVERB on 10 % of 80 looks instead of 24 %. The
average over the pieces asks whether the room shows along the sound. The last piece asks whether the sound rings
out at its end, which is all that a reverb on dense material may give away (three of the looped reverbs read as
no room in every piece but the last, which reads 94 to 96 %). Its price is a row of clips that ends on a clip that reads
as a room by itself: 11 of the 41 rows are shown for that reason (rows of clips in all: 73 % -> 32 %), the same
error the app makes on that clip alone. Cut-off reverbs: 71 % -> 70 %. What the app's number says on these 265 looks:
65 to 80 % sure, 10 of 18 have a reverb; 80 to 90, 15 of 21; over 90, 82 of 89.
The app as a whole on 126 of these looks (`bench/long_app.py`: the analysis as the app runs it, the look without the
other found effects included) shows REVERB on 3 of the 12 clean rows, 10 of the 29 rows with an effect, 2 of 17 clean
loops, 43 of the 53 reverbs that ring out and 13 of 15 cut-off rows, as the study tool says; the look without the
other effects runs on 25 of them and changes no verdict. In the app itself a clip looped to 39 s with a plate reverb
shows REVERB at 94 %, and the 48 s clean row that build 0.4.4 showed at 78 % shows no effect.
The pieces: 6 s long, a new one every 3 s at most, at most 24 spread evenly, silence before and after the sound left
out, each weighted by how much of it sounds; the last piece is asked only when it holds a fifth of what the fullest
holds. No room on the whole sound: nothing is shown, as before. The pieces take about twice as long as the look at
the whole sound, a fifth more analysis time on a long sound. Sounds of up to 9 s are judged exactly as before:
`bench/idcheck.py`, 72 scoreboard sounds through the render tool of the code before and after give the same text
and the same output file.

**How well a reverb comes out of a long sound** (`bench/prep_long_truth.py`, `long_score.py`, `long_once.py`,
`long_parts.py`). Scored for the first time, in the way `train/cut_app.py` scores sounds of 9 s: the 53 long sounds
with a reverb that ring out, the true direct part from the tool's click responses, the app itself with REVERB Amount
half way left and at far left (100 = the true direct part, 0 = untouched; not shown counts 0).

| | shown | half way left: median where shown | worse than untouched | far left: median where shown | worse than untouched |
|---|---|---|---|---|---|
| the 40 test reverbs of 9 s | 36 | 26 | 1 | 18 | 11 |
| of them direct sound under 5 dB over the reverb (24) | 24 | 30 | 0 | 20 | 6 |
| of them direct sound 5 dB and more over the reverb (16) | 12 | 25 | 1 | 8 | 5 |
| the 53 long sounds (33 to 51 s) | 43 | 22 | 6 | -3 | 23 |
| of them direct sound under 5 dB over the reverb (21) | 20 | 24 | 0 | 7 | 6 |
| of them direct sound 5 dB and more over the reverb (32) | 23 | 20 | 6 | -11 | 17 |

A heavy reverb comes out of a long sound about as it does out of a short one. A light one does not: all six long
sounds that come out worse than untouched at half way left are light reverbs (down to -64, and -244 at far left),
and at far left more than half of the light ones are worse. Length is not the main reason. The same clip with the
same reverb played once instead of six times over (`long_once.py`, 38 pairs, 9 s against 39 s) scores 32 where shown
against 22, and is worse than untouched on 2 against 5 (far left: 10 against 17); where both are shown the looped
sound scores 7 lower (the median of the pairs). Stretch by stretch (`long_parts.py`, six of the pairs) the engine does much
the same to the sound itself either way, and what differs is how much the ring-out weighs:

| half way left | the clip played once | each of the six times through | the ring-out after it |
|---|---|---|---|
| a heavy reverb (direct 7 dB under it) | 36 | 34 to 35 | 59 to 60 |
| a light reverb the engine handles (direct 12 dB over it) | 50 | 43 to 53 | 77 to 80 |
| light, on a clip that reads as a room by itself (6 dB over) | -30 | -32 to -42 | 59 to 66 |
| light, on another such clip (6 dB over) | -26 | -14 to -38 | 52 |
| light (8 dB over) | -56 | -76 to -87 | 66 to 75 |
| light (11 dB over) | 2 | -13 to -29 | 70 to 77 |

The ring-out is where the engine gains most on every sound, and it is a third of a 9 s test sound and a thirteenth
of a looped one. So a score of a sound that rings out is carried in part by its last three seconds, and on a light
reverb the engine can make the sound itself worse while the whole file still scores above zero (the fifth row: 7 for
the 9 s file). On the last two rows the looped sound is also treated worse than the same six seconds played once (a
guess at why: played once, a third of what the room is read from is clean ring-out). Two of the six that come out worse sit on a clip
that reads as a room with no effect on it: the engine takes that ring out as well, and the truth keeps it.

Working in the same 6 s pieces as the verdict (each piece's answer, cross-faded where they meet) was tried on five of
the long reverbs and dropped: 28, -53, -13, 46 and -18 at half way left, against 36, -64, -16, 49 and -35 for the
whole sound. What is bad stays bad, so 0.4.5 works on the whole sound as before.

**Tried on the way and dropped.** A look at left and right summed to mono (a real room survives the sum; a room
read from the two sides differing does not): it takes away a quarter of the false REVERB on effect sounds, and
15 to 19 % of the real ones on cut-off files, so inside the rule it buys nothing. More evidence numbers in the rule
(share, strength, decay time, levels at the file's two ends): no gain on the unseen half.

**Left open.** On cut-off files and excerpts the model does not see the room on about a quarter of the real
reverbs (light ones most: found on 92 % under 0 dB direct-to-reverb, 66 % at 10 to 15 dB); where it does not, its
answer has no cell under -3 dB, so a better reader of decays would not help and a larger model is the next lever.
Sampled instruments are still read as rooms on 7 of the corpus's 24 clean melodies (cello, choir, horn, kalimba,
plucked strings, steel drum, vibraphone): samples that ring by themselves or carry the hall they were recorded in.
On long sounds only the verdict is repaired: the room that is worked with is still the one read off the whole
sound. A light reverb (direct sound 5 dB and more over it) is the engine's weak case at any length: it can take more
out of the sound itself than the reverb put in, and the scores of sounds that ring out hide part of that behind the
ring-out. Next for it: a score of the sound without its ring-out on the scoreboard, these cases in training, and a
depth that follows how much room was found. And the verdict on a short excerpt depends on where exactly it is cut: on one
looped clip with a light reverb, 6 s pieces that start within 50 ms of each other read between 22 and 97 %
(a look at a few slightly shifted copies, averaged, is the obvious thing to try).

## What was tried and did not help
- A plain physical rule as the answer (direct sound smeared with the room's decay): `look_test.py`.
- Rendering that treats the sound's left/right direction separately: `axis.py`, `axis_test.py`.
- Reading the best depth per sound from the answer's own statistics: `depth_test.py`, `bold_test.py`.
- A third look: `third_look.py`. A whole-sound part in the model (`GLOB`): level with the plain model.
- Everything in `proto/` came before the model: hand-made rules and what they taught.
