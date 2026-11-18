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

Build 0.4.4 on the same 53 long sounds (its own tool and models, `long_score.py`): REVERB shown on 46, half way left
19 where shown with 10 worse than untouched (down to -329), far left 1 where shown with 22 worse (down to -667).
So 0.4.5 is no step back on long sounds: better at half way left (22, 6 worse, down to -64), level at far left
(-3, 23 worse, down to -244). Of the four it no longer shows, three came out worse than untouched in 0.4.4.

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

## 0.4.6: the Amount fader, and the engine under the sound
**A score without the ring-out** (`train/depth2_test.py`). Every scoreboard clip is 6 s long and its file holds 3 s
more, in which the reverb rings out. All scores above are over the whole file. Scored separately (the same measure,
cell by cell; engine alone with the 0.4.5 pair, 120 cases; median / mean / worse than untouched), the answer gets
26.5 / 21.8 / 14 % over the whole file, **11.0 / 11.2 / 28 % over the sound itself** (the first 6 s) and 74.8 over
the ring-out. Most of what a whole-file score shows is won after the sound has stopped.

**The far left of the fader.** Up to 0.4.5 the answer was in full at half way left, and from there to far left it
was made bolder (`engine.deeper`: the doubtful cells move most). 28 ways of reshaping the answer, scored the same
way (b: the old bolder end, shift of "how sure"; k: plain scaling in dB; sharp, deep: only the cells the answer
already takes deep go deeper; c lo/hi: shallow cells scaled by lo, deep ones by hi):

| the answer reshaped | whole file | the sound itself | sound further from dry by 0.5 dB+ | ring-out |
|---|---|---|---|---|
| as it is (far left since 0.4.6) | 26.5 / 21.8 / 14 % | 11.0 / 11.2 / 28 % | 3 of 120 | 74.8 |
| b 1.5 (far left of 0.4.1 to 0.4.5) | 12.1 / -5.0 / 37 % | -7.1 / -22.3 / 56 % | 40 | 72.3 |
| b 0.5 | 25.3 / 16.9 / 19 % | 7.4 / 4.5 / 40 % | 11 | 77.9 |
| b -1 (gentler) | 19.9 / 20.4 / 9 % | 13.1 / 12.7 / 18 % | 1 | 53.4 |
| k 0.5 (half way left since 0.4.6) | 17.8 / 15.8 / 10 % | 12.7 / 11.2 / 19 % | 1 | 39.3 |
| k 0.75 | 24.0 / 20.3 / 13 % | 11.6 / 12.7 / 22 % | 2 | 59.6 |
| k 1.25 | 26.3 / 19.4 / 16 % | 9.2 / 7.2 / 34 % | 9 | 80.5 |
| sharp 1 | 26.7 / 20.2 / 16 % | 9.1 / 8.2 / 32 % | 8 | 80.3 |
| deep 1.5 | 27.2 / 20.8 / 17 % | 9.1 / 8.7 / 31 % | 9 | 81.5 |
| c 0.5 / 1.0 (from 6 to 18 dB) | 26.8 / 23.5 / 12 % | 12.7 / 13.6 / 21 % | | 70.4 |

Nothing bolder than the answer is better anywhere but in the ring-out (75 to 80 or 83), and all of it costs under
the sound. The old far left was the worst of the 28. Picking the best shape for every case by hand would give 32.6
(whole) and 16.2 (the sound itself): no shaping repairs the answer. The fourth column is the same in plain dB: on
how many of the 120 the sound itself ends up further from the true dry sound than it was, by 0.5 dB per cell or
more (the sound itself is 2.5 dB from its dry version at the median, 1.4 dB with a light reverb).

So since 0.4.6 the fader runs straight (in dB) from untouched in the middle to the answer in full at far left
(`Estimate.h` `roomFader`, `engine.fader`; the old one is kept as `engine.fader_bold` for these comparisons). The
new far left is the old half way left to the last bit (five sounds through both tools: -329 dB or better), so every
"half way left" figure above is the new far left. In the app, still with the model of 0.4.5 (`cut_app.py`,
`long_score.py`, `long_abs.py`; with the model of 0.4.6: "In the app" at the end of this section):

| Amount at far left: cases worse than untouched | 0.4.5 | 0.4.6 |
|---|---|---|
| 40 test reverbs: ringing out / cut off at 6 s / excerpt | 11 / 9 / 7 | 1 / 2 / 2 |
| 53 long reverbs | 23 | 6 |
| 53 long reverbs, further from the dry sound by 0.5 dB+ (by 1 dB+) | 8 (7) | 1 (0) |

**Worse than untouched by how much?** On a light reverb the sound is close to its dry version to begin with, so
"worse than untouched" can be a hair. Of the 34 cases whose sound itself comes out worse with the plain answer, 25
are further from the dry sound by less than 0.25 dB per cell, 6 by 0.25 to 0.5 dB and 3 by more (0.58, 0.74 and
1.68 dB). Those three sit on recordings that read as a room with no reverb on them (the model turns 47 to 75 % of
the dry recording's own cells down by 3 dB or more): the engine takes out the room the recording already had along
with the one that was added, and the truth keeps it. By the scoreboard's rule that is harm; for someone who wants
the recording dry it may be the point.

**Is it the training data?** (`train/body_diag.py`: the band score of the picture the model sees, sound itself /
ring-out, both looks.) No: the model is as weak under the sound on the pictures it was trained on.

| pictures | the sound itself | with a light reverb (direct 5 dB+ over) | ring-out |
|---|---|---|---|
| training: simulated rooms (354) | 36.9 | 18.9, 24 % worse | 92.6 |
| training: outside reverb tools (333) | 28.5 | 22.0, 22 % worse | 91.7 |
| training: scenes of several sounds (301) | 22.5 | 3.7, 40 % worse | 92.2 |
| training: with other effects (212) | 35.4 | 17.2, 33 % worse | 91.6 |
| scoreboard, tools also in training (176) | 31.8 | 12.9, 29 % worse | 90.2 |
| scoreboard, tools never seen (64) | 27.6 | 19.9, 25 % worse | 91.5 |

**How much is there to win?** (`train/body_ceiling.py`: answers that know the truth, rendered the engine's way, one
gain per band and 10.7 ms; as heard, the sound itself / whole file.) The right answer per cell: 48.1 / 61.8. From
the powers of the direct part and of the reverb part alone (no knowledge of how the two add up in a cell): 45.0 /
60.0. The same from powers known only to 32 ms: 42.2, to 96 ms: 33.5, to 288 ms: 16.2. The right answer only where
it is 6 dB or deeper: 41.0; 12 dB or deeper: 28.9. The model's answer kept from ever going deeper than the right
one: 27.8, and used only where the right one is 6 dB or deeper: 24.6. So under the sound this way of working tops
out at 48, the win is in cells that are mostly room for a moment (after a note, between two hits), they have to be
placed to a few frames, and half of the model's deficit is cells it turns down that belong to the sound.

**Where the answer loses** (`train/body_why.py`, points of the band score over the sound itself). All 120: 49 won,
32 lost (18 of them by going more than 3 dB too deep), 24 left in. Light reverbs: 46 won, 57 lost. Of those 57, 33
are cells taken more than 3 dB too deep, and 69 % of that is in cells the model also turns down when it is given
the dry sound alone (the sound's own ring, read as a room). The other 24 points are a haze: cells that should stay
and lose 1 to 3 dB. The loss explains the haze: its cross-entropy part is at its lowest at the mean of what a
doubtful cell could be, not at the most likely value.

**Tried and dropped: a cap by the found room** (`train/cap_test.py`). A room with a given decay and strength cannot
put more into a cell than the sound before it, smeared with that decay, times the strength, so the answer could be
kept from taking out more than that. Three such bounds (the app's own numbers; the wet sound smeared, strength
fitted on the free decays; the split run forward in time) follow the true reverb level under the sound well: off
by 0.2 to 1.8 dB at the median, 10th to 90th percentile within a sound about -4 to +5 dB (the app's own numbers:
-3 to +8). As a cap they buy
little: light reverbs go from 13.9 to 21 (band score of the sound itself, worse than untouched 27 % to 15 %) while
the ring-out falls from 91.5 to 60, because a bound that is 3 dB short in a free decay holds the answer back
there; with 6 dB of margin the ring-out is back and the gain is gone (16.1). Even the TRUE reverb level as the cap
loses most of its effect with 6 dB of margin (35.5 to 21.9): the harm is not in cells where the room cannot be,
it is in cells where room and sound are about equally loud, and no bound that is good to 4 dB splits those.

**A loss that is careful under the sound** (`train/cnntrain4.py`, `run_y1.sh`, `run_y2.sh`). Two changes to the
loss, both aimed at what the last paragraphs found. The cross-entropy part, the one that seeks the mean and makes
the haze, weighs 0.3 instead of 3. And a cell left under the true direct level (too deep) counts 1.5 times one left
over it (`ASYM`). Both looks were trained on from the 0.4.5 pair for four epochs with it, the second look on the new
first look's answers (`mkpass2.py`). Band scores, 0.4.5 pair then new pair (median / mean / worse than untouched):

| pictures | scoreboard, tools also in training (176) | tools never seen (64) |
|---|---|---|
| rings out | 44.6 / 26.1 / 6 % then 47.7 / 31.8 / 3 % | 54.2 / 39.4 / 9 % then 52.6 / 40.9 / 8 % |
| narrow | 36.6 / 15.5 / 6 % then 35.5 / 20.1 / 3 % | 42.5 / 39.1 / 6 % then 43.9 / 38.2 / 3 % |
| mono | 34.0 / 14.8 / 9 % then 35.0 / 18.5 / 6 % | 39.2 / 32.1 / 12 % then 41.1 / 36.9 / 8 % |
| cut off at 6 s | 29.6 / 10.1 / 12 % then 27.7 / 17.4 / 8 % | 24.7 / 22.5 / 12 % then 28.6 / 24.1 / 6 % |
| excerpt | 22.9 / 9.3 / 12 % then 22.5 / 14.5 / 5 % | 23.4 / 23.8 / 17 % then 20.5 / 24.3 / 6 % |

Under the sound with a light reverb (the table of "Is it the training data?" above, same columns): simulated rooms
18.9, 24 % worse then 28.0, 11 %; outside tools 22.0, 22 % then 30.6, 11 %; scenes 3.7, 40 % then 12.1, 17 %; with
other effects 17.2, 33 % then 19.8, 15 %; scoreboard 12.9, 29 % then 21.9, 19 %; tools never seen 19.9, 25 % then
25.0, 8 %. Over all reverbs the sound itself stays where it was (36.9 then 35.7 on the simulated rooms, 27.6 then
31.6 on the tools never seen), and the ring-out gains two points everywhere (90 to 93 then 93 to 95). From a sound
without any reverb the new pair takes less: the energy taken away falls from -15.8 to -23.3 dB (clean clips, median;
the worst tenth from -6.5 to -9.2) and from -16.5 to -24.0 dB (other effects; -5.5 to -11.6).

**The second look digs where the room's numbers are wrong** (`train/first_vs_pair.py`, `nums_all.py`,
`evid_study.py`). As heard on the 120, the new pair is better in the middle and at both ends of the table below,
but on two cases (both a simulated large room on a recording) it did clear harm: the sound itself 4.6 and 2.1 dB
further from its dry version, where the 0.4.5 pair has 1.7 dB on the first of them, its own worst case. In all
three the first look alone is nearly harmless (0.4, 0.0 and 0.5 dB). The second look goes 10 to 11 dB deeper than
the first on average there (6 dB for the 0.4.5 pair); on the other 118 cases never more than 3.9 dB. What misleads
it is the decay time read from the first answer: only six bands, all under 600 Hz, gave a free decay, and they read
5 to 6 s where the room has 1.3 s (in the mid bands) and 3.1 s. The low end of such a recording sinks slowly by
itself, which reads as a long room, and what is read in the low bands is carried flat to all bands above, where a
room is always shorter. With a decay that long the plane "how much room to expect" says room nearly everywhere.
Over all 120 the new first look gives the reader numbers as good as the old one (decay more than 1.5 times too long
on 9 of 110 found against 11 of 112; strength off by more than 6 dB on 18 against 22).

Where the evidence sits tells the two kinds apart (`evid_study.py`, `guard_bands2.py`; the scoreboard's 240
pictures, a room found on 223): with six or more mid bands (450 Hz to 7 kHz) that gave a free decay of their own
(184 pictures) the pair scores 54.7 at the median and 51.2 on average and none falls under -30; with fewer (39
pictures) it scores 28.8 at the median and -31.5 on average, and six fall under -30. Of the 66 rooms read into the
scoreboard's clips without any reverb, 27 rest on at most two such bands (with a reverb: 25 of 223).

**The guard** (`roomnum.hold_back`, in the app `RoomNumbers.h` `holdBack`; `train/guard_test.py`,
`guard_bands2.py`, `guard_rules.py`). The second answer may go deeper than the first by a limited amount on average
(over the cells within 50 dB of the loudest): 2 dB when at most two mid bands gave a decay, 8 dB from six on, a
straight line between. Beyond that it is pulled back towards the first answer until the excess is at the limit.
Rules of this kind on 3,233 pictures with a reverb (scoreboard in five forms and training samples) and 1,437
without (band score: median / mean / worse / under -30):

| rule | all with a reverb | mono (240) | never-seen tools (64), mean | acts on |
|---|---|---|---|---|
| pair as it is | 38.8 / 32.3 / 3.2 % / 38 | 36.2 / 23.4 | 40.9 | |
| first look only | 33.0 / 29.5 / 4.5 % / 37 | 24.8 / 18.0 | 42.8 | |
| limit 4 dB for all | 38.8 / 32.4 / 3.1 % / 38 | 36.6 / 23.2 | 44.5 | 109 |
| limit 3 dB for all | 38.4 / 32.4 / 3.0 % / 36 | 35.0 / 22.9 | 45.1 | 216 |
| limit 2 dB for all | 38.0 / 32.3 / 2.9 % / 33 | 33.4 / 22.3 | 45.5 | 413 |
| 2 dB on thin evidence, 8 dB on rich (in the app) | 38.8 / 32.6 / 3.1 % / 33 | 36.2 / 23.4 | 44.9 | 52 |

A limit for all costs on mono and narrow sounds, where the first look is weak and the second look rightly goes much
deeper; the limit by evidence leaves those alone. It is a net, not a repair: most of the bad cases come from the
first look itself (37 under -30 with the first look alone). What would repair the cause is a reader that does not
carry low-band decays flat upward, and a second look that has met wrong numbers in training.

As heard on the 120 scoreboard reverbs (engine alone; `depth2_test.py`, absolute figures as above):

| | 0.4.5 pair | new pair | new pair with the guard (0.4.6) |
|---|---|---|---|
| whole file: median / mean / worse | 26.5 / 21.8 / 14 % | 32.6 / 23.7 / 9 % | 32.6 / 27.1 / 8 % |
| the sound itself | 11.0 / 11.2 / 28 % | 14.4 / 12.6 / 22 % | 14.4 / 15.9 / 21 % |
| with a light reverb (48) | 10.9 / 6.7 / 35 % | 11.8 / 13.0 / 25 % | 11.8 / 13.0 / 25 % |
| ring-out, median | 74.8 | 82.9 | 82.7 |
| sound itself closer to dry by 0.5 dB+ | 49 | 54 | 54 |
| further by 0.25 / 0.5 / 1 dB and more | 9 / 3 / 1 | 5 / 3 / 2 | 4 / 2 / 0 |
| the worst case | 1.68 dB | 4.59 dB | 0.66 dB |

The target set before this run was 15 % worse under the sound; it came to 21 % (25 of 120), of which four are
further from the dry sound by a quarter of a dB or more. On the same 40 cases as the 0.4.5 write-up
(`cut_eval.py`) a room is found on 30 cut-off files (0.4.5 pair: 28) and on 23 excerpts (26), the room's numbers
are as good as before (strength off by 3.1 dB at the median against 2.8, decay within 20 % on 71 % against 74 %),
and the found room on a second sound scores 20.4 / 23.7 with 15 % worse against 18.3 / 24.7 with 20 %
(`final_eval.py 3`).

**In the app** (`bench/release_a.sh`, `release_ab.sh`, `release_b.sh`: the study tool and the app's own processing,
both built from the code of 0.4.6, over every app-level set). The sureness rule fitted again for the new pair with
the guard (`bench/fit_sure.py`): 1.892 + 0.758 log10(deep + 0.003) + 2.875 bands / 41, and 0.25 when no room is
found (for the 0.4.5 pair: 1.902, 0.847, 2.649 and 0.22). On the half of the 2,470 looks that neither training nor
the fit has seen, 0.4.5 pair then new pair:

| | room found | REVERB shown (65 %) | after the look without the other effects |
|---|---|---|---|
| reverbs that ring out (252) | 96 % then 95 % | 95 % then 94 % | |
| reverbs cut off at 6 s (188) | 77 % then 76 % | 71 % then 68 % | |
| no reverb (577) | 18 % then 13 % | 11.4 % then 10.7 % | 8.7 % then 9.4 % |
| no reverb, the other half (661) | 19 % then 14 % | 13.0 % then 9.8 % | 11.5 % then 8.9 % |

The new pair reads a room into fewer sounds that have none and is surer of those it does, so on the unseen half
the share shown stays where it was; over both halves REVERB is shown on 113 of the 1,238 sounds without reverb
against 126, and on 82 of the 912 with one other effect against 93 (before the look without the other effects: 96
against 119). The number still says what it means (called 65 to 80 % sure: 77 % have a reverb; 80 to 90: 85 %;
over 90: 95 %). Of 334 sounds with a real reverb that is shown (every third of them), 85 are looked at again
without their other effects and two lose their group (0.4.5: 78 and one; `bench/recheck_eval.py`).

The 40 test reverbs in three forms (`train/cut_app.py`), 0.4.5 then 0.4.6, each with the engine's own answer in
full (half way left in 0.4.5, far left in 0.4.6); a file without a REVERB group counts 0:

| | REVERB shown | median / mean / worse than untouched | median where shown | 0.4.6 at half way left |
|---|---|---|---|---|
| rings out | 36 then 37 | 25.2 / 26.6 / 1 then 30.3 / 29.6 / 1 | 26.4 then 32.9 | 18.7 / 18.1 / 1 |
| cut off at 6 s | 27 then 28 | 11.4 / 17.4 / 2 then 13.1 / 18.1 / 2 | 24.4 then 26.8 | 9.2 / 12.2 / 1 |
| excerpt | 24 then 20 | 4.7 / 13.7 / 2 then 0.0 / 13.8 / 2 | 24.1 then 29.9 | 2.9 / 9.0 / 0 |

The price is on excerpts: four files lose their REVERB group (0.4.5 scored 22, 7, 3 and 1 on them). At far left no
file in any form ends up further from the dry sound by 0.5 dB per cell or more, and one cut-off file by a quarter
of a dB. The far left of 0.4.5 (its bolder end) made 11, 9 and 7 of the 40 worse.

The scoreboard's 136 cases of the other effect families (`bench/famboard.py`): every family's own group and score
are as in 0.4.5. REVERB is shown where there is none on 9 cases instead of 13 (gone: one chorus, delay, gate and
phaser case each; none new). With every shown group at far left 6.6 % of the cases come out worse than untouched
(0.4.5: 8.8 %; with the new fader alone 8.1 %), and with everything shown put onto a second sound 22.8 % (24.3 %).

Long sounds (`bench/long_eval3.py`, the app's verdict in pieces): REVERB is shown on 15 of the 121 without a reverb
(0.4.5: 21; the looped clean clips 4 of 80 against 8, the twelve clean rows of clips 3 as before) and on 104 of the
144 with one (107).
How well the reverb comes out of the 53 long reverbs (`bench/long_score.py`, `long_abs.py`; REVERB shown on 43
in both builds):

| Amount position | mean score (all 53) | median where shown | worse than untouched | further from dry by 0.25 / 0.5 dB+ |
|---|---|---|---|---|
| 0.4.5, far left (the bolder end) | -11.9 | -3.4 | 23 | 15 / 8 |
| 0.4.5, half way left (its answer) | 14.1 | 22.4 | 6 | 3 / 1 |
| 0.4.6, far left (the answer) | 18.7 | 23.9 | 4 | 2 / 1 |
| 0.4.6, half way left (half of it) | 13.7 | 15.4 | 0 | 0 / 0 |

Of the 32 with a light reverb, three come out worse at far left (0.4.5: 17 at far left, six with its answer). The
one long sound that ends up half a dB further from its dry version is the same in both builds (0.56 dB with the
0.4.5 answer, 0.76 dB now).

## What was tried and did not help
- A plain physical rule as the answer (direct sound smeared with the room's decay): `look_test.py`.
- Rendering that treats the sound's left/right direction separately: `axis.py`, `axis_test.py`.
- Reading the best depth per sound from the answer's own statistics: `depth_test.py`, `bold_test.py`.
- A bolder far left of the Amount fader, and 27 other ways of reshaping the answer: `depth2_test.py` (0.4.6).
- Holding the answer back by what the found room can account for: `cap_test.py` (0.4.6).
- A third look: `third_look.py`. A whole-sound part in the model (`GLOB`): level with the plain model.
- Everything in `proto/` came before the model: hand-made rules and what they taught.
