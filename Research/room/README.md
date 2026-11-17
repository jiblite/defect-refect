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

`bench/cut_test.py` is the test behind the second look of 0.4.2 (`Estimate.h`, `recheckRoomEnd`): the verdict on
sounds as files, cut off at 6 s, and with 3 s of silence after them. The model learned from sounds that end inside
the file. Cut off while still playing, a room showed on 54 of 120 dry sounds (on 18 with silence after them) and
on 90 of the corpus's 198 dry clips (29). The app now looks a second time with silence after the sound and the
less sure verdict counts: 18 and 29. Sounds with a natural ending are unchanged (104 of 120 with a reverb, 18 of
120 dry); cut sounds with a real reverb pay, 68 of 120 shown instead of 88. Other ways to join the two looks
(their mean, the root of their product) keep 75 to 78 of those and show 21 to 22 of the dry ones. Training the
model on sounds that are cut off is the real repair and is still to do.

## What was tried and did not help
- A plain physical rule as the answer (direct sound smeared with the room's decay): `look_test.py`.
- Rendering that treats the sound's left/right direction separately: `axis.py`, `axis_test.py`.
- Reading the best depth per sound from the answer's own statistics: `depth_test.py`, `bold_test.py`.
- A third look: `third_look.py`. A whole-sound part in the model (`GLOB`): level with the plain model.
- Everything in `proto/` came before the model: hand-made rules and what they taught.
