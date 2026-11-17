# The verification step: Python side

`Source/Core/Verify.h` checks every effect against the whole sound before it is shown. This folder is the work
behind it: the measurements of the detection as it was up to 0.4.1, the sounds and features the verifier's models
were trained and tested on, the training, and the Python the C++ is a port of. There is no audio and no feature
data here; every sound is made, measured and thrown away.

## What was wrong

Up to 0.4.1 an effect was shown when its highest 3 s window scored above the cut. Measured with the app's own
code on 438 clean sounds (198 dry, 88 with a tone effect only, 152 with a reverb only), 13 % came out clean and
1.6 effects per sound were shown that are not there. Of those false finds 73 % are the CNN itself, 24 % a decider
lifting a CNN "no", 3 % the whole-file echo rule; 69 % exist only because the highest window counts, so a longer
sound collects more (one window alone: 0.84 per sound). Thresholds alone do not repair it: cuts chosen for 0.07
false effects per clean sound leave 28 % of the effects found.

## What the verifier does

Per effect one set of gradient-boosted trees (150 trees, at most 12 leaves) that reads 354 numbers about a stretch
of sound: what the two detectors computed, gathered over the windows (mean, highest, lowest of the CNN outputs, the
hand numbers and the deciders' scores; the scores as 0.4.1 showed them; the echo check), and 55 physical
measurements on the stretch itself (`evidence.py`, in C++ `evidence()`). Trained on 5503 sounds from 313 sources,
2150 of them clean (dry, reverb only, tone only, mixes, other speeds, segments, lossy codecs, silence around), with
the app's own effects and with outside effects, every sound at a random level and sample format. The cuts come from
out-of-fold scores (folds by source; a sound made with an outside effect is scored by trees that never saw that
effect): 0.07 false effects per clean sound in all, split over the effects so that most real ones are found, no
cut below 0.3 (`pickcut.py` is the check that fixed these two numbers before the test sounds were looked at).

## Results on sounds it was not tuned on

Held-out: test-split clips, test-split outside effects, nothing of it used for fitting or for the cuts.
"Before" is the app up to 0.4.1 on the same sounds.

| 195 clean sounds                | before         | verifier                     |
|---------------------------------|----------------|------------------------------|
| come out clean                  | 12 %           | 93 % (95 % range 89 to 96)   |
| false effects per sound         | 1.61           | 0.08 (0.04 to 0.12)          |
| dry (83)                        | 22 %, 1.29     | 94 %, 0.06                   |
| reverb only (64)                | 6 %, 1.91      | 94 %, 0.06                   |
| tone only (48)                  | 2 %, 1.77      | 90 %, 0.12                   |
| dry + 3 s of silence (83)       | 17 %, 1.52     | 94 %, 0.06                   |
| dry as 16-bit (83)              | 27 %, 1.16     | 94 %, 0.07                   |
| dry through a lossy codec (83)  | 20 %, 1.28     | 93 %, 0.07                   |
| two clips mixed (41)            | 12 %, 1.44     | 90 %, 0.12                   |

On 746 held-out sounds with an outside effect the verifier finds 46 % (mean over the 17 effects; before 50 %,
the difference is -3.6 points with a 95 % range of -6.6 to -0.8) and shows 0.09 other effects per sound (before 1.07).

| found, % of 44 sounds | before | verifier |   | found, % of 44 sounds | before | verifier |
|-----------------------|--------|----------|---|-----------------------|--------|----------|
| flanger               | 36     | 9        |   | tremolo               | 68     | 57       |
| chorus                | 45     | 30       |   | auto pan              | 98     | 100      |
| phaser                | 73     | 39       |   | saturation            | 7      | 0        |
| haas                  | 68     | 55       |   | exciter               | 2      | 7        |
| width (42)            | 31     | 24       |   | ring mod              | 23     | 82       |
| delay                 | 89     | 93       |   | filter                | 34     | 9        |
| distortion            | 20     | 23       |   | stutter               | 52     | 93       |
| bitcrush              | 75     | 70       |   | gated rhythm          | 68     | 61       |
| filter sweep          | 61     | 39       |   |                       |        |          |

A fixed filter, a mild saturation and an exciter cannot be told from the sound's own colour from one sound alone,
and flanger, phaser and filter sweep were found before only together with as many false ones (flanger on 12 %,
phaser on 21 %, filter sweep on 33 % of the clean sounds).

## The whole app on the scoreboard

`Research/room/bench/famboard.py` runs the app's processing (`drrender`) on the 136 cases: what is shown on sound A,
what taking it out does, and what putting it onto sound B does, each scored against the truth. Build 0.4.2 has the
verifier, the echo engine for the delay, and the reverb's second look with silence after the sound (the drop in
REVERB below is that one, not the verifier):

| 136 cases, 17 effects from outside tools                        | 0.4.1    | 0.4.2   |
|------------------------------------------------------------------|----------|---------|
| the case's own effect shown                                      | 77       | 75      |
| other groups shown: REVERB / the rest                            | 63 / 156 | 19 / 12 |
| cases with no other group shown                                  | 16       | 107     |
| own group far left: cases made worse than doing nothing          | 12 %     | 3 %     |
| every shown group far left: cases made worse than doing nothing  | 78 %     | 19 %    |
| everything shown put onto a second sound: cases made worse       | 71 %     | 36 %    |

What still does harm when everything shown is used: a REVERB group on a few sounds with a saturation, a flanger,
a chorus or a phaser, and the own groups of flanger, chorus, phaser and stutter, whose removal is not good yet.

## The Sureness slider

The trees' raw output z is shown as `sigmoid(slope * (z - z65) + logit(0.65))`: the fixed cut z65 sits at 65 %, the
slider's default, and one rule serves every effect: shown when the score is above the slider. The slope (0.59) is
the median of the logistic fits of right against wrong on the out-of-fold scores (`calib.py`, `export_verify.py`).
Held-out sounds (`slider.py`):

| slider | clean sounds that come out clean | false effects per clean sound | effects found | other effects per sound with an effect |
|--------|----------------------------------|-------------------------------|---------------|----------------------------------------|
| 30 %   | 67 %                             | 0.43                          | 63 %          | 0.53                                   |
| 50 %   | 87 %                             | 0.15                          | 54 %          | 0.20                                   |
| 65 %   | 93 %                             | 0.08                          | 46 %          | 0.09                                   |
| 80 %   | 98 %                             | 0.02                          | 39 %          | 0.03                                   |
| 99 %   | 100 %                            | 0.00                          | 15 %          | 0.00                                   |

How a score reads, on the 1231 held-out sounds (746 with one of the 17 effects, each effect on about 43): of the
effects with a score of 65 to 80 % 42 % are really there, of 80 to 95 % 74 %, above 95 % 99 %; of everything above
65 % 76 %. Where effects are rarer than in this mix the same scores are right less often.

## Long and short sounds

The verifier is trained on stretches of about 6 s. A longer sound is cut into blocks of 6 s (an end piece under
4.5 s joins the block before it), every block is judged on its own, and an effect's score is the highest score
that at least half of the blocks reach. Without that a long sound collects false effects block by block
(`longsim.py`, a model: k held-out sounds of one kind stand in for the blocks of one long sound; "found" is the
mean over the effects, the blocks being k sounds with the same outside effect):

| the sound       | shown when ..        | clean sounds that come out clean | false effects per clean sound | effects found |
|-----------------|----------------------|----------------------------------|-------------------------------|---------------|
| 6 s, 1 block    |                      | 91 %                             | 0.09                          | 49 %          |
| 12 s, 2 blocks  | in half of the blocks| 85 %                             | 0.15                          | 60 %          |
| 24 s, 4 blocks  | in any block         | 75 %                             | 0.26                          | 71 %          |
| 24 s, 4 blocks  | in half of the blocks| 92 %                             | 0.08                          | 54 %          |
| 48 s, 8 blocks  | in any block         | 66 %                             | 0.37                          | 79 %          |
| 48 s, 8 blocks  | in half of the blocks| 96 %                             | 0.04                          | 51 %          |

The price: an effect that is on less than half of a long sound is not shown (in the test below, an auto pan on the
last 12 s of 30 s). A sound under 6 s is one block. `lengths.py` makes real short (1, 2, 4 s) and long (30 s)
sounds for the same question; it has not been run to the end, so there are no numbers for short sounds yet.

## The C++ against this Python

`Tools/verify_parity.py` runs `drverify` (`Tests/verify.cpp`: the app's detection and `Verify.h`) and this Python on
the same sound and compares every number: the windows, the 55 measurements, the trees' outputs, the final scores.

- 494 sounds (83 dry, 27 with silence around, 384 stored cases): the largest difference of a final score is 0.0016,
  the 99.9th percentile 3e-08; of 8398 shown / not shown decisions none differs.
- Long sounds (five made ones of 30 to 36 s: clean, an effect on all of it, on a part, an odd end piece, silence in
  the middle): the final scores agree to 3e-17, of 85 decisions none differs.
- The app's build as shipped (after the detectors were made to share their hand numbers and echo check, which
  changed no result): 311 sounds, 5287 scores, 99.9th percentile of the difference 0.0033, the largest 0.069 (a
  flanger score on a mix of two clips, on both sides far under the cut); no decision differs.
- Cost: the verification adds about 8 % to the inspection (a 30 s sound: detectors 30.6 s, verification 2.4 s on
  one core of the test machine); sharing the hand numbers between the two detectors takes about 40 % off the
  detectors' own time.

## Files

- `evidence.py` the 55 measurements (numpy / scipy only). `vref.py` the verification as `Verify.h` does it: blocks,
  inputs, trees, the score. `calib.py` the shown score. `export_verify.py` writes `Resources/verify.bin` and reads it
  back. `Tools/verify_parity.py` compares all of this with the C++ (`Tests/verify.cpp`).
- `appdet.py` the app's detection in Python (the CNNs and deciders straight from `Resources/models.bin` and
  `models27.bin`), `own_fx.py` the app's own effects with the ranges the detectors were trained on. Both need the
  first prototype's modules (`feat.py`, `detect.py`, `echo.py`, `fxlib.py`, `fxlib27.py`), which are not in the repo.
- `zoo_fx.py` the outside effects: 64 open-source implementations for the 17 effects (LV2 and LADSPA plugins, sox
  and ffmpeg built-ins, a Python effects library), run through the hosts of `Research/room/bench/zoo_core.py`; 33
  of them are "dev", 31 "test". `mkpos.py` builds the detection scoreboard from them (136 cases, two sounds each).
- `extra_clean.py` 208 more clean sounds for training, from material the scoreboard's clips do not use.
- `gen_feats.py` makes every training and test sound and stores its features. `verifier.py` out-of-fold scores,
  the cuts, the final models. `pickcut.py`, `vreport.py`, `slider.py`, `bootstrap.py` the tables.
- `parity.py` the C++ against this Python on sounds made on the spot (the numbers above), `longsim.py` and
  `lengths.py` long and short sounds.
- `paths.py` where things are (environment variables), `cpuslot.py` an optional limit on the cores used.

## To reproduce

Needs the scoreboard's clips (`DR_CORPUS`: `meta.json` and `dry/<clip>.wav`, made by `Research/room/train/mkclips.py`),
the reverb and tone cases (`DR_SETS`), the first prototype's modules (`DR_PROTO`), the plugin hosts and plugins of
`Research/room/bench`, and numpy, scipy, scikit-learn 1.9, torch, numba, soundfile, pretty_midi, ffmpeg, sox.

    python3 mkpos.py                              # the scoreboard's 17 effect families (DR_SETS/<family>/)
    python3 gen_feats.py train 0 1                # 4777 rows; PART NPARTS splits the work over processes
    python3 gen_feats.py pos2 0 1                 # 820 more rows for the effects that are hardest to tell
    python3 gen_feats.py test 0 1                 # 1367 rows
    python3 verifier.py oof v2                    # out-of-fold scores (6 minutes on one core)
    python3 pickcut.py v2                         # the budget and the floor, judged inside the training sounds
    python3 verifier.py final v2 0.07 0.3         # the models, the cuts, the test scores
    python3 vreport.py v2 both 0.07 0.3           # the tables above; bootstrap.py v2 their ranges; slider.py v2 the slider
    python3 export_verify.py v2 --install         # Resources/verify.bin

The feature run behind the shipped file measured 29 further numbers per sound (an older check for comb teeth and
sweeps, the movement of the spectrum's shape and of its roll-off, echoes of onsets). The shipped models do not read
them (`verifier.py`, `DROP`; variant `v2`), and `evidence.py` here no longer computes them. With them (variant
`all`) the held-out numbers were 92 % clean, 0.09 false effects per sound and 48 % of the effects found; flanger is
the effect that loses by leaving them out (20 % found with them, 9 % without).
