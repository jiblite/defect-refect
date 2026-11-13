# TONE engine prototype: report

All numbers come from the scoreboard's own distance `d` (bench/score.py). Nothing was listened to.
"Removed %" is how much of the effect the fader at 0 takes away. "Refect %" is how close the second sound comes to its true wet version when the found curve is put on it. 100 is perfect, 0 is no gain, below zero is worse than doing nothing.
The three test effects (cabinet, fil4, sox_eq) were scored once, at the end, with every setting frozen. Later edits changed no number.

## In short

- **The machinery works.** With an effect's true curve, the engine's filter removes 98 to 100 % (median per effect) of every EQ, filter, resonance and comb, and puts them on a second sound as well (96 to 100 %). Its limits are the box (a curve cannot undo a small room) and the cabinet (undoing it needs more boost than the +18 dB cap allows).
- **Blind, only reflections are found.** For the comb effect the default removes 93.6 % (median of 8 dev cases; ceiling 99.7 %) and carries it to a second sound with 94.7 % (ceiling 99.8 %).
- **Blind, everything else stays unseen.** For EQ, filter, resonances, cabinet and box the default reports nothing and scores exactly 0 %. That covers all 24 test cases: the test split has no comb.
- **Clean sounds stay clean.** The default left all 128 dry sounds of the scoreboard untouched, bit for bit. On 1057 clean training clips it reported a reflection that is not there on 2 (0.2 %). Flattening those two changes them by 0.79 and 1.15, as much as a typical effect does (0.7 to 1.1).
- **A learned model was built to guess the rest, and is switched off by default.** It adds little (test: median 2.8 %, mean 7.8 % removed; 3 of 24 cases get worse) and it invents a curve on some clean sounds (9 of the 128 clean scoreboard sounds change by more than 0.1, the worst by 1.7). The brief rules that out for a default.
- **Tone Match is the dependable way for EQ-like curves.** Given both versions of the same sound it removes 97.6 % (dev) and 99.4 % (test) at Detail 1 (medians).

## 1. Files

In `/tmp/claude-0/v2/tone/`:

| File | What it is |
|---|---|
| `tone.py` | The engine: `contour`, `smooth`, `apply_curve`, `put_curve`, `match`, `to_ir`, plus `find_echoes` and `comb_db`. Needs numpy and scipy only. |
| `tone_model.npz` | The optional model's weights (73 042 numbers, float32). Only read when `contour(..., model=True)` is called. |
| `tone_eval.py` | Prints the tables for all 64 cases, the check on clean sounds and the Tone Match rows. `--show` prints them again from `tone_eval.json`. A full run took 3 to 5 minutes with 2 processes. |
| `tone_eval.json` | Results of the final run, per case and per clean clip (no clip names). |
| `lab/` | The experiments: ceilings (`ceil.py`), the classic estimator (`classic.py`), the echo finder's history (`echo*.py`), model training (`feat.py`, `curves.py`, `model.py`, `train.py`, `export.py`), dev scoring (`evaldev.py`), Tone Match (`match_eval.py`), impulse-response and fader checks (`ir_check.py`), clean-clip checks (`clean_check.py`, `echo_sim.py`). `lab/work/` holds caches and logs. |

The interface:

```
grid_hz, curve_db = contour(x, detail=1.0, stereo='linked', echo=True, model=False, restraint=None, sens=1.0, info=None)
y = apply_curve(x, grid_hz, curve_db, amount)     # the analysed sound: 0 = curve removed, 1 = untouched, 2 = applied once more
y = put_curve(y, grid_hz, curve_db, amount)       # a second sound:    0 = none, 1 = as found, 2 = double
grid_hz, curve_db = match(a, b, detail=0.6, stereo='linked')    # Tone Match: gives b the long-term spectrum of a; apply with put_curve
h = to_ir(grid_hz, curve_db, n=16384, phase='minimum')          # or phase='linear'
```

The curve is in dB on the 4097 analysis bins (5.86 Hz apart, 0 to 24 kHz). All functions accept any other increasing grid too.
An all-zero curve means "nothing found"; the fader then returns the sound itself.

## 2. What a curve can reach at all (ceilings)

Before estimating anything I put the **true** curve of each effect through different filters. This says how to represent the curve and what a perfect estimate would score.
Removed %, median of the 8 cases per effect. The ladder ran on all 64 cases, test included; it involves no estimate.

| True curve applied as | box | calf_eq | comb | pb_eq | reson. | cabinet | fil4 | sox_eq |
|---|---|---|---|---|---|---|---|---|
| exact inverse filter (phase included, not a curve) | 100 | 100 | 100 | 100 | 100 | 99.8 | 100 | 100 |
| curve in 2.9 Hz steps, 32k taps, minimum phase, per channel, no cap | 61.3 | 99.9 | 99.6 | 99.9 | 99.5 | 95.7 | 100 | 99.9 |
| the same, gains capped at 24 dB | 61.7 | 99.9 | 99.6 | 99.9 | 99.5 | 69.6 | 100 | 99.9 |
| capped at 18 dB | 63.1 | 99.9 | 99.6 | 99.9 | 99.5 | 57.0 | 100 | 99.9 |
| capped at 12 dB | 62.4 | 99.9 | 99.6 | 99.9 | 98.3 | 45.2 | 97.8 | 99.9 |
| cap 24, one curve for both channels | 17.5 | 99.9 | 99.6 | 99.9 | 99.5 | 69.6 | 100 | 99.9 |
| cap 24, linear phase instead of minimum | 46.0 | 96.7 | 78.7 | 97.4 | 86.8 | 67.3 | 98.0 | 97.5 |
| cap 24, 8k taps | 57.9 | 99.9 | 99.6 | 99.9 | 99.6 | 69.5 | 100 | 99.9 |
| cap 24, 2k taps | 43.4 | 99.9 | 99.6 | 99.9 | 98.2 | 68.5 | 100 | 99.9 |
| cap 24, steps of 5.9 Hz (the engine's choice) | 63.1 | 99.9 | 99.6 | 99.9 | 98.4 | 69.5 | 100 | 99.9 |
| cap 24, steps of 23.4 Hz (what 2048-sample frames give) | 49.2 | 99.7 | 97.3 | 99.4 | 93.2 | 67.8 | 99.9 | 99.8 |
| cap 24, log grid, 1/48 octave | 48.7 | 99.9 | 92.2 | 99.9 | 97.1 | 68.7 | 99.9 | 99.9 |
| cap 24, log grid, 1/24 octave | 39.8 | 99.8 | 79.4 | 99.8 | 90.7 | 68.0 | 99.9 | 99.8 |
| cap 24, log grid, 1/12 octave | 26.5 | 99.3 | 61.0 | 99.6 | 80.1 | 66.6 | 99.9 | 99.1 |
| cap 24, log grid, 1/3 octave | 11.7 | 92.2 | 19.2 | 95.6 | 40.6 | 52.4 | 99.0 | 89.4 |

What follows, and where it departs from the brief's starting idea:

- **The curve lives on evenly spaced bins, not on a log grid.** A 1/24-octave grid loses a fifth of a comb and a tenth of the resonances: comb teeth and narrow peaks are evenly spaced in Hz and too dense for a log grid at the top. EQ curves do not care.
- **Frames of 8192 samples (hop 2048), not 2048/512.** They give the 5.86 Hz steps. The analysis runs once over the whole sound, so the long frame costs no latency.
- **Minimum phase by default.** The effects are themselves (nearly) minimum phase, so the minimum-phase inverse also undoes their phase. Linear phase costs 2 to 3 points on EQ and 21 on comb.
- **16384 taps.** Fewer taps only hurt the box and narrow resonances.
- **One curve for both channels by default.** Only the box differs between left and right. `stereo='dual'` gives one curve per channel.
- **Caps: +18 dB boost, -30 dB cut**, bent in softly from three quarters of the cap, applied after the fader. The boost cap holds the cabinet's Removed ceiling at 57 %. I kept it: a higher cap lifts noise by as much, and raising it only to help test cases would have been tuning on them.

## 3. Blind estimation: what was tried

### 3.1 The classic estimator (fine detail minus broad trend): gains nothing

The idea: the effect's curve is what remains of the long-term spectrum once the sound's own broad shape is taken off.
I split the long-term spectrum into six detail bands, from finer than 1/24 octave to about an octave, and asked band by band how much of what is seen belongs to an effect.
On 6000 training examples (clean clip plus random curve) the sound's own detail is 3 to 4 times larger than the curve's at every scale: 2.0, 1.2, 1.3, 1.5, 2.0, 3.1 dB rms against 0.50, 0.32, 0.37, 0.45, 0.59, 0.87 dB, finest to broadest.
The best gain per band is 0.05 to 0.09 (0.11 to 0.17 with a statistic that looks between the partials). The best threshold is "keep nothing".
On held-out clips the fitted estimator's curve error is 2.49 dB, the same as predicting no curve (2.49 dB). On the real dev wet sounds it is within 0.03 dB of predicting nothing for every effect.
So no fixed rule of this kind can tell a mild EQ from the sound's own colour.

### 3.2 The echo finder: works (this is the default)

A reflection adds a ripple to the log spectrum that is periodic in Hz. In the cepstrum of the long-term log spectrum it is one peak, at the reflection's delay, as high as its gain.
A candidate peak between 0.5 and 15 ms counts as a reflection when it passes four tests:

1. Its gain is at least 0.25 (0.40 for delays under 1 ms, where a sound's own formants ripple too).
2. Its second peak, at twice the delay, is negative, as a reflection's is. A steady pitch gives a positive one.
3. Every sixth of the sound shows it, with the same sign and at least 0.4 of its size. A played note does not last that way.
4. It stands alone: once its comb is taken out, nothing half its size is left within 10 samples. A cluster of peaks is the sound's own ripple.

Delay and gain are then refined by fitting the whole comb curve, to a fiftieth of a sample. Up to two reflections are reported.

Three earlier versions failed, and the reasons matter for a port.
Judging by steadiness alone gave 20 to 40 % false reports on melodic clips. Measuring the ripple frame by frame underestimated the gain on sparse spectra.
The decisive fix: the weights that say where the sound has energy must be smoothed over 1.5 kHz first. Otherwise the comb's own notches enter the weights and split its peak into side peaks.

Results with the shipped code:

| Check | Result |
|---|---|
| 8 dev comb cases | all found; delay within 0.06 samples on 7, off by 0.32 on the eighth (a 0.66 ms delay); gain within 0.035 |
| Hand-made reflection on 120 clean training clips, gain 0.30 | found on 72 % (91 % for delays over 2 ms) |
| the same, gain 0.45 / 0.70 | found on 93 % / 96 %; delay error 0.01 to 0.02 samples (median), gain error 0.01 to 0.02; never a wrong second reflection |
| False reports, 128 dry scoreboard sounds (94 different ones) | 0 |
| False reports, the 56 wet scoreboard sounds without a comb | 0 |
| False reports, 1057 clean training clips | 2 (0.2 %), both MIDI-rendered clips; in the one I examined, the instrument's own spectrum ripples steadily like a comb |

The thresholds were set while looking at the dev combs and the training clips, so the last row is not an independent test. The scoreboard's dry sounds and the test split's wet sounds are.

### 3.3 The learned model: a little gain, real harm (optional, off by default)

A small convolution net along a 1/24-octave grid of 240 cells from 20 Hz to 20 kHz: a 1x1 input layer, nine residual blocks (5-tap convolutions, 40 channels, dilations 1, 2, 4, 8, 16, 8, 4, 2, 1), a 1x1 output layer. 73 042 weights. Inference is plain numpy in `tone.py`; it matches the trained net to 0.00003 dB.

**Its inputs, exactly** (14 rows of 240 cells, from the power spectra of the frames that are not silent):

| Row | Content |
|---|---|
| 0 | level of the long-term log spectrum under its loudest region, divided by 20 dB |
| 1 | long-term log spectrum minus its own half-octave trend, divided by 10 dB |
| 2 to 9 | eight level statistics relative to the long-term log spectrum, each divided by 10 dB: the level between the partials, the 10th, 25th, 50th, 75th and 90th percentile over time, the level of the loud frames, the plain power mean |
| 10 | mean change from frame to frame, divided by 5 dB |
| 11 | level at note onsets relative to the long-term log spectrum, divided by 10 dB |
| 12 | how line-like the spectrum is (difference to the 9-bin neighbourhood), divided by 5 dB |
| 13 | position on the frequency axis, -1 to +1 |

**Its outputs:** the curve in dB per cell, and its own doubt per cell (the error in dB it expects).
The curve is shrunk where it is small against the doubt: `c³ / (c² + (restraint x doubt)²)`, restraint 2. A curve that stays under 1 dB everywhere counts as nothing.

**Training.** Clean clips plus random curves. A fixed filter shifts every long-term log statistic of a bin by exactly its gain, so no audio has to be rendered. Checked on real wet sounds: 0.02 to 0.4 dB rms error for EQ, comb and resonances, about 1 dB for the box.
957 training clips plus 5000 random mixes of two or three clips. 100 clips were held out for checking; they can still occur inside mixes.
40 000 curves: EQ and resonances with the dev effects' parameter ranges, boxes from the dev box generator, combs, and two inventions of my own (measured-looking responses with band limits and bumps, and tilts). The three test effects were not used.
40 % of the examples carry no curve at all, so the net learns to say "nothing". 15 000 steps, 35 minutes on two threads.

**Curve error** (dB, mean absolute, level-free), model against predicting no curve:

| | model | no curve |
|---|---|---|
| held-out clips, all curve types | 1.87 | 2.57 |
| of which EQ | 1.65 | 1.75 |
| of which resonances | 0.64 | 0.67 |
| of which measured-looking | 2.65 | 4.33 |
| of which box | 1.97 | 3.38 |
| held-out clips with no curve on them | 0.09 | 0 |
| real dev wet sounds: pb_eq / calf_eq | 2.45 / 1.86 | 2.71 / 1.92 |
| real dev wet sounds: box / resonances | 3.37 / 0.79 | 4.13 / 0.79 |

So the model sees large, broad curves (band limits, boxes, tilts) and hardly sees a mild EQ.
An earlier, bolder version (fewer no-curve examples, no doubt output) removed more on the dev EQs (median 24.6 % on pb_eq, 11.1 % on calf_eq) but changed clean sounds by 0.25 on average and by 3.1 at worst. The cautious version is the one shipped.

Where it goes wrong on clean sounds: a noise-like synthetic sound is taken for treble-boosted, and band-limited loops and recordings are taken for filtered. An unusual sound and an effect look the same to it. Its doubt output does not flag these cases reliably.

## 4. Results on all 64 cases (final run)

Each cell: median / mean / share of cases below zero. Ceilings: median / mean.
"Default" is `contour(x)`. "With model" is `contour(x, model=True)`. "Model, no restraint" is the same without shrinking.
"Ceiling, per channel" is the true curve through the engine's own filter and caps, left and right apart: what a perfect blind estimate would reach. "Ceiling, one curve" averages the two channels' true curves into one, the engine's default layout.

### Removed % (fader at 0 on the analysed sound)

| | default | with model | model, no restraint | ceiling, one curve | ceiling, per channel |
|---|---|---|---|---|---|
| **dev, all 40** | 0.0 / 18.8 / 0% | 3.4 / 23.6 / 2% | 6.6 / 20.3 / 5% | 99.8 / 82.6 | 99.8 / 90.6 |
| **test, all 24** | 0.0 / 0.0 / 0% | 2.8 / 7.8 / 12% | 10.7 / 3.6 / 21% | 100.0 / 87.5 | 100.0 / 87.5 |
| comb (dev) | 93.6 / 93.9 / 0% | 93.1 / 91.0 / 0% | 90.3 / 78.9 / 0% | 99.7 / 99.4 | 99.7 / 99.4 |
| box (dev) | 0.0 / 0.0 / 0% | 2.0 / 3.5 / 0% | 1.5 / 4.8 / 12% | 21.1 / 17.1 | 63.0 / 56.9 |
| calf_eq (dev) | 0.0 / 0.0 / 0% | 0.0 / 3.7 / 0% | 0.4 / 4.7 / 0% | 100.0 / 99.9 | 100.0 / 99.9 |
| pb_eq (dev) | 0.0 / 0.0 / 0% | 0.2 / 2.4 / 0% | 3.7 / 6.6 / 0% | 99.9 / 99.9 | 99.9 / 99.9 |
| pb_resonances (dev) | 0.0 / 0.0 / 0% | 17.3 / 17.6 / 12% | 7.2 / 6.5 / 12% | 98.4 / 96.8 | 98.4 / 96.8 |
| cabinet (test) | 0.0 / 0.0 / 0% | 11.9 / 10.5 / 12% | 12.1 / 10.6 / 12% | 56.6 / 62.7 | 56.6 / 62.7 |
| fil4 (test) | 0.0 / 0.0 / 0% | 0.1 / 2.7 / 12% | 4.1 / 1.7 / 25% | 100.0 / 100.0 | 100.0 / 100.0 |
| sox_eq (test) | 0.0 / 0.0 / 0% | 1.1 / 10.3 / 12% | 3.0 / -1.4 / 25% | 100.0 / 99.7 | 100.0 / 99.7 |

### Refect % (the curve found on A put onto the second sound B)

| | default | with model | model, no restraint | ceiling, one curve | ceiling, per channel |
|---|---|---|---|---|---|
| **dev, all 40** | 0.0 / 19.0 / 0% | 3.9 / 24.1 / 2% | 6.3 / 18.9 / 8% | 99.8 / 85.5 | 99.8 / 92.7 |
| **test, all 24** | 0.0 / 0.0 / 0% | 10.5 / 15.9 / 12% | 14.7 / 14.8 / 21% | 99.8 / 97.0 | 99.8 / 97.0 |
| comb (dev) | 94.7 / 95.2 / 0% | 94.0 / 92.0 / 0% | 92.6 / 78.2 / 0% | 99.8 / 99.6 | 99.8 / 99.6 |
| box (dev) | 0.0 / 0.0 / 0% | 3.3 / 4.5 / 0% | 3.7 / 6.4 / 0% | 33.8 / 34.3 | 72.5 / 70.4 |
| calf_eq (dev) | 0.0 / 0.0 / 0% | 0.0 / 3.9 / 0% | 0.0 / 3.9 / 12% | 100.0 / 100.0 | 100.0 / 100.0 |
| pb_eq (dev) | 0.0 / 0.0 / 0% | 0.5 / 3.1 / 0% | 5.5 / 8.6 / 0% | 99.9 / 99.8 | 99.9 / 99.8 |
| pb_resonances (dev) | 0.0 / 0.0 / 0% | 14.5 / 17.1 / 12% | 3.0 / -2.6 / 25% | 95.9 / 93.8 | 95.9 / 93.8 |
| cabinet (test) | 0.0 / 0.0 / 0% | 21.5 / 26.7 / 0% | 31.8 / 29.2 / 12% | 92.7 / 92.5 | 92.7 / 92.5 |
| fil4 (test) | 0.0 / 0.0 / 0% | 0.0 / 9.2 / 25% | 10.1 / 11.9 / 25% | 100.0 / 99.2 | 100.0 / 99.2 |
| sox_eq (test) | 0.0 / 0.0 / 0% | 0.8 / 11.9 / 12% | 3.6 / 3.4 / 25% | 100.0 / 99.4 | 100.0 / 99.4 |

Reading the tables:

- The default's dev mean of 18.8 % is the eight comb cases and nothing else. Its test score is exactly zero because nothing was reported on any test sound, wet or dry.
- With the model, the cases that gain most are four dev resonance cases (33 to 41 %), cabinet_00 (24 %), fil4_02 (26 %) and sox_eq_07 (70 %). Cases that get worse: pb_resonances_02 (-8 %), cabinet_06 (-10 %), fil4_04 (-12 %), sox_eq_06 (-12 %).
- The model also disturbs two comb cases that the echo finder alone solves: comb_00 falls from 86.5 to 76.2 %, comb_06 from 98.6 to 85.3 %.
- The eight comb cases, default, Removed %: 86.5, 91.9, 98.3, 93.5, 96.3, 93.7, 98.6, 92.7.
- All per-case numbers: `python3 tone_eval.py --show`.

### Harm on clean sounds: d(dry, flatten(dry)), fader at 0 on a sound with no effect

For scale, an effect changes a sound by d(dry, wet) = 0.68 (dev median) and 1.14 (test median); the mildest effect on the scoreboard by 0.15.

| Clean sounds | Setting | Untouched | Mean | 90th percentile | Largest | Changed by more than 0.1 |
|---|---|---|---|---|---|---|
| dev: 80 dry sounds (57 different) | default | 80 of 80 | 0.000 | 0.000 | 0.000 | 0 |
| | with model | 67 of 80 | 0.044 | 0.022 | 1.696 | 4 |
| | model, no restraint | 50 of 80 | 0.112 | 0.231 | 2.869 | 14 |
| test: 48 dry sounds (37 different) | default | 48 of 48 | 0.000 | 0.000 | 0.000 | 0 |
| | with model | 35 of 48 | 0.071 | 0.154 | 1.137 | 5 |
| | model, no restraint | 23 of 48 | 0.207 | 0.591 | 2.111 | 17 |
| 200 clean training clips | default | 199 of 200 | 0.006 | 0.000 | 1.150 | 1 |
| | with model | 189 of 200 | 0.008 | 0.000 | 1.150 | 2 |
| all 1057 clean training clips | default | 1055 of 1057 | | | 1.150 | 2 |

"Untouched" means the curve is exactly zero and the output equals the input.
The one changed clip in the default's 200-clip row is a false reflection (section 3.2).

## 5. Tone Match

`match(a, b, detail)` is the difference of the two plain long-term power spectra, smoothed by Detail. It is taken only where both sounds have something (within 60 dB of their loudest region). Its level is set so that b keeps its long-term power. No model is involved.

**Two versions of the same sound** (target A dry, sound A wet, scored like Removed; median / mean):

| | Detail 0.6 | Detail 1 | ceiling, one curve |
|---|---|---|---|
| **dev, all 40** | 70.9 / 63.8 | 97.6 / 83.9 | 99.8 / 82.6 |
| **test, all 24** | 96.7 / 80.6 | 99.4 / 84.4 | 100.0 / 87.5 |
| comb (dev) | 41.5 / 42.5 | 96.8 / 95.8 | 99.7 / 99.4 |
| box (dev) | 15.9 / 17.8 | 29.9 / 30.4 | 21.1 / 17.1 |
| calf_eq (dev) | 97.8 / 96.8 | 99.5 / 99.3 | 100.0 / 99.9 |
| pb_eq (dev) | 98.0 / 97.8 | 99.4 / 99.3 | 99.9 / 99.9 |
| pb_resonances (dev) | 63.3 / 63.9 | 95.8 / 94.9 | 98.4 / 96.8 |
| cabinet (test) | 47.8 / 46.2 | 55.9 / 54.7 | 56.6 / 62.7 |
| fil4 (test) | 99.5 / 99.3 | 99.8 / 99.6 | 100.0 / 100.0 |
| sox_eq (test) | 96.7 / 96.2 | 99.8 / 98.8 | 100.0 / 99.7 |

More dev measurements on the same pairs:

- By Detail, median over the 40 dev cases: 19.6 % at 0.25, 57.0 % at 0.5, 70.9 % at 0.6, 86.2 % at 0.75, 93.5 % at 0.9, 97.6 % at 1.
- Carrying the pair's curve to the other sound B (Refect): 59.0 % at Detail 0.6, 95.6 % at Detail 1.
- Using the mean of the log spectrum instead of the power mean changes nothing (96.9 against 97.6 %), so the simpler power mean stays.
- Box pairs: `stereo='dual'` lifts the median from 29.9 to 57.9 %, close to the per-channel ceiling of 63 %.

**Two different sounds** (target A wet, sound B dry, 40 dev pairs). Here there is no truth to score against, only how close the long-term spectra come.

| Detail | Difference before | Difference after (median) | Pairs where the curve comes within 1 dB of a cap | Level change of the sound |
|---|---|---|---|---|
| 0.25 | 7.9 dB rms | 1.5 dB rms | 24 of 40 | -0.01 dB (median) |
| 0.6 | 8.3 dB rms | 1.6 dB rms | 29 of 40 | -0.05 dB, range -0.28 to +0.10 |
| 1 | 10.2 dB rms | 2.8 dB rms | 36 of 40 | -0.43 dB, range -1.75 to -0.03 |

What is left after matching sits where the curve hits its caps: two unrelated sounds often differ by more than 18 dB somewhere. The worst pair keeps 11.6 dB rms at Detail 0.6.

## 6. Fader and impulse-response checks

- **Fader positions against the truth** (6 dev cases, true curve; "doubled" is compared with the effect really applied a second time): fader 0 removes 98 to 100 %, fader 2 matches the double effect to 99 to 100 %, the second sound at 1 and at 2 matches to 99 to 100 %.
- **With the blind curve on two comb cases:** 86 and 92 % at fader 0, 86 and 95 % at fader 2, 88 and 94 % on the second sound at 1, 89 and 94 % at 2.
- **Between the stops** the result moves steadily: on comb_00 the distance to the dry sound is 1.19, 0.91, 0.62, 0.34, 0.16 at fader 1, 0.75, 0.5, 0.25, 0.
- **Fader 1** returns the sound itself, bit for bit. So does any fader position when nothing was found.
- **Impulse response against its curve** (error of its magnitude, 20 Hz to 20 kHz, rms / largest, in dB):

| Curve | minimum, 2048 taps | minimum, 16384 taps | linear, 2048 taps | linear, 16384 taps |
|---|---|---|---|---|
| smooth EQ (bells and a shelf) | 0.00 / 0.0 | 0.00 / 0.0 | 0.00 / 0.0 | 0.00 / 0.0 |
| reflection, gain 0.7 | 0.01 / 0.0 | 0.00 / 0.0 | 0.02 / 0.1 | 0.00 / 0.0 |
| reflection, gain -0.9 (20 dB notches) | 0.32 / 2.7 | 0.02 / 0.2 | 1.47 / 8.6 | 0.06 / 0.9 |
| a true box curve | 0.48 / 12.9 | 0.02 / 0.4 | 2.10 / 15.5 | 0.06 / 1.0 |

- The linear-phase response is symmetric around tap n/2 to 3e-9; its latency is n/2 samples. The minimum-phase one has its largest tap at 0 and no latency.
- Two curve rows give a two-channel response. A curve on a 240-point log grid is accepted.
- A hand-made reflection on noise (150 samples, gain 0.6) is found and taken out: 0.3 % of it is left, level-matched, against 53 % before.

## 7. Recommended defaults

| Setting | Value | Why |
|---|---|---|
| Blind curve | echo finder only (`model=False`) | the only part that never touched a clean scoreboard sound |
| Model's guess | off; offer it as a labelled suggestion the user switches on | 9 of 128 clean scoreboard sounds changed by more than 0.1; gain on EQ near zero |
| Restraint (when the model is on) | 2 | dev mean Removed 23.6 % against 20.3 % unrestrained, harm 0.044 against 0.112 |
| Detail, blind curve | 1 | it acts on the model's part only; a reflection is kept exactly, smoothing would erase its teeth |
| Detail, Tone Match, different sounds | 0.6 (a Gaussian of 1/12 octave) | spectra come within 1.6 dB; at 1 the target's individual partials would be copied |
| Detail, Tone Match, two versions of one recording | 1 | 97.6 % against 70.9 % removed (dev medians) |
| Stereo | one curve for both channels; per channel for Tone Match on a recording pair with a stereo effect | box pairs: 57.9 against 29.9 % |
| Filter | minimum phase, 16384 taps | ceilings in section 2 |
| Gain caps | +18 dB boost, -30 dB cut, soft knee from 3/4 of the cap, after the fader | protects against lifting noise; costs the cabinet's Removed ceiling |
| Level | the curve's level is set so that the fader does not change the long-term power | a loudness jump would mislead the ear; the scorer ignores level |

**How far to trust the blind curve.**
A reported reflection can be trusted: delay to a few hundredths of a sample, gain to about 0.03, 86 to 99 % removed on every dev case.
"Nothing found" does not mean nothing is there: EQ, filters, resonances, cabinets and boxes are invisible to the default.
The model's guess should be shown as a guess. On EQ it is right about a few per cent of the effect, and on about one clean sound in fourteen it changes the sound by more than 0.1.

**Does Tone Match need anything special?**
Yes, four things, all in `match`: the caps (most unrelated pairs reach one), the rule to compare only where both sounds have content, keeping the level of the sound, and two Detail defaults depending on whether the sounds are the same recording.
For a recording pair with a stereo effect it also needs the per-channel option.

## 8. Weak spots

1. **No blind estimate for seven of the eight effect types** in the default. This is the main finding, not an oversight: sections 3.1 and 3.3 show how little a single sound's long-term spectrum reveals.
2. **False reflections on about 0.2 % of clean sounds**, and when it happens flattening does real damage (0.79 and 1.15 on the two clips). The interface should show the delay and gain found, and leave the fader at 1 until the user moves it.
3. **Weak or coloured reflections are missed.** Under gain 0.25 (0.40 under 1 ms) nothing is reported, by design. Gain 0.30 is found 72 % of the time. A box's wall reflections (gains 0.3 to 0.65, but several at once and coloured) are not found.
4. **Limits of the echo finder:** delays of 0.5 to 15 ms, two reflections at most, and at least about one second of sound.
5. **Very short delays are less exact:** comb_00 (0.66 ms) is found 0.32 samples off and scores 86.5 %, the lowest comb.
6. **The box** stays out of reach even with the true curve: 63 % per channel, 21 % with one curve.
7. **The cabinet** can be put on (Refect ceiling 92.7 %) but only half removed (56.6 %) under the +18 dB cap.
8. **Memory on long sounds:** the analysis keeps every frame's spectrum (4097 values per 43 ms) to take a percentile over time. A port should cap the number of frames or use running estimates.
9. **The level rule** means the fader at 0 does not restore the dry sound's level, and at 2 does not add the effect's own level change. The reflections found are returned separately (`info`), so an exact inverse is one call to `comb_db` away.

## 9. Not verified

- **Listening.** Every number is the scorer's. Whether a 93 % removed comb sounds clean was not checked by ear.
- **Finding reflections on unseen effects.** The test split has no comb, so the echo finder's hit rate is known only from the 8 dev cases and from hand-made reflections on training clips, both of which I saw while setting its thresholds.
- **The model's scores may flatter it.** 14 of the 80 dev dry sounds and 5 of the 48 test dry sounds have a training clip whose long-term spectrum is within 1 dB rms (training clips among themselves are never closer than 2.9 dB). The model may know those sounds. The default does not use the model, so its numbers are unaffected.
- **Tone Match at Detail 0.25, 0.5, 0.75 and 0.9**, and its Refect numbers, were measured before a late change to how `match` handles the caps. The Detail 0.6 and 1 values were measured again afterwards and did not change.
- **Detail on the model's part** was not evaluated; all model numbers are at Detail 1.
- **Per-channel blind curves** were tried on dev only and gained nothing measurable.
- **Sample rates other than 48 kHz** are not handled. Mono, very short and silent inputs were only smoke-tested.
- **The impulse responses** were checked numerically, not loaded into a convolution host.
- **The C++ port** is not started. Python timing for a 9-second sound: 0.26 s for the blind curve, 0.05 s to filter.

Repeatability was checked: the final tables were produced three times, the last time with the final code, with identical numbers.

## 10. Notes for the port

The default path needs no model and only one percentile statistic:

1. Power spectra of Hann frames (8192, hop 2048); keep frames within 45 dB of the loud ones.
2. Per bin: the 95th percentile over time, then the mean over time of the log power held at 40 dB under that percentile. Do this for the whole sound and for each sixth of it.
3. Weights: a Hann taper from 100 Hz to 16 kHz times "where the sound has energy", from the level smoothed over 257 bins.
4. One inverse FFT of size 8192 gives the cepstrum; candidates are tested with plain cosine sums over the 4097 bins.
5. The comb curve is a formula: `10 log10(1 + g² + 2 g cos(2 pi f d))`, f in Hz, d in seconds.
6. Filter design: log magnitude, fold its cepstrum (two FFTs of 65536), keep the first 16384 taps with a short fade.

Tone Match needs steps 1 and 6 and a running-mean smoother. The model, if wanted, is 11 small matrix layers on a 14 x 240 input.