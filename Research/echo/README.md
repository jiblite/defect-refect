# The echo engine: Python side

`echo2.py` is the prototype that `Source/Core/Echo.h` is a port of. `Tools/echo_parity.py` compares the two on
whole sounds: the same lags stand out, everything measured about them agrees to six digits or better, the rule
gives the same answer, and the rendered sounds agree to -90 dB or better (seven sounds, 0.4.2). The scripts here are a record of how the engine was built and judged,
as they were run: paths point to the working folder they ran in, and the audio is not in the repo.

## The method
An echo is the sound plus copies of itself: `y = x * h`, with `h` = 1, then a spike at the echo time, then (with
feedback) at its multiples. In the log spectrum that is a sum: `log |Y| = log |X| + log |H|`, and `log |H|` is a
ripple along frequency whose period is one over the echo time. The spectrum of that ripple (the cepstrum of the
sound) therefore has a narrow peak at the echo time and at its multiples, on top of whatever the sound itself puts
there.

1. `cep` the cepstrum of up to 19 s of a channel: log power spectrum, minus its own trend.
2. `candidates` lags from 30 ms to 2.5 s that stand out of the cepstrum's background (median-based noise level).
3. `measure` everything about the most echo-like lags (below), and the echo's part of the cepstrum: short windows
   (4 ms) around the lag and its multiples, with what does not stand out of the background set to zero.
4. Those windows ARE the logarithm of the echo filter (a minimum-phase filter's log spectrum is the transform of
   the causal part of its cepstrum). So `exp(+transform)` is the echo filter, `exp(-transform)` its exact inverse:
   - taking the echo out = filtering the sound with the inverse (`apply`, own sound, Amount 0),
   - putting it on another sound = filtering that sound with the filter itself,
   - everything between and beyond is the same filter with the echoes scaled: `1 + s * (H - 1)`.
   Nothing is fitted to a model of a delay effect, so a repeat with its own tone, a flipped repeat or uneven
   repeats come out as they are.
5. The faders work on that filter (`Echo.h`, `responseOn`): Time stretches the lags, Feedback scales the repeats one
   by one (fewer) or continues the train (more), Brightness is a shelf on the echoes alone.

## Is it an echo? (the rule)
A loop, a steady rhythm or a repeated note also put a peak into the cepstrum. What separates them from an echo is
measured per lag and a small boosted-tree model decides (`fit*.py`, exported to `Source/Core/EchoRule.h` by
`export_rule.py`; the header is checked against the model tree by tree):
- how far the peak stands out, how loud the copy is, how loud the next copy is against the first;
- what taking it out does: an echo only ever adds level, so removing a real one takes level away everywhere
  (`removed`) and adds almost none (`added`);
- whether the ripple is there in every band (`even`, `bands`), how many other lags stand out as well;
- the copies themselves (`pairs`): cells of the spectrogram that repeat after exactly this lag. For an echo
  their level difference sits at the echo's own level (a real echo is quieter by a fixed amount, moment for
  moment), a loop's copies are as loud as the original, and a chance likeness is spread out. Measured: the share of
  cells with a copy, the median level difference and its spread, how often the copy is louder, how many cells there
  are, how much of the sound they carry, and how many of them carry the weight.

The model's answer is a probability for the lag. The app turns it into "how sure is it that this sound has an
echo" (`Echo.h`, `sureness`: fewer sounds carry an echo than the lags of the training set) and the Sureness slider
compares that number: the default, 65 %, asks for p = 0.90.

The rule in the app is the fifth round's (`fit5.py`, variant b: 300 trees of depth 3 over 19 measures). Rounds 4
and 5 are fitted on what the app's own engine measures (`drecho measure`), on 6 s clips and on whole sounds:
clean, with a synthetic echo at any time (round 4), with an echo in time with the sound's own repeats (round 5:
a delay on rhythmic material is nearly always set to the tempo, and then its echoes fall where the sound repeats
anyway), and the wet sounds of every other effect family as examples of "no echo". The corpus sounds marked
"test", and everything made from them, are kept out of the fit and only judged.

## Files
- `echo2.py` the prototype: `measure`, `feats`, `find`, `apply`.
- `mkdata2.py`, `fit3.py` the rule's third round: 1057 clean 6 s clips with and without a synthetic echo, measured
  by the prototype (the rule of the first echo engine in the repo; its model is not kept).
- `mkdata4.py`, `fit4.py` the fourth round: measured by the app's own engine (`drecho measure`) on the clips and on
  whole sounds (clean, with synthetic echoes, and the wet sounds of every other effect family as examples of
  "no echo"), fitted on one half of the sources and judged on the other.
- `mkdata5.py`, `fit5.py` the fifth round: echoes in time with the sound added (two per whole sound that has a
  repeat of its own, at a simple fraction of its period), five variants of the fit judged side by side, and the
  threshold for a second echo time on the other side (`pOwn`). `rule5.pkl` is the model in the app.
- `stats2.py`, `judge3.py` the prototype and rule 3 on whole sounds; `statscpp.py`, `judgecpp.py` the same for the
  app's engine: per effect family, on how many sounds a DELAY group would show (rounds 4 and 5 replace them).
- `evcpp.py` the app's engine scored on the scoreboard's delay sounds (echoes out, echoes onto the other sound),
  as the app shows it at the default Sureness and with every echo the engine has.
- `fadertest.py`, `owntest.py` the faders checked on the filter they make and on the sound itself.
- `roomcheck.py` the room engine's verdict on sounds with an echo, before and after the echo is taken out.
- `export_rule.py` writes `EchoRule.h` from a fitted model.

## Numbers
The rule, on the sources kept out of the fit (`fit5.py`; the Sureness slider at its default, 65 %):

|                                                    | round 3 | round 4 | round 5 (in the app) |
|----------------------------------------------------|---------|---------|----------------------|
| 83 clean whole sounds: an echo shown               | 1       | 1       | 0                    |
| 176 sounds with another effect: an echo shown      | 5       | 1       | 1                    |
| 249 whole sounds with an echo: found               | 76 %    | 80 %    | 84 %                 |
| 148 whole sounds with an echo in time: found       | 74 %    | 76 %    | 79 %                 |
| .. found with an echo time that was not put on     | 1       | 1       | 1                    |
| the scoreboard's 16 delay sounds: shown            | 15      | 14      | 14                   |

Quiet echoes are the hard part: with the echo under 0.3 of the sound's level 57 % are found, in time with the
sound 61 %; from 0.3 up 89 to 91 % and 78 to 87 %. At slider 50 % round 5 finds 87 % and 82 % and shows an echo on
1 of the 83 clean sounds and on 3 of the 176; at 80 % 78 % and 73 %, on none and on 1. The one sound of the 176 is
a ring modulator on a guitar loop that has a quieter copy of itself exactly 1 s later in the sample.

The scoreboard's 8 delay cases (16 sounds through delay effects that are not this project's code; `evcpp.py`):
- at the default Sureness an echo is shown on 14 of the 16 sounds, none with a wrong echo time. The two others
  are one case's two sounds, with an echo at a tenth and at a fifth of the sound's level; round 3 showed the
  second of them, rounds 4 and 5 do not, and round 5 is the rule in the app because of the table above. Both come
  up when Sureness is lowered (to 34 % and to 21 %).
- echoes taken out: 89 of 100 for the typical sound (median; mean 78 with the two not shown counted as 0, 85 with
  Sureness lowered); put onto the case's other sound: 95 (means 83 and 90).
- The whole app on the 8 cases (`Research/room/bench/famboard.py`: sound A's echoes out, and onto sound B): the
  DELAY group shows on 7, out 89 (median; mean 77), onto 95 (mean 83), no other group shown on any of them.
  Build 0.4.1 (the detector with a fitted delay): shown on 6, out 54, onto 50, and 2.8 other groups per case.
- The filter is exact where it applies: Amount in the middle leaves the sound untouched (-344 dB), and the echoes
  written by Export IR, used in a convolution, equal the app's own result to -153 dB.

A room and an echo together (`roomcheck.py`, `roomcheck2.py`): a train of echoes dies away as a room does, and the
room engine showed a REVERB group on 5 of the 16 delay sounds, one of which has a room of its own. With the room
looked at again after the echoes are taken out (`recheckRoom` in `Estimate.h`, the less sure verdict counts) 3 of the
4 false ones go and none is added. The other way round it does no harm: of 45 reverb sounds on which the echo engine
finds a reflection and the REVERB group shows, 44 keep it, and with a synthetic echo on top of 14 reverb sounds the
verdict after the second look is the verdict the sound gets without the echo (10 shown before, 11 after).

## Known limits
- One echo time per channel (with its repeats). A second, unrelated echo time on the same channel is not found.
- An echo that crosses sides (left's sound repeated on the right) on a wide stereo sound is not an echo of the
  channel it lands on and is not found; on a mono or narrow sound it is.
- Echo times under 30 ms belong to the tone engine (COMB FILTER, up to 15 ms); between 15 and 30 ms neither looks.
- A quiet echo (a fifth of the sound's level or less) on a busy sound is often not shown at the default Sureness
  rather than guessed at; under about a tenth it is mostly missed at any setting.
- A loop with a moving pan or tremolo on it can read as a quieter copy of itself.
