# The ring engine: Python side

`ring.py` is the prototype that `Source/Core/Ring.h` is a port of. `Tools/ring_parity.py` compares the two on
whole sounds (through `drring`, built from `Tests/ring.cpp`): on 10 pairs (the 8 scoreboard pairs of the RING MOD
family and two sounds without a tone) the verdicts and every number of the rule are the same, the rendered
sounds agree to -149 dB, and Amount in the middle gives the sound back bit for bit. The other scripts are a
record of how the engine was built and judged, as they were run: paths point to the working folder they ran in,
and no audio is in the repo.

## What is looked for
A sound multiplied by a steady tone, the same on both sides:

    wet = dry * B (rho + sin(w t + p)),    B = 1 / sqrt(rho^2 + 1/2)

rho is how much plain sound stays beside the tone part (0: a pure ring modulation; 1: the gain just touches
zero, as with a mix of one half; over 1: it never reaches zero). Read off the sound itself, without the dry
sound: the tone's pitch (20 to 5000 Hz), its phase, and rho.

## The method
1. **Candidates.** The sound's power over time, divided by its own local mean, carries a line at w that is
   rho / (rho^2 + 1/2) of its mean and one at 2w that is 1/4 / (rho^2 + 1/2) of it: for every rho up to 2 one of
   the two is 0.44 or more. The highest lines of that spectrum are tried (0.2 of the mean or more, one per 5 Hz;
   six from 80 Hz up with a local mean of 21 ms, four from 20 to 80 Hz with one of 171 ms), each as the tone's
   own line and as the line at twice the tone. Dividing by the local mean is what makes this work on a drum loop
   or a row of short strokes, whose own loudness otherwise hides the lines among their neighbours.
2. **First look**, on little data: the power is sorted by where in the tone's cycle it falls (64 steps) and by
   where in the sound (8 stretches). The gain that explains that table best is looked for, the sound taken as
   noise whose level is the gain (a likelihood). The candidate that wins the most over "no gain at all" goes on.
3. **The fit**: the same likelihood on the samples themselves, by Newton's method, with the floor under the gain
   (what the sound holds where the gain is zero) fitted along with it. The floor comes down a step at a time
   (0.05 down to 1.6e-5, eight steps) and is only then set free: under a low floor every sample next to a zero of
   the gain digs a narrow pit of its own, and a fit that lets the floor fall at once can end on the wrong side of
   one. Where the cost does not curve upwards in every direction the step is made with the curvature it would
   have on average (always downhill). Near the gain's zeros this pins the tone to a thousandth of a sample.
   If rho comes out over 1 there is no floor to measure: it is held at 1e-3 and the rest fitted again.
4. **The level** against which the power is measured is its own mean over eight of the tone's cycles (a power of
   two, 21 ms at least). For a tone under 94 Hz that would be 170 or 340 ms, longer than a drum stroke: there
   it is a whole number of the tone's cycles instead, 85 ms at least. Samples count if the sound around them is
   within 80 dB of its loudest moment.

**The rule.** The power sorted by the tone's cycle must have the gain's own shape (1 dB at most apart), the gain
must dip by 10 dB or more (a shallower one is a tremolo), the sound divided by the gain must not carry a steady
offset (-20 dB at most: a plain note at the tone's pitch is a "ring modulated" constant), the win over "no gain"
must be 0.3 or more, and at most a tenth of the samples may lie where the gain is next to zero.

**Taking out.** Where the gain is far from zero the sound is divided by it. Next to a zero a sample says little
(it is the sound times next to nothing), but the sound's own course says where it has to go: there each sample
is the value that fits best with both, the sample over the gain counted by (gain / floor)^2 and the linear
prediction of the sound around it (order 24) counted by how well that sound can be predicted at all. Samples
that do not fit after that count less in two more rounds.

**The faders.** Amount: left of the middle a straight line from no gain to the found one, right of it the tone
part grows; the middle is the sound itself. Carrier: the tone's pitch, an octave each way. Onto a second sound:
multiplied by the gain.

## Results
All with the app's own code unless said otherwise.

**Fresh cases** (`rt7.py`: 16 draws for each of four outside effects and of a plain ring modulation, on any clip
of the corpus, 80 cases):

| effect | found | pitch off by at most | taken out (median, least) |
|---|---|---|---|
| four outside tools | 64 of 64 | 0.0009 Hz | 99.9, 95.3 |
| plain | 16 of 16 | 0.0014 Hz | 99.9, 91.2 |

**Every clip of the corpus with a tone put on, twice** (`rk2.py`: 396 cases, tones from 25 to 4500 Hz, once with
rho = 0 and once with rho between 0.2 and 1.4): found on 395. The one not found is a spoken sentence with a tone
at the voice's own pitch (the shape is 1.09 dB apart, the rule allows 1).

| | cases | found | taken out (median, least) | left beside the dry sound (median, worst) |
|---|---|---|---|---|
| rho = 0 | 198 | 198 | 99.9, 99.2 | -68 dB, -45 dB |
| rho up to 1 | 131 | 130 | 99.9, 98.9 | -67 dB, -44 dB |
| rho over 1 | 67 | 67 | 98.7, 79.4 | -46 dB, -18 dB |

**Tones with a round pitch** (`rf1.py`, prototype: 40 cases whose cycle is a whole number of samples, many with
zeros exactly on samples): all 40 end at the best fit; left beside the dry sound -55 dB in the middle case,
-32 dB at worst (samples that sit exactly on a zero of the gain are lost).

**The tone among the candidates** (`rk1.py`): in 516 of 516 cases (the 80 fresh ones, the 40 round ones, the 396).

**Wrong finds** (`afind.py`): none on the 198 clean clips (37 of them give a candidate line at all, none gets
past the first look), none on the 384 scoreboard sounds of the other families; all 16 of the RING MOD family
found.

**Scoreboard** (8 pairs): found 8 of 8; taken out 100 in the middle case (seven at 99.6 or more, one at 79.5,
a row of bass drum strokes with rho 1.2); onto a second sound 98.5 to 100. Before the engine: found 8 of 8 by the
detector, taken out 0, onto below zero.

**Long sounds**: a two-minute sound with a tone all through is found in 5.4 s; with a tone on its first part
only, not found (4.6 s); clean: nothing (0.6 s). A 6 s sound takes about a second when a tone is found and
0.2 s when there is no candidate.

**A tone and an echo on one sound** (`rc1.py`, 16 clips, both orders, several levels: 112 sounds). When both are
found the echo engine looks at the sound without the tone as well (`recheckEchoWithout` in Estimate.h): the same
echo time found there, the less sure verdict counts; not found there, the echo was put on after the tone and its
verdict stays. On these 112 the DELAY group neither comes nor goes by it; on the scoreboard one DELAY group on a
sound with a tone only (a loop that repeats every half second) goes. With an echo put on after the tone, the
tone is still found on 6 to 8 of 16 clips.

## What is not done
- **rho over 1** (the gain never reaches zero) is measured less exactly: nothing pins it but how deep the
  power's cycle is, and the sound's own course at that pitch counts into it (rho off by 1 to 10 %).
- With **rho = 0** the sound may come out with its sign turned: which half of the cycle is "up" cannot be known.
- **Zeros exactly on samples** (a round pitch at phase 0): those samples are filled from their neighbours.
- A tone on **part of a sound** is not found (it fails safe), a tone whose **pitch moves** is not followed, and a
  tone under an echo or a reverb is found about half the time.
- An echo that the echo engine sees clearly only without the tone stays hidden: the echoes that are worked with
  are the ones measured on the sound as it is.
- Tried and not kept: counting only samples within 50 dB of the loudest moment (loses slow tones on short
  strokes), and a whole number of cycles for the level at every pitch (`rw1.py`, `rw2.py`: better under 94 Hz,
  worse where the tone is near the sound's own pitch).

## Files
`ring.py` the prototype. `rt7.py` fresh cases, `rk2.py` every clip, `rf1.py` round pitches, `rk1.py` candidates,
`afind.py` wrong finds, `rc1.py` a tone and an echo, `rw1.py` / `rw2.py` the level window; `ring_all2.sh` runs
the checks in a row. `rt1.py`, `rt2.py`, `rt_report.py` are the first study (scoreboard pairs, prototype only).
