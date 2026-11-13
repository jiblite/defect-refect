# The tone engine: Python side

`tone.py` is the prototype that `Source/Core/Tone.h` is a port of (its default path and Tone Match).
`Tools/tone_parity.py` compares the two on pairs of sounds: the curves agree to a billionth of a dB, the rendered
sounds to -152 dB.

- `REPORT.md` the full report: what a curve can reach at all, what was tried for a blind estimate, all scores.
- `tone_eval.py`, `tone_eval.json` the scores on the scoreboard's 64 tone cases (`--show` prints the tables again).
- `tone_model.npz` the optional model for a blind guess at an EQ. It is off by default in the prototype and not
  in the app: it gains little and changes about one clean sound in fourteen (report, section 3.3).
- `lab/` the experiments behind the report, as they were run. Paths point to the working folder they ran in, and
  the audio is not in the repo.

In short:
- With both versions of a sound, **Tone Match** recovers an EQ, filter, resonance or comb to 98 to 99 %.
- Blind, from one sound alone, only **short reflections** (comb filters) can be found: 94 % removed, nothing
  reported on any of the scoreboard's 184 sounds without one, 2 false reports on 1057 clean clips.
- Blind, a plain EQ cannot be told from the sound's own colour: on 6000 examples the sound's own spectral detail
  is 3 to 4 times larger than an EQ's at every scale, and the best fixed rule is "report nothing".
- A by-ear "flatten" (the long-term spectrum against its own one-octave trend, taken out as a curve) was measured
  as well and is not built: on 32 of the tone cases it moves the sound away from the true dry sound on 6 of the 8
  effect types (typical score -44 to -61 with the curve fully out, -12 to -19 half way) and changes a clean sound
  by more than a typical effect does (1.1 to 1.3 against 0.7 to 1.1). It evens out the sound's own colour, not
  the effect's.
- Two files of one recording often do not start together. `lagOf` in `Tone.h` finds the shift first (the peak of
  the cross-correlation with the spectrum evened out); without it the same-recording check fails from a few
  milliseconds on (coherence 0.04 instead of 1.0). With it, 62 scoreboard pairs moved by 6 to 600 ms all come out
  as before, and two different sounds stay at 0.18 or lower.
