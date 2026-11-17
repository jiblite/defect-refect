# Defect Refect

Standalone app. Drop a sound: the effects found on it become faders, grouped by effect.

- **One sound:** middle = as found, left = take the effect out (best effort), right = double.
- **Two sounds:** the first sound's effects go onto the second. Left = not applied, right = double.
- Swap, live playback (space bar), Result/Original compare, Export (24-bit 48 kHz WAV).
- Click an effect or move one of its faders: the display on the right shows what was found
  (frequency curve, decay, echoes, transfer curve...), with the found state faint behind it once a fader moves.
- Every effect has an On/Off switch. Off = left alone: with one sound it stays as found, with two sounds it is not
  put on the second. For comparing with and without, and for an effect that is shown but not there.

**Reverb (new engine in 0.4.0, second model in 0.4.1):** a small model looks at the sound twice and says, band by
band and moment by moment, what is room and what is the sound itself. From that the app reads the room (decay time
and level per band, width) and the REVERB group works on it:
- one sound: Amount takes the reverb out (half way left = the engine's own answer, far left = bolder) or adds
  more of it; Length, Brightness and Width reshape the sound's own reverb;
- two sounds: the found room, rebuilt from its numbers, goes onto the second sound;
- the display shows how the room dies away (lows, mids, highs); **Export IR** saves it as an impulse response.

Whether the REVERB group shows at all follows how sure the engine is that there is a room (the Sureness slider;
the rule is in `Estimate.h`, `roomSureness`). The first model told room from sound largely by what left and right
do not share. The model in 0.4.1 was also trained on recordings that are narrow or mono as a whole and on wide
sounds under the reverb: on narrow and mono recordings it now finds the room about as often as on stereo ones and
takes out about twice as much as before. Known limits: with a wide stereo sound under the reverb it still takes
less out and errs more often than with a mono or narrow one, and a light reverb is the hardest case.

Since 0.4.2 a found room is looked at a second time with three seconds of silence after the sound, and the less
sure verdict counts. The model learned from sounds that end inside the file, where a room shows as the tail after
the last note; on a sound that is cut off while it is still playing it read a room into 45 % of dry sounds (15 %
when the sound ends inside the file). With the second look that is 15 % for both. Sounds with a natural ending
lose nothing; sounds with a real reverb that are cut off pay for it (the room shows on 57 % of them instead of
73 %; with the tail ringing out 87 %). What it still shows on dry sounds is mostly sample loops, many of which
carry a room of their own.

**Tone (new engine in 0.4.1):**
- **Tone Match** (two sounds, the Tone Match button): the second sound takes the long-term tone of the first.
  Amount and Detail faders, the curve as a picture, and **Export IR** saves it as an impulse response. It comes
  after the other effects, so the result has the first sound's tone whatever went on before. When the two sounds
  are versions of one recording (the app checks whether one is a filtered copy of the other, also when the two
  files do not start at the same moment), the curve is exactly what lies between them, for example an EQ: on the
  test sounds that brings 98 to 99 % of an EQ or filter across. It is off until switched on, because on two
  unrelated sounds it changes the second one a lot.
- **COMB FILTER** (found on one sound): a short reflection, a copy of the sound 0.5 to 15 ms later, which colours
  the tone with evenly spaced notches. Amount takes it out or doubles it, Delay moves it, and with two sounds it
  goes onto the second. On the test sounds it is found on every sound that has one and on none of the others,
  and taken out to 94 %.
- What it does not do: find a plain EQ on one sound alone. From the outside an EQ and the sound's own colour
  look the same; the numbers are in `Research/tone/REPORT.md`.

**Echoes (new engine in 0.4.2):** an echo is the sound plus later copies of itself, and that leaves a regular
ripple along the sound's spectrum. The engine reads the echoes straight from that ripple (echo times from 30 ms to
2.5 s, one per side, with its repeats) instead of fitting a delay effect to the sound, so a repeat with its own
tone, a flipped repeat or uneven repeats come out as they are. The DELAY group works on what was read:
- one sound: Amount takes the echoes out (far left = the exact inverse of what was found) or makes them louder,
  Time moves them (half to double), Feedback shortens the train of repeats or continues it, Brightness changes
  the tone of the echoes alone;
- two sounds: the found echoes go onto the second sound;
- the display shows the echoes in time; **Export IR** saves the echoes alone as an impulse response (wet and dry
  at the same level give the echoes as found).

On the scoreboard's eight cases with a delay from outside effects the app shows the DELAY group on 7 at the
default Sureness and never with a wrong echo time, takes 89 of 100 out and puts 95 of 100 onto the case's other
sound (medians; build 0.4.1: 6 of 8, 54 and 50). No other group is shown on these cases, where 0.4.1 showed
nearly three per case. Whether an echo is shown is decided by a rule fitted on what the engine measures on about
3900 sounds; on sounds kept out of that fit it shows an echo on none of 83 clean sounds and on 1 of 176 sounds
with another effect, and finds 84 % of the echoes put on (79 % of those that fall in time with the sound's own
rhythm).

When echoes and a room are both found, the room is looked at again on the sound with the echoes taken out and
the less sure verdict counts, because a train of echoes dies away as a room does (`Estimate.h`, `recheckRoom`).
Known limits: one echo time per side; under 30 ms a copy belongs to the comb filter (which looks up to 15 ms;
between 15 and 30 ms nothing looks); a quiet echo (a fifth of the sound's level or less) on a busy sound is
often not shown at the default Sureness and comes up when Sureness is lowered; an echo that crosses sides on a
wide stereo sound is not found. Method, numbers and limits: `Research/echo/README.md`.

**Level movement (new engine in 0.4.3):** tremolo, auto pan and a gated rhythm are one thing: a gain that moves
over time and repeats, on both sides together or on left against right. The engine measures that gain on the
sound itself (how fast, the shape of one cycle for each side, how deep, a gate's steps to a fraction of a
millisecond) and the TREMOLO, AUTO PAN and GATED RHYTHM groups work on it:
- one sound: Amount takes the movement out (the sound is lifted by what the gain took) or deepens it, Speed
  makes it slower or faster (half to double);
- two sounds: the movement goes onto the second sound in the first sound's beat, at the times at which the first
  sound has it;
- the display shows the gain over two cycles, and says from when to when the movement is there when it is not on
  all of the sound.

A gain that was put on the sound is told from the sound's own rhythm by two things: it moves all bands by the
same amount at the same moment, and it repeats exactly. Notes and hits do neither. A sound of 12 s or more is
measured piece by piece, and every stretch of about a second is asked whether it does what would be taken out, so
an effect that is on a part of the sound is taken out of that part and the rest is left exactly as it was.

On the scoreboard's 24 cases with a tremolo, an auto pan or a gate from outside effects (sounds of 6 s; medians, a
case that is not shown counts 0; build 0.4.2 in brackets): AUTO PAN is shown on 8 of 8, taken out at 96 of 100
and put onto the case's other sound at 98 (64 and 71); TREMOLO on 5 of 8, 85 and 78 (41 and 44); GATED RHYTHM on
6 of 8, 60 and 79 (0 and -18). No case comes out worse than it went in, and on the 198 clean sounds of the corpus
none of the three groups is shown.

Long sounds (20 to 62 s, test clips laid end to end; `Research/level/long.py`), at the default Sureness:
- the effect on the whole sound: shown on 81 of 88, taken out at 88 (tremolo), 94 (auto pan) and 71 (gate) in the
  median; one sound comes out a little worse than it went in (a tremolo, -4);
- the effect on one half: shown on 36 of 44 and taken out at 77, none worse than it went in, and the half without
  the effect comes back unchanged in 31 of the 36 (the difference to the input is under -170 dB; in the other
  five the border sits a little off and it is at -73 to -35 dB);
- slow pans, 3 to 23 s a cycle: shown on 21 of 24 and taken out at 94; the other three pass about once in their
  sound and are offered at a lower Sureness (see the limits);
- 48 clean long sounds: nothing shown.
TODO-S1

Known limits: a tremolo of a few dB on a busy sound and a sawtooth that falls (a level that jumps up and sinks
back is what struck notes do) are not shown at the default Sureness; a movement from side to side that the sound
shows only once is offered at a lower Sureness, because something that passes by does the same; one movement per
sound; nothing faster than 16.7 times a second; a gate that closes fully is left closed (what is under it is as
likely noise); a tremolo whose speed wanders is taken out less well (67 where shown) and goes onto a second
sound poorly. Method, numbers and limits: `Research/level/README.md`.

**Flanger (new engine in 0.4.3):** a flanger is the sound plus a copy of itself whose delay moves. The copy
leaves a line in the picture of the sound's short-time cepstrum, at its delay, as high as its gain. The engine
follows that line (delays of 0.2 to 16 ms), reads the copy's gain and feedback, and the FLANGER group works on
what was read:
- one sound: Amount takes the copy out with the exact inverse of the moving filter, or makes it louder; Speed,
  Sweep and Feedback change the movement, how far the delay travels and how much it rings;
- two sounds: the same moving filter goes onto the second sound, moving as it does on the first;
- the display shows the copy's delay over time, and says from when to when the copy is there when it is not on
  all of the sound (a line under the waveform shows it too).

A line is followed through the whole sound, also through a stretch that has no flanger, where it would hold on
to whatever the picture has. So every half second is asked two things the rule asks of the whole sound: is there
a moving line (a copy's line is smooth; a made-up one wanders), and does taking the copy out lower the level by
what a copy adds? The inverse filter works only where the answer is yes: a flanger that comes in halfway is taken
out from there, and the rest of the sound is left exactly as it was. A sound of 24 s or more is measured in
pieces of 12 to 24 s. Onto a second sound the flanger goes all the way.

On the scoreboard's eight cases with a flanger from outside effects (sounds of 6 s) the app shows the FLANGER
group on 4 at the default Sureness (build 0.4.2: none), takes 91, 89, 96 and 39 of 100 out and puts 91, 91, 94
and 27 onto the case's other sound. Not shown: the two thru-zero cases, a setting whose delay moves by 4 samples
only, and a light copy (a quarter of the sound's level). On the 198 clean sounds of the corpus it shows a flanger
on none. Long sounds (33 to 62 s, 24 a set): no flanger on 24 clean ones; on 24 flanged from end to end it is
shown on 3 (93, 85 and 30 of 100 taken out) and found on 3 more when Sureness is lowered, and none comes out
worse than it went in; on 24 flanged in one half it is shown on 4 (65, 75, 63 and -5 taken out), and the clean
half comes back as it went in. A piece of a long flanged sound shows the copy in 16 % of the cases, so on long
sounds a flanger is found less often than on short ones.

Known limits: a light flanger on a busy sound is found on some sounds and not on others; thru-zero flangers and
flangers whose delay stays under 1.3 ms are not found; chorus, phaser and filter sweep have no engine yet (the
flanger engine finds nothing on them, as it should). Method, numbers and limits: `Research/sweep/README.md`.

**What is shown (new in 0.4.2):** up to 0.4.1 an effect was shown as soon as one 3 s window of the sound scored
above the cut. Measured with the app's own code that showed 1.6 effects per sound that are not there. Now every
effect is checked against the whole sound before it is shown (`Verify.h`): per effect a small model reads what the
two detectors computed over the sound together with 55 physical measurements. On sounds it was not tuned on
(the 17 effects without an engine of their own; "clean" = dry sounds and sounds with only a reverb or a tone change):

| held-out sounds                               | up to 0.4.1 | 0.4.2 |
|-----------------------------------------------|-------------|-------|
| clean sounds with none of these effects shown | 12 %        | 93 %  |
| effects shown per clean sound                 | 1.61        | 0.08  |
| effects found (17 effects from outside tools) | 50 %        | 46 %  |
| other effects shown per sound with an effect  | 1.07        | 0.09  |

And the whole app on the scoreboard (`Research/room/bench/famboard.py`; the drop in REVERB is the second look
described above):

| 136 cases, 17 effects from outside tools                        | 0.4.1    | 0.4.2   | 0.4.3   |
|------------------------------------------------------------------|----------|---------|---------|
| the case's own effect shown                                      | 77       | 75      | 79      |
| other groups shown: REVERB / the rest                            | 63 / 156 | 19 / 12 | 19 / 10 |
| cases with no other group shown                                  | 16       | 107     | 109     |
| own group far left: cases made worse than doing nothing          | 12 %     | 3 %     | 3 %     |
| every shown group far left: cases made worse than doing nothing  | 78 %     | 19 %    | 19 %    |
| everything shown put onto a second sound: cases made worse       | 71 %     | 36 %    | 30 %    |

What changed from 0.4.2 to 0.4.3 besides the scores of the four reworked groups: four flangers are found; a
tremolo that 0.4.2 took for a gate is shown as a tremolo; another tremolo, which 0.4.2 showed and took out at 24
of 100, is now offered only at a lower Sureness; FLANGER on a chorus and AUTO PAN on a phaser are no longer shown.
New and wrong: TREMOLO, just over the default Sureness, on a bass loop with a low shelf. Every other case is as
it was.

The price is in the third row of the first table, and it is uneven: flanger, phaser, filter sweep and a fixed filter are found less
often than before (9, 39, 39 and 9 % against 36, 73, 61 and 34 %), because most of their earlier finds came with
as many false ones; ring mod and stutter are found far more often (82 and 93 % against 23 and 52 %). The
Sureness slider means the same for every effect now: shown when the score is above the slider. At 30 % more is
shown (63 % found, 67 % of clean sounds clean), at 80 % less (39 %, 98 %). A long sound is judged in blocks of 6 s
and an effect has to be there in half of them, so an effect on a short part of a long sound is not shown (the
level and flanger engines of 0.4.3 look for themselves where on the sound their effect is). Reverb, the comb
filter, the delay, tremolo, auto pan, gated rhythm and the flanger keep their own engines' verdicts. Method and tables: `Research/detect/README.md`.

Effects it knows so far (19): flanger, chorus, phaser, Haas width, stereo width, comb filter, delay, reverb,
distortion, bitcrush, filter sweep, tremolo, auto pan; added in 0.2.0: saturation, exciter, ring mod,
filter, stutter, gated rhythm (from a larger 27-effect detector).

## Layout
- `Source/Core/` plain C++17, no JUCE: features, detector (small CNN), hand checks, tree models,
  effects, settings reading, removal controls, the picture for each effect (`Display.h`).
  The reverb engine: `Room.h` (what the model sees, the model, rendering), `RoomNumbers.h` (the room in
  numbers), `RoomEngine.h` (the one file to include: find, impulse response, room onto a sound, reshaping).
  The tone engine: `Tone.h` (long-term spectra, the finder of short reflections, Tone Match, a curve as a
  minimum-phase filter and as an impulse response).
  The echo engine: `Echo.h` (the ripple picture, what is measured about an echo time, the echo filter and its
  faders), `EchoRule.h` (the fitted rule "is this an echo", written by `Research/echo/export_rule.py`).
  The level engine: `Level.h` (what is measured about a gain that moves and repeats, the fitted rule, a gate's
  steps, long sounds piece by piece, where the movement is, the gain for taking out and putting on).
  The flanger engine: `Sweep.h` (the cepstrum picture, the line through it, gain and feedback, the rule, where
  along the sound the copy is, the moving filter and its inverse, long sounds piece by piece).
  `Verify.h` the check of every effect against the whole sound before it is shown.
- `Source/App/Main.cpp` the JUCE app.
- `Resources/models.bin` (12 effects) and `models27.bin` (27 effects, `DR_MODEL27=1`) the trained models, written by `Tools/export_models.py`.
- `Resources/room.bin` and `room2.bin` the reverb engine's model (first and second look), written by `Tools/export_room_model.py`.
- `Resources/verify.bin` the verifier's models, written by `Research/detect/export_verify.py`.
- `Tests/render.cpp` the app's processing without the UI: `drrender Resources first.wav [second.wav|-] out.wav [id=value ...]`
  (out `-` only prints what was found; `ir=file.wav` writes what Export IR writes, `echoir=file.wav` the same for
  the DELAY group; `match=1` switches Tone Match on;
  `-- [second.wav|-] out.wav [id=value ...]` adds another render of the same first sound, which is analysed once).
- `Tests/room.cpp` the reverb engine alone: `drroom room.bin in.wav ...` (see the top of the file).
- `Tests/tone.cpp` the tone engine alone: `drtone a.wav b.wav prefix [detail]`; `Tools/tone_parity.py` compares it with the prototype.
- `Tests/echo.cpp` the echo engine alone: `drecho measure|out|onto|ir ...` (see the top of the file);
  `Tools/echo_parity.py` compares it with the prototype.
- `Tests/verify.cpp` the verification step alone: `drverify Resources sound.wav ...`; `Tools/verify_parity.py`
  compares it with the Python side.
- `Tests/level.cpp` the level engine alone: `drlevel measure|out|onto|pieces|own|put|both ...` (see the top of the
  file); `Tools/level_parity.py` compares it with the prototype.
- `Tests/sweep.cpp` the flanger engine alone: `drsweep ...` (see the top of the file); `Tools/sweep_parity.py`
  compares it with the prototype.
- `Research/room/` the Python side of the reverb engine as it was run: the scoreboard, the reference engine the
  C++ matches, training and the experiments (see its README).
- `Research/tone/` the Python side of the tone engine: the prototype the C++ matches, its scores and the report.
- `Research/echo/` the Python side of the echo engine: the prototype the C++ matches, the data and the fit of
  its rule, the scores.
- `Research/detect/` the Python side of the verification step: what was wrong with the detection, the sounds and
  features, the training, the tables.
- `Research/level/` the Python side of the level engine: the prototype the C++ matches, the data and the fit of
  its rule, the tests on long sounds.
- `Research/sweep/` the Python side of the flanger engine: the prototype the C++ matches, the numbers behind its
  rule, the ceilings, the tests on long sounds.

## Build
`cmake -S . -B build && cmake --build build --config Release --target DefectRefect`
(Windows exe built by GitHub Actions.)
