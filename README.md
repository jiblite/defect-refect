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

Effects it knows so far (19): flanger, chorus, phaser, Haas width, stereo width, comb filter, delay, reverb,
distortion, bitcrush, filter sweep, tremolo, auto pan; added in 0.2.0: saturation, exciter, ring mod,
filter, stutter, gated rhythm (from a larger 27-effect detector, named only above 80%).

## Layout
- `Source/Core/` plain C++17, no JUCE: features, detector (small CNN), hand checks, tree models,
  effects, settings reading, removal controls, the picture for each effect (`Display.h`).
  The reverb engine: `Room.h` (what the model sees, the model, rendering), `RoomNumbers.h` (the room in
  numbers), `RoomEngine.h` (the one file to include: find, impulse response, room onto a sound, reshaping).
  The tone engine: `Tone.h` (long-term spectra, the finder of short reflections, Tone Match, a curve as a
  minimum-phase filter and as an impulse response).
- `Source/App/Main.cpp` the JUCE app.
- `Resources/models.bin` (12 effects) and `models27.bin` (27 effects, `DR_MODEL27=1`) the trained models, written by `Tools/export_models.py`.
- `Resources/room.bin` and `room2.bin` the reverb engine's model (first and second look), written by `Tools/export_room_model.py`.
- `Tests/render.cpp` the app's processing without the UI: `drrender Resources first.wav [second.wav|-] out.wav [id=value ...]`
  (out `-` only prints what was found; `ir=file.wav` writes what Export IR writes; `match=1` switches Tone Match on;
  `-- [second.wav|-] out.wav [id=value ...]` adds another render of the same first sound, which is analysed once).
- `Tests/room.cpp` the reverb engine alone: `drroom room.bin in.wav ...` (see the top of the file).
- `Tests/tone.cpp` the tone engine alone: `drtone a.wav b.wav prefix [detail]`; `Tools/tone_parity.py` compares it with the prototype.
- `Research/room/` the Python side of the reverb engine as it was run: the scoreboard, the reference engine the
  C++ matches, training and the experiments (see its README).
- `Research/tone/` the Python side of the tone engine: the prototype the C++ matches, its scores and the report.

## Build
`cmake -S . -B build && cmake --build build --config Release --target DefectRefect`
(Windows exe built by GitHub Actions.)
