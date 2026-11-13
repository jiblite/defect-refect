# Defect Refect

Standalone app. Drop a sound: the effects found on it become faders, grouped by effect.

- **One sound:** middle = as found, left = take the effect out (best effort), right = double.
- **Two sounds:** the first sound's effects go onto the second. Left = not applied, right = double.
- Swap, live playback (space bar), Result/Original compare, Export (24-bit 48 kHz WAV).
- Click an effect or move one of its faders: the display on the right shows what was found
  (frequency curve, decay, echoes, transfer curve...), with the found state faint behind it once a fader moves.

**Reverb (new engine in 0.4.0):** a small model looks at the sound twice and says, band by band and moment by
moment, what is room and what is the sound itself. From that the app reads the room (decay time and level per
band, width) and the REVERB group works on it:
- one sound: Amount takes the reverb out (half way left = the engine's own answer, far left = bolder) or adds
  more of it; Length, Brightness and Width reshape the sound's own reverb;
- two sounds: the found room, rebuilt from its numbers, goes onto the second sound;
- the display shows how the room dies away (lows, mids, highs); **Export IR** saves it as an impulse response.

Whether the REVERB group shows at all follows how sure the engine is that there is a room (the Sureness slider;
the rule is in `Estimate.h`, `roomSureness`). Known limits of this first model: it tells room from sound
largely by what left and right do not share. With a wide stereo sound under the reverb it takes less out and errs
more often than with a mono or narrow one. And on a recording that is narrow or mono as a whole, room included,
it finds the room less often (when it finds none there, the old reverb handling stays) and takes out only about
a third as much.

Effects it knows so far (18): flanger, chorus, phaser, Haas width, stereo width, delay, reverb,
distortion, bitcrush, filter sweep, tremolo, auto pan; added in 0.2.0: saturation, exciter, ring mod,
filter, stutter, gated rhythm (from a larger 27-effect detector, named only above 80%).

## Layout
- `Source/Core/` plain C++17, no JUCE: features, detector (small CNN), hand checks, tree models,
  effects, settings reading, removal controls, the picture for each effect (`Display.h`).
  The reverb engine: `Room.h` (what the model sees, the model, rendering), `RoomNumbers.h` (the room in
  numbers), `RoomEngine.h` (the one file to include: find, impulse response, room onto a sound, reshaping).
- `Source/App/Main.cpp` the JUCE app.
- `Resources/models.bin` (12 effects) and `models27.bin` (27 effects, `DR_MODEL27=1`) the trained models, written by `Tools/export_models.py`.
- `Resources/room.bin` and `room2.bin` the reverb engine's model (first and second look), written by `Tools/export_room_model.py`.
- `Tests/render.cpp` the app's processing without the UI: `drrender Resources first.wav [second.wav|-] out.wav [id=value ...]`
  (out `-` only prints what was found; `ir=file.wav` writes what Export IR writes).
- `Tests/room.cpp` the reverb engine alone: `drroom room.bin in.wav ...` (see the top of the file).
- `Research/room/` the Python side of the reverb engine as it was run: the scoreboard, the reference engine the
  C++ matches, training and the experiments (see its README).

## Build
`cmake -S . -B build && cmake --build build --config Release --target DefectRefect`
(Windows exe built by GitHub Actions.)
