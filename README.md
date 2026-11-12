# Defect Refect

Standalone app. Drop a sound: the effects found on it become faders, grouped by effect.

- **One sound:** middle = as found, left = take the effect out (best effort), right = double.
- **Two sounds:** the first sound's effects go onto the second. Left = not applied, right = double.
- Swap, live playback (space bar), Result/Original compare, Export (24-bit 48 kHz WAV).

Effects it knows so far (18): flanger, chorus, phaser, Haas width, stereo width, delay, reverb,
distortion, bitcrush, filter sweep, tremolo, auto pan; added in 0.2.0: saturation, exciter, ring mod,
filter, stutter, gated rhythm (from a larger 27-effect detector, named only above 80%).

## Layout
- `Source/Core/` plain C++17, no JUCE: features, detector (small CNN), hand checks, tree models,
  effects, settings reading, removal controls.
- `Source/App/Main.cpp` the JUCE app.
- `Resources/models.bin` (12 effects) and `models27.bin` (27 effects, `DR_MODEL27=1`) the trained models, written by `Tools/export_models.py`.
- `Tests/render.cpp` the processing without the UI: `render models.bin first.wav [second.wav|-] out.wav [id=value ...]`.

## Build
`cmake -S . -B build && cmake --build build --config Release --target DefectRefect`
(Windows exe built by GitHub Actions.)
