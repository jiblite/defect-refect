# The moving-tone engine (PHASER, FILTER SWEEP): Python side

`moving.py` (with `pic.py`, the band picture and the rendering, and `net.py`, the model) is the prototype that
`Source/Core/Moving.h` is a port of. `Tools/moving_parity.py` compares the two on whole sounds: on the 32 phaser and
filter sweep sounds of the scoreboard the answers are the same (found, kind, stretches, how often the curve was
read), the curve agrees to 5e-5 dB and the rendered sounds to -125 dB or better, with one exception at -72 dB (a
second reading in which a cell at the edge of silence fell on the other side). The scripts are a record of how the
engine was built and judged, as they were run: paths point to the working folder they ran in, and no audio is in
the repo.

## The method
A phaser and a filter sweep are a filter whose curve moves. In the picture of the band levels over time (96 bands,
12 an octave from 70 Hz, a frame every 21 ms) that is an edge, a peak or a few notches that travel while the
sound's own partials stay where they are.

1. `pic.picture` the band levels; `pic.inputs` what the model reads: the level under the loudest cell, the level
   against the band's own middle level, that middle level, the band's place.
2. `net.UNet` (232 000 weights; plain 3x3 layers, the picture halved five times in time and three times across the
   bands and brought back up) answers the tone curve in dB per band and frame, and per frame nothing / filter
   sweep / phaser.
3. `moving.find`: stretches of a second or more where the model says "moving filter" are candidates; each is
   judged for itself (below); the curve works inside the accepted stretches only, with 0.1 s fades.
4. `moving.apply` puts the curve on or takes it off with one gain per bin and moment in a short-time transform
   (2048 points, four windows over every sample; never more than 24 dB up or 60 dB down). Speed: the curve is read
   faster or slower; a curve that repeats goes round its period.

The truth for fitting and for the ceilings: a steady probe signal through the same effect with the same settings
gives the true curve (`pic.probe`, `pic.curve`). With the true curve (`oracle.py`) a phaser comes out at 86 of 100
and goes onto the other sound at 89; a filter sweep 63 and 96 (a deep low-pass throws away what no gain brings
back). That is what this picture and this way of rendering allow.

## The fit (`fx.py`, `mkdata.py`, `train.py`)
16 000 clips of 6 s from the training sounds (none of the scoreboard's sounds): 48 % through this study's own
moving filters (a state-variable filter whose cutoff moves; a chain of all-pass stages with feedback; LFO shapes,
a random walk, one slow ramp, the sound's own loudness, steps), 16 % through four outside tools with new settings
(two of the scoreboard's four phaser tools and two of its four sweep tools; the other four are never seen), and
the rest without a moving filter: plain sounds, a filter that stands still, a level movement, 29 outside effects
of the other families. A quarter of the moving filters are on a stretch of the clip only. A filter that changes
the picture by less than 1 dB counts as nothing.

Lessons: batch normalisation after every layer and He initialisation, Huber loss on the curve; with a plain L1
loss the model settles on "nothing anywhere". The model `mv_d` (10 rounds, 2 hours on 4 cores) underfits: on
unseen clips its curve is off by 8.2 dB where a sweep's curve is 10.6 dB, and by 3.2 dB where a phaser's is 3.5 dB.
A model four times as large did not finish on this machine. This is the engine's weak point, most of all for
phasers.

## What the model says is not a sureness (`rule.py`, `rule2.py`, `feat.py`)
On all 838 sounds of the scoreboard and the clean corpus the first rule (a stretch where the model says "moving
filter", sureness = its mean say-so, shown above 0.65) gives:

| sounds                                        | first rule: shown | final rule: shown |
|-----------------------------------------------|-------------------|-------------------|
| filter sweeps (16)                            | 16                | 16                |
| phasers (16)                                  | 12                | 7                 |
| clean sounds (198, 40 of them recordings)     | 28                | 0                 |
| sounds with one of the 17 other effects (608) | 114               | 1                 |

A sound's own tone moves too, and the model takes that for a filter with a say-so of 0.65 to 0.9. What separates:
- a real filter sweep has a big curve: 8 to 24 dB (root mean square inside the stretch) against 2 to 7 dB on the
  sounds without one that the model is equally sure about (two false ones reach 9 dB, with a say-so under 0.76);
- a phaser's curve (0.3 to 4 dB) is as small as what the model invents on clean sounds (0.3 to 2.8 dB), and
  nothing else measured tells them apart: not how much of the sound the stretch covers, how much the curve moves,
  whether the two sides agree, nor what the model says once its curve is taken off. Only the say-so itself does,
  at the top: 7 of the 16 phasers have 0.95 or more, and of the 822 other sounds one has (a reverb that moves by
  itself).

The final rule (`moving.sureness`), on the app's scale where 0.65 is the default cut: a phaser needs a say-so of
0.93; a filter sweep 0.81 and a curve of 7.4 dB; a stretch shorter than 2 s is trusted less, in proportion. Every
stretch is judged for itself, the surest says which kind the sound has, and stretches of the other kind or less
than half as sure are left alone. Lowering the Sureness slider shows the less sure finds, as for every effect.

What the phasers under the cut would do if shown (`ev2.py`): of the six, two come out a little better (20 and 4),
one does nothing, and three get worse (-5, -27 and -315: phasers so light that the model reads the sound's own
movement instead).

## Reading again (`iter.py`)
The model reads a curve too small. After its curve is taken off, it reads a quarter to 70 % of the same curve again on
the seven sure phasers. So where the say-so is 0.9 or more the curve is taken off, what is left is read and added:
a phaser up to twice, each time only if the model says "moving filter" of what is left with 0.87 or more (without
that test a third reading makes five of the seven worse); a filter sweep once (further readings keep adding the
part of a deep cut that the 24 dB limit does not bring back).

| the scoreboard's cases, both sounds of each (`ev2.py`) | one reading | final |
|---------------------------------------------------------|-------------|-------|
| phasers shown: taken out (median of 7)                  | 18          | 23    |
| phasers shown: onto the other sound                     | 21          | 23    |
| filter sweeps: taken out (median of 16)                 | 28          | 28    |
| filter sweeps: onto the other sound                     | 52          | 56    |

The best phaser goes from 29 to 51 (onto: 34 to 58), two filter sweeps of a tool the fit never saw from 62 and 50
to 79 and 78 onto the other sound. None of the shown ones comes out worse than it went in.

## Long sounds (`mklong.py`, `tlong.py`, through the app's code)
Sounds of 30 to 48 s made of 5 to 8 corpus sounds one after another; the effects from the outside tools with new
settings, on the whole sound or from its middle on.

| long sounds                                   | shown at the default | taken out where shown | onto another long sound |
|-----------------------------------------------|----------------------|-----------------------|-------------------------|
| 12 clean                                      | 0 (sureness up to 0.62) |                    |                         |
| 8 with a filter sweep from end to end         | 7                    | 19 to 32              | 25 to 64                |
| 4 with a filter sweep from the middle on      | 4                    | 27, 34, 27 and 7      |                         |
| 8 with a phaser from end to end               | 1                    | 29                    | 2                       |
| 4 with a phaser from the middle on            | 0                    |                       |                         |

The half before the effect comes back as it went in, except in two cases where the model names 2.6 s before the
effect as well (the sweep that comes out at 7). The long phasers sit just under the cut (sureness 0.41 to 0.65);
shown with a lower Sureness, the seven come out at -5 to 26, five of them better than they went in. Reading takes
0.8 to 4.7 s for these sounds.

## Known limits
- Phasers: shown only when the model is very sure (7 of 16 scoreboard sounds, 1 of 8 long ones), and then taken
  out by a quarter to a half. A light phaser is not told from the sound's own movement.
- Filter sweeps: found reliably when they cut by 8 dB or more; a gentle sweep is not shown at the default. Taking
  a deep sweep off brings back at most what the 24 dB limit allows.
- One of 240 reverb sounds shows a false PHASER (a reverb whose tail is modulated).
- The model is small and underfits; a larger one, fitted for longer, is the next step.
