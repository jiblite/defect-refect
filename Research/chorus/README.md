# The chorus engine: Python side

`chorus.py` (the picture, the peaks, the voices) and `engine.py` (find, apply; with `gains.py`, `inv.py`,
`prim.py`) are the prototype that the single-piece part of `Source/Core/Chorus.h` is a port of.
`Tools/chorus_parity.py` compares the two on whole sounds: the same peaks (delays to 4e-8 samples), the same
voices and decisions, and the rendered sounds to -160 dB or better (five pairs, among them a three-voice chorus
and a copy louder than the sound). Long sounds (pieces) and the second sound's handling were written in C++ only
and are tested from here through `drchorus` (`tlong.py`, `cbench.py`, `cfind.py`). The scripts are a record of
how the engine was built and judged, as they were run: paths point to the working folder they ran in, and no
audio is in the repo.

## The method
A chorus voice is a copy of the sound whose delay moves round and round (3 to 45 ms here).

1. `picture`: a copy whose delay grows by r per second is the sound played r slower. So each frame (2048 samples
   at 24 kHz, every 512) is matched against the sound stretched by -6 to 6 % and read up to 45 ms earlier, every
   frequency divided by its smoothed level. A voice is a sharp peak at its delay and its stretch; in a plain
   autocorrelation it smears over the delays it passes through.
2. `peaks` the 8 highest of each frame; `chains` link peaks whose stretch predicts the next delay; `rounds`
   folds the moving chains at every speed and keeps the speeds at which the rounds look alike.
3. `voices`: per speed and shape (sine, triangle) every chain gives a curve; the curves that collect most are
   kept and refined (middle, swing, phase, speed), all voices of a side at one speed.
4. `gains.find`: each voice's loudness in 8 ranges (edges 150 Hz to 9.6 kHz) without knowing the dry sound: the
   voices are taken out with the loudness so far (the recursive inverse `x[n] = y[n] - sum g x(n - d[n])` per
   range), what is still found along each curve is measured and added, six rounds.
5. Tests before a voice counts (`engine.find`): loud enough (0.08 of the sound in the middle ranges), gone after
   removal (under 45 % left along its curve), and the result not flatter than the sound was.
6. `engine.apply`: out with the inverse; Amount, Speed and Depth; onto another sound as copies read along the
   same curves with the same loudness per range.

## False voices: the sound's own pitch (`t24.py`)
With the tests of step 5 alone, 5 of 198 clean sounds showed a chorus (a plucked string, two drum loops, two
recordings): a pitch that drifts draws one side of a curve at the delay of its period. Two tests in `voices`,
before any loudness is measured, removed all five and none of the real ones:
- seen all round: the voice's cycle is cut in eighths, and in every eighth that can be judged a share of the
  frames must have a peak on the curve (the least share at least 0.1). A drifting pitch lies on one flank;
- no twins: a pitch period also shows at twice and half its delay and at 3/2, 2/3 and so on. A voice with a
  partner at a simple ratio of its delay in 40 % of its frames is the sound's own pitch.

Result with the app's code (`cfind.py`): a chorus on 0 of 198 clean sounds, on 0 of the 624 sounds of the 18 other
effect families, on 15 of the 16 chorus sounds of the scoreboard and on 11 of 12 extra ones, each of the found
with a sureness of 0.99 or 1.

## Copies louder than the sound (`t25.py`, `t26.py`, `t27.py`)
`y = x + g x(n - d)` with g over 1 has no stable inverse forwards. Seen from the copy it is `v + (1/g) v(n + a)`:
the sound is an early, quieter copy of the copy. So the sound is turned round, the usual loudness search and
inverse run on it, and the answer is turned back (`engine.backwards`). Which way round a one-voice sound is, is
asked only where a first measure (`lean`: is the later of the two the weaker?) leaves it open, and then decided by
four votes (the lean, what is left along the curve, how peaked and how uneven the result is) or by the usual way
failing its tests where the other passes. On the scoreboard's and the extra cases every sound is put the right
way round; the vote is wrong once and the tests catch it.

## Long sounds (in `Chorus.h` only; `mklong.py`, `tlong.py`)
Pieces of 12 s every 6 s, each read for itself and faded into the next over 2 s. A piece must start on a whole
picture frame (1024 samples at 48 kHz): with starts every 6 s exactly the curves sat up to 768 samples off the
sound, and the middle of a 42 s sound kept its chorus. Onto a second sound the clearest piece's voices go on all
the way.

## Results (the app's code; 100 = the dry sound or the true wet one, 0 = nothing gained)
`cbench.py`, both sounds of each scoreboard case (four outside tools, two settings each):

| tool (what its chorus is)                                  | found  | taken out          | onto the other sound |
|------------------------------------------------------------|--------|--------------------|----------------------|
| three voices at one speed                                  | 4 of 4 | 86, 82, 79, 73     | 88, 85, 85, 76       |
| one voice, light                                           | 4 of 4 | 67, 1, 71, 59      | 94, 77, 80, 80       |
| one voice with feedback; in one setting louder than the sound | 3 of 4 | 2, 9, 87        | 47, 39, 87           |
| several voices at different speeds, delays to 41 ms        | 4 of 4 | 1, 2, -9, 11       | 23, 8, 29, 36        |

Median where found: 59 taken out, 77 onto the other sound; one sound comes out worse than it went in (-9). The
build before had no chorus engine: it showed CHORUS on two of eight cases and made both far worse (about -300).

Extra cases from one outside tool (`mkx.py`): an even mix 52 and 57 out (86 and 85 onto); another even mix 29 out
and, on a sound of bursts, worse by the score (91 and 78 onto); a copy louder than the sound 71 and 74 (86 and 87);
with feedback 35 and 21 (40 and 39); a slow chorus at 20 ms 69 and 52 (94 and 75); a faint one 84 on one sound
and not found on the other.

Long sounds: 42 s with a chorus from end to end 81 out (76 to 87 by 6 s, 59 at the very end) and 91 onto a 27 s
sound; a chorus that starts at 17 s of 36 s: the part before it comes back untouched, the first 4 s of the chorus
stay in, the rest 84; 2.5 s: 84; 13 s: 84. Reading takes 1.7 to 2.6 s for 6 s of stereo and about 0.45 times the
length of a long sound.

## Known limits
- Voices at different speeds: only those of one speed are read, so such a chorus is found but hardly taken out.
- A copy as loud as the sound comes out only partly (29 to 57); louder and fed back it is found but hardly taken
  out (2 to 9). The common chorus feeds back with a minus sign: `Y = a X (1 + (b/a + f) z^-d) / (1 + f z^-d)`. A
  filter `y + f y(n - d)` in front would take the feedback out exactly, and f could be read from what the result
  still shows along the doubled curve. Not built yet.
- A faint chorus (the copy at a quarter of the sound) can be missed. Voices that cross zero delay are not read.
- A chorus that starts part way keeps its first seconds (the pieces are 12 s long).
