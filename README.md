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
- one sound: Amount takes the reverb out (far left = the engine's own answer in full; up to 0.4.5 that sat at
  half way left, with a bolder end beyond it) or adds more of it; Length, Brightness and Width reshape the
  sound's own reverb;
- two sounds: the found room, rebuilt from its numbers, goes onto the second sound;
- the display shows how the room dies away (lows, mids, highs); **Export IR** saves it as an impulse response.

Whether the REVERB group shows at all follows how sure the engine is that there is a room (the Sureness slider;
the rule is in `Estimate.h`, `roomSureness`). The first model told room from sound largely by what left and right
do not share. The model in 0.4.1 was also trained on recordings that are narrow or mono as a whole and on wide
sounds under the reverb: on narrow and mono recordings it now finds the room about as often as on stereo ones and
takes out about twice as much as before. Known limits: with a wide stereo sound under the reverb it still takes
less out and errs more often than with a mono or narrow one, and a light reverb is the hardest case.

**Files that stop while the sound plays (model trained again in 0.4.5).** Most files do not end with the sound
ringing out: they stop while it plays, or they are a piece from the middle of something longer. Up to 0.4.4 the
model had learned only from sounds that end inside the file, where a room shows as the tail after the last note.
On a file that is cut off it read a room into nearly half of the dry sounds, which 0.4.2 patched with a second
look with silence after the sound, and on a real reverb it did as much harm as good (40 test reverbs cut off after
6 s, the engine alone: typical score 1.5, 48 % of the cases worse than untouched; whole files 23 and 12 %). The
model of 0.4.5 is the same pair trained on with files that stop, with excerpts cut at both ends, and with sounds
that carry one of the other effects before the reverb or no reverb at all. The engine alone on the same cases: cut
off 11 and 18 %, excerpts 11 and 15 % (before 9.5 and 45 %), whole files 25 and 8 %. The app as a whole, at the
default Sureness: on cut-off files the REVERB group shows on 68 % of the real reverbs instead of 48 % and the mean
score is 17 instead of 9; on excerpts 60 % instead of 45 % and 14 instead of 6; on whole files 90 % instead of
82 % and 27 instead of 23. The look with silence is gone: for this model it changes nothing where there is no
reverb and costs real ones.

**A room that is not there (0.4.5).** Two kinds of false REVERB were told apart. Most sit on the sound itself:
recorded loops and recordings, which may carry a room of their own, and sampled instruments that ring by
themselves. The new training halves these on sample loops and leaves the sampled instruments. The others are made
by another effect: echoes, the copies of a chorus or flanger, a sweeping filter, one side a few milliseconds
behind the other. For those a found room must now still be there with the other found effects taken out (the app
takes out what it shows and can remove cleanly, and the room engine looks again; `Estimate.h`,
`recheckRoomWithout`). On 912 sounds with one of 17 other effects and no reverb that halves the effect-made false
REVERB (50 to 26); of 340 real reverbs one loses its group. On sounds the model never met, REVERB now shows on
9 % of those without a reverb (0.4.4: 14 %) and on 85 % of those with one (0.4.4: 79 %; cut-off files 71 % against
60 %). Known limits: on cut-off files and excerpts about a quarter of the real reverbs are still not seen, light
ones most; with Amount at far left a fifth to a quarter of the cases came out worse than untouched (repaired in
0.4.6, below); effects the
app cannot take out yet (distortion, and up to 0.4.6 widening and ring modulation) can still make a false REVERB.
Method and numbers: `Research/room/README.md`.

**Long sounds (0.4.5).** All of the above is measured on sounds of 6 and 9 s, which is also what the model has
learned from. On longer sounds every build so far read a room far too easily: of 121 test sounds of 18 to 48 s
without a reverb (a clean clip looped, different clips in a row, the same with a sweep, a phaser or a chorus) the
look at the whole sound showed REVERB on 40 %, where the looped clips show it on 10 % by themselves. A sound of
more than 9 s is now judged in pieces of 6 s, each looked at as a sound of its own: it counts as having a room when
the pieces say so on average, or when the last piece does, which is where a sound rings out. That shows REVERB on
17 % of those sounds, and on 74 % of 144 long sounds with a reverb (before: 76 %). Sounds of up to 9 s are judged
exactly as before. Known limit: a row of different sounds that ends on one that rings by itself is still shown.

How well a reverb comes out of a long sound was scored for the first time (53 long sounds with a reverb, 33 to
51 s; REVERB shown on 43). A heavy reverb comes out about as it does out of a 9 s test sound. A light one (direct
sound 5 dB and more over the reverb) is the weak case: Amount half way left scores 20 where shown, and 6 of the 32
come out worse than untouched (9 s test sounds: 1 of 16); at far left 17 of the 32 (5 of 16). Length is not the
main reason. On a light reverb the engine can take more out of the sound itself than the reverb put in, and it
gains most where the sound rings out, which is a third of a 9 s test sound and little of a long file, so the scores
of short sounds that ring out flatter it. In 0.4.5 half way left was therefore the safer setting for a light reverb
and far left was not. Method, numbers and what was tried: `Research/room/README.md`.

**The Amount fader (0.4.6).** Up to 0.4.5 the engine's own answer sat at half way left, and the fader went on to a
bolder end at far left. Scored on the 120 test reverbs in 28 shapes, that end was the worst of them: under the
sound it left 40 of the 120 further from the true dry sound than they were by 0.5 dB or more, the plain answer 3.
Now the fader runs straight from untouched (middle) to the answer in full (far left), and half way left is half of
it in dB.

**The engine under the sound (0.4.6).** The same work gave the scoreboard a score for the sound without its
ring-out. There the answer of 0.4.5 gets 11 of 100, against 26 with the ring-out, and the best possible answer
rendered this way gets 48. The model was as weak under the sound on its own training material, so the limit was
the model and the rule it was trained by, not what it had seen: it takes a sound's own ring for a room, and it
turns doubtful cells down by 1 to 3 dB. Both looks were trained on with a loss that does not reward that hedge and
counts a cell taken too deep 1.5 times. And the second look is held back where it goes much deeper than the first
without evidence for the room in the mid bands (`RoomNumbers.h`, `holdBack`): the worst cases of both pairs were
rooms on recordings whose slow low end read as a decay of 5 to 6 s. The engine alone on the 120: whole file 33
(0.4.5: 26), the sound itself 14 (11), ring-out 83 (75); worse than untouched 8 % (14 %), under the sound 21 %
(28 %); the worst harm under the sound 0.7 dB (1.7 dB). The app with Amount at far left: of the 40 test reverbs
1, 2 and 2 come out worse than untouched (ringing out, cut off, excerpt; the far left of 0.4.5: 11, 9 and 7), the
typical score where REVERB is shown is 33, 27 and 30 (the answer of 0.4.5: 26, 24 and 24), and of the 53 long ones
4 come out worse (0.4.5: 23 at far left, 6 with its answer). The price: the new pair is more careful about calling
something a room, and on excerpts REVERB shows on 20 of the 40 instead of 24. In return it shows where there is no
reverb on 113 of 1,238 short test sounds instead of 126, on 12 % of the long ones instead of 17 %, and on 9 of the
scoreboard's 136 other cases instead of 13; the other effects' own groups and scores are unchanged. Known limits:
under the sound the engine is still weak (14 where 48 is possible), and a recording that carries a room of its
own loses it with the reverb that was added. `Research/room/README.md`, section 0.4.6.

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
the less sure verdict counts, because a train of echoes dies away as a room does (since 0.4.5 together with the
other found effects: `Estimate.h`, `recheckRoomWithout`).
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
6 of 8, 67 and 79 (0 and -18). No case comes out worse than it went in, and on the 198 clean sounds of the corpus
none of the three groups is shown.

Long sounds (20 to 62 s, test clips laid end to end; `Research/level/long.py`), at the default Sureness:
- the effect on the whole sound: shown on 81 of 88, taken out at 88 (tremolo), 94 (auto pan) and 74 (gate) in the
  median; one sound comes out a little worse than it went in (a tremolo, -4);
- the effect on one half: shown on 36 of 44 and taken out at 80, none worse than it went in, and the half without
  the effect comes back unchanged in 31 of the 36 (the difference to the input is under -170 dB; in the other
  five the border sits a little off and it is at -73 to -35 dB);
- slow pans, 3 to 23 s a cycle: shown on 21 of 24 and taken out at 94; the other three pass about once in their
  sound and are offered at a lower Sureness (see the limits);
- 48 clean long sounds: nothing shown;
- a second draw of such sounds gives the same picture: shown on 76 of 88 and 39 of 44, taken out at 86, 95 and
  71 and at 84 on one half, none worse than it went in, the half without the effect unchanged in 36 of the 39,
  nothing shown on 48 clean ones. That draw brought out three faults with gates (a hard gate's clicks, its open
  level, its place in time on long sounds), which were mended on it: as first run its gates on one half came out
  at 61 and one far worse than it went in;
- a third draw, made after that for tremolos and gates: gates come out at 54 on a whole sound and at 68 on one
  half (the code before the fixes: 45 and 52, two worse than they went in, now none), tremolos at 86 and 75 as
  before. What went wrong there: one tremolo comes out worse than it went in (-12), one gated sound is shown as
  a slow AUTO PAN instead of a gate, and a gate with a pattern goes onto a second sound badly (-31).

Known limits: a tremolo of a few dB on a busy sound and a sawtooth that falls (a level that jumps up and sinks
back is what struck notes do) are not shown at the default Sureness; a movement from side to side that the sound
shows only once is offered at a lower Sureness, because something that passes by does the same; one movement per
sound; nothing faster than 16.7 times a second; a gate that closes fully is left closed (what is under it is as
likely noise); a gate with a pattern (some steps of the bar stay open) is followed step by step only where a
stretch of the sound shows the gate by itself, and its pattern is not carried on onto a second sound; a tremolo
whose speed wanders is taken out less well (67 and 24 where shown, on two draws) and goes onto a second sound
poorly. Method, numbers and limits: `Research/level/README.md`.

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
flangers whose delay stays under 1.3 ms are not found. Method, numbers and limits: `Research/sweep/README.md`.

**Chorus (new engine in 0.4.4):** a chorus is the sound plus copies of itself whose delays move round and round
(3 to 45 ms). A copy whose delay grows is the sound played a little slower, so the engine matches every moment
of the sound against itself stretched by up to 6 % either way: a copy is a sharp peak at its delay and its
stretch. Peaks that repeat as a sine or a triangle are a voice. For each voice the engine reads its loudness in
eight frequency ranges (it takes the voices out, measures what is left along each curve, and corrects, six
rounds), and the CHORUS group works on what was read:
- one sound: Amount takes the voices out with the inverse of the moving copies, or makes them louder; Speed and
  Depth change how fast and how far they drift;
- two sounds: the same voices go onto the second sound, along the same curves;
- the display shows every voice's delay over time, and says from when to when the chorus is there.

A drifting pitch draws curves too. A voice counts only if it is seen all round its cycle and has no twins at
simple ratios of its delay (a pitch period shows at twice and half its length; a copy does not), if it is loud
enough, and if taking it out leaves less than half along its curve. A copy louder than the sound cannot be taken
out forwards; it is told apart and taken out from the end of the sound backwards. A sound longer than 12 s is
read in pieces of 12 s, one every 6 s; onto a second sound the clearest piece's voices go all the way.

On the scoreboard's eight chorus cases (four outside tools, sounds of 6 s) the app shows CHORUS on all eight
(build 0.4.3: two, both made far worse), takes 86, 79, 87, 67 and 71 of 100 out of five of them and 2, 1 and -9
out of three, and puts 88, 85, 87, 94, 80, 47, 23 and 29 onto the case's other sound. It shows a chorus on none
of the 198 clean sounds and on none of the 624 sounds with one of the 18 other effects. A copy louder than the
sound comes out at 71 and 74; a 42 s sound at 81 from end to end, and its chorus goes onto a 27 s sound at 91.

Known limits: a chorus whose voices move at different speeds is found but hardly taken out (only the voices of
one speed are read); so is a copy that is louder than the sound and fed back; a copy as loud as the sound comes
out only partly (29 to 57); a faint chorus can be missed; a chorus that starts part way keeps its first seconds;
reading takes about half the sound's length. Method, numbers and limits: `Research/chorus/README.md`.

**Phaser and filter sweep (new engine in 0.4.4):** both are a filter whose curve moves. What they leave is in
the picture of the band levels over time: an edge, a peak or a few notches that travel while the sound's own
partials stay where they are. A small model reads that picture and answers the tone curve per band and moment,
and whether a filter sweep or a phaser is at work. The PHASER and FILTER SWEEP groups work on that curve:
- one sound: Amount takes the curve off or makes it stronger; Speed moves it slower or faster;
- two sounds: the same curve goes onto the second sound at the same times;
- the display shows the curve at three moments of its movement and where along the sound the filter is.

What the model says is not enough by itself: a sound's own tone moves too, and at the usual setting the model's
plain answer showed a filter on one clean sound in seven. So a filter sweep is shown when its curve is also big
(a real one cuts by 8 dB and more), and a phaser, whose curve is as small as a sound's own movement, only when
the model is very sure. The model also reads a curve too small: where it is sure, the curve is taken off, what
is left is read again and added.

On the scoreboard (four outside tools each, two of each four never seen in the fit; both sounds of a case): all
16 filter sweeps are shown, taken out at 28 of 100 (median; a deep sweep throws away what no gain brings back)
and put onto the other sound at 56. Of the 16 phasers 7 are shown, taken out at 23 (17 to 51) and put on at 23;
none of the shown comes out worse than it went in. On the 198 clean sounds neither group is shown; on the 608
sounds with other effects one false PHASER (a reverb that moves by itself). Long sounds (30 to 48 s): nothing on
12 clean ones; a filter sweep is shown on 7 of 8, and one that starts half way leaves the part before it as it
was; a phaser on 1 of 8.

Known limits: the phaser is the weak one. It is shown only when it is strong and clear, and then taken out by a
quarter to a half; a light phaser is not told from the sound's own movement (lowering Sureness shows the
candidates, some of which get worse when taken off). A gentle filter sweep (under 8 dB) is not shown at the
default. The model is small; a larger one is the next step. Method, numbers and limits:
`Research/moving/README.md`.

**Haas width and stereo width (new engine in 0.4.7):** both are done between the two sides of a sound. HAAS
WIDTH: one side arrives later than the other (1.25 to 40 ms). STEREO WIDTH: a sound from the middle is made wide,
by sending it to the two sides through two short filters that differ, or by putting a filtered copy of the middle
at the sides. The engine looks for those filters themselves: the few taps with which one side is told from the
other, fitted on the whole sound, both ways round. With the filters known, the one sound behind both sides can be
worked out, and what the two sides hold beyond it says where along the sound the effect is. The groups work on
that:
- one sound: Amount takes the effect out or makes it stronger. A plain delay: the late side moves in time and the
  levels come together, so no comb is heard on the way. Anything else: a straight line between the sound as found
  and the one sound behind both sides;
- two sounds: each side of the second sound goes through its own filter, so a second sound that is wide already
  keeps what it has at its sides;
- the display shows the taps of both sides (or the copy's filter), and says from when to when the effect is there.

On the scoreboard's 16 cases (sounds of 6 s; build 0.4.6 in brackets): HAAS WIDTH is shown on 8 of 8 (6) and taken
out at 98 of 100 in the median (33); STEREO WIDTH on 6 of 8 (5), taken out at 77 (11) and put onto the case's
other sound at 45 (-17). None of the 16 is made worse, taken out or put on (0.4.6: five of the width cases went
onto the other sound worse than nothing). HAAS WIDTH onto a second sound scores 31 in the median (17), and that
number says little: the outside tools fold the second sound to mono before they work, the app does not, and the
score counts the difference (the two cases with a mono second sound score 98 and 100, the six with a wide one 5
to 75). On 112 fresh cases (eight outside effects on any clip of the corpus) all 56 Haas delays are
found and come out as well as they can (a tool that folded the sound to mono has thrown its sides away for good),
and 37 of the 42 wideners that can be read at all. No HAAS WIDTH and no STEREO WIDTH on the 198 clean clips, nor
on the 384 scoreboard sounds of the other effects.

Known limits: a side that was simply made louder is not found (nothing ties it to the middle: two of the
scoreboard's eight width cases); an effect on less than half of the sound is not found; a delay that changes
along the sound is not followed; in a file cut from the middle of something longer, the late side's first
milliseconds stay as they are. Method, numbers and limits: `Research/sides/README.md`.

**Ring modulation (new engine in 0.4.7):** a sound multiplied by a steady tone. Its loudness then rises and falls
in the tone's cycle whatever the sound itself does, which shows as a line in the spectrum of the sound's power
over time. The engine reads the tone's pitch (20 to 5000 Hz), its phase and how much plain sound is left beside
it, and pins them down on the samples themselves: near the zeros of the gain to a thousandth of a sample. The
RING MOD group works on that:
- one sound: Amount takes the tone out (the sound divided by the gain; next to the gain's zeros, where a sample
  says little, the sound's own course fills in) or deepens it; Carrier moves the tone's pitch, an octave each way;
- two sounds: the second sound is multiplied by the same gain;
- the display shows the gain over a few of the tone's cycles.

On the scoreboard's eight cases (four outside tools): shown on 8 of 8, taken out at 100 of 100 in the median
(seven at 99.6 or more, one at 79.5) and put onto the other sound at 98.5 to 100 (0.4.6: shown on 8 by the
detector, taken out 0, put on at 16). On 80 fresh cases, and on every clip of the corpus with a tone put on twice
(396 cases), it is found on all but one, the pitch off by 0.0014 Hz at most, and taken out at 99.9 in the median
(least 79). None on the 198 clean clips, none on the 384 scoreboard sounds of the other effects.

Known limits: where more plain sound is left than tone part (the gain never reaches zero) that share is measured
less exactly (taken out at 98.7 in the median, least 79); a pure ring modulation may come out with its sign
turned, since which half of the cycle is "up" cannot be known; a tone on a part of the sound is not found, a tone
whose pitch moves is not followed, and a tone under an echo or a reverb is found about half the time. Method,
numbers and limits: `Research/ring/README.md`.

**Bitcrush (new engine in 0.4.7):** a crusher rounds the samples to a grid of values (fewer bits) and holds each
value for several samples (a lower sample rate). Both can be read off the sound exactly: the grid shows as spikes
a step apart in the distribution of the sample values, the hold in the places where the sound changes at all. The
engine reads each side's step and offset and the hold (also one that is not a whole number of samples), and the
BITCRUSH group works on them:
- two sounds: the second sound goes through the same crusher, with the same grid of values and the same hold; Bit
  Depth makes the steps a sixteenth to sixteen times as large, Sample Rate the hold a quarter to four times as
  long;
- one sound: a bare crusher cannot be taken out, because what the rounding and the hold threw away is gone. Left
  of the middle nothing happens; right of it the sound goes through steps twice as large and a hold twice as long;
- a crusher that mixes plain sound back in (one of the four outside tools does) loses nothing. Its mix and the
  rounding of its steps are read too, Amount at far left gives the sound before the crusher, and from there every
  other setting can be had;
- the display shows a quiet tone before and after the found crusher.

On the scoreboard's eight cases: shown on 7 of 8 (as in 0.4.6, then by the detector) and put onto the case's other
sound at 91 in the median (0.4.6: 10): 99.8 and 99.4 for the crusher that mixes plain sound in, 86 to 100 for two
bare ones; the mixing crusher is taken out at 99.6 and 92.7. On 70 fresh cases grid and hold are right on all 42
with a bare crusher, and the mixing crusher is found on 14 of 14 and undone at 99.7 in the median (least 91.6).
No BITCRUSH on the 198 clean clips (also not when they are saved as 16-bit files, with or without dither, or as
24-bit files), none on the 384 scoreboard sounds of the other effects.

Known limits: a crusher with a filter after it is not read (found on 1 of 14 fresh cases and on one of the
scoreboard's two, and then only its grid); which dry sample a held value was and the crusher's rounding rule
(nearest, down, towards zero) cannot be told from the sound, so one outside tool that does both its own way goes
onto a second sound at 80 in the median; a crusher under another effect is not found (the values are off the
grid); a file that was recorded with 16 bits and made louder by 18 dB or more afterwards lies on a grid of 13
bits or coarser and reads as a light crusher. Method, numbers and limits: `Research/crush/README.md`.

**Stutter (new engine in 0.4.7):** a slice of the sound played several times over, in place of what came after
it. A repeat made that way is exact, sample for sample, and nothing a sound does by itself is. The engine finds
every such stretch with a table of all windows of 64 samples, walks each one out to its first and last equal
sample, and reads the events: where each slice starts, how long it is (20 ms to 2 s), how many times it is played,
and whether each play fades in and out. The STUTTER group works on them:
- two sounds: the second sound's own slices are played again at the same times; Amount makes that fewer or more
  plays, Slice the slices half to twice as long;
- one sound: right of the middle each slice is played more often, over what follows it. A stutter cannot be
  undone, since what the repeats covered is gone: left of the middle each slice is played fewer times and the sound
  closes up, so far left is the dry sound with the covered parts missing, and it ends early;
- the display shows the slices and their repeats over the sound's length, and they are marked under the waveform.

On the scoreboard's eight cases: found on 8 of 8 and put onto the case's other sound at 90 in the median (65.5 to
99.8; 0.4.6: -30, a stutter at made-up places, worse than none on all eight). On every clip of the corpus with a
stutter put on (198 clips, 1259 repeats) it is found on all, and 1138 of the 1152 repeats read are right to the
sample; on 48 fresh cases (two outside tools and a plain stutter) 521 are right to the sample, 4 are missed and
one is read that was not made. Far left against the dry sound with the covered parts cut out: 98 to 99 of 100 in
the median. No STUTTER on the 198 clean clips (16-bit and 24-bit files included), none on the 384 scoreboard
sounds of the other effects (0.4.6 showed one on a saturation case).

Known limits: a slice that is silent or stands still is not seen, and where a cut lies inside silence cannot be
known; a repeat that is not exact is not a repeat here (a stutter under another effect, after a change of the
sample rate, or saved with dither); with fades, the 2 ms at each slice's edges stay as they are. Method, numbers
and limits: `Research/stutter/README.md`.

**Where these two sit in the chain, and files that are not at 48 kHz (0.4.7).** A crusher and a stutter can be
read only when they were the last thing the sound went through. So onto a second sound the crusher goes on after
the other found effects and the stutter last, after Tone Match; on the sound itself the crusher is worked first
and the stutter last. A file at another sample rate is read by these two engines at its own rate, because the
app's change to 48 kHz takes the values off the grid and makes the repeats unequal; what they find is carried
over to 48 kHz. On 32 test files at 44.1 and 96 kHz grid and hold are right on all of them, and 276 of 277
stutter events to the sample.

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

And the whole app on the scoreboard (`Research/room/bench/famboard.py`, `famsum.py`; the drop in REVERB in 0.4.2
was the look with silence after the sound, the drop in 0.4.5 is the room model trained again and the look without
the other effects, both described above):

| 136 cases, 17 effects from outside tools                        | 0.4.1    | 0.4.2   | 0.4.3   | 0.4.4  | 0.4.5  | 0.4.6 | 0.4.7 |
|------------------------------------------------------------------|----------|---------|---------|--------|--------|-------|-------|
| the case's own effect shown                                      | 77       | 75      | 79      | 89     | 89     | 89    | 92    |
| other groups shown: REVERB / the rest                            | 63 / 156 | 19 / 12 | 19 / 10 | 19 / 7 | 13 / 7 | 9 / 7 | 9 / 4 |
| cases with no other group shown                                  | 16       | 107     | 109     | 112    | 116    | 120   | 123   |
| own group far left: cases made worse than doing nothing          | 12 %     | 3 %     | 3 %     | 1 %    | 1 %    | 1 %   | 1 %   |
| every shown group far left: cases made worse than doing nothing  | 78 %     | 19 %    | 19 %    | 15 %   | 9 %    | 7 %   | 7 %   |
| everything shown put onto a second sound: cases made worse       | 71 %     | 36 %    | 30 %    | 27 %   | 24 %   | 23 %  | 9 %   |

(0.4.7: the eight stutter cases are left out of the two far-left rows. A stutter's far left closes the sound up,
and the scoreboard compares with the dry sound at the same times; `Research/stutter/so1.py` scores it against the
dry sound with the covered parts cut out.)

What changed from 0.4.2 to 0.4.3 besides the scores of the four reworked groups: four flangers are found; a
tremolo that 0.4.2 took for a gate is shown as a tremolo; another tremolo, which 0.4.2 showed and took out at 24
of 100, is now offered only at a lower Sureness; FLANGER on a chorus and AUTO PAN on a phaser are no longer shown.
New and wrong: TREMOLO, just over the default Sureness, on a bass loop with a low shelf. Every other case is as
it was.

What changed from 0.4.3 to 0.4.4: the three new groups. CHORUS is shown on 8 of 8 cases (2), FILTER SWEEP on 8
of 8 (5), PHASER on 3 of 8 (2, both of which went onto a second sound far worse than nothing); two false FILTER SWEEPs (on a Haas widener and on a
stutter) are gone. Every other case is as it was. Still open: a false REVERB on some chorus, phaser and flanger
sounds, which is most of what the last two rows have left.

What changed from 0.4.4 to 0.4.5: only REVERB. Every case's own group is shown as before, with the same scores.
Of the 19 REVERB groups on cases without a reverb, ten are gone and four are new. Ten of the 13 that are left sit
on sounds that show REVERB with no effect on them (recorded loops, recordings, sampled plucked strings): the ring
is in the sound, and no look at the effect can take it away. Three are made by the effect: a chorus that comes
out only in part, a delay whose echoes are taken out and a room is still read, and a delay whose echo is not
found. With fewer false rooms to pull out, the last two rows move as well.

What changed from 0.4.5 to 0.4.6: only REVERB again (the Amount fader and the model that is careful under the
sound, both above). Four more false REVERB groups are gone and none is new; every case's own group is shown as
before, with the same scores.

What changed from 0.4.6 to 0.4.7: the five groups with new engines, and nothing else. The other twelve effect
types score exactly as before, case by case. HAAS WIDTH is shown on 8 of 8 cases (6) and STEREO WIDTH on 6 of 8
(5); a crusher with a filter after it is now shown on one of its two cases and not on the other (the detector
had it the other way round). Three groups that were not there are gone: a DELAY on a ring modulation, a COMB
FILTER on a widener, a STUTTER on a saturation. The last row falls from 31 cases to 12: the 18 cases that 0.4.6
made worse in these families (all eight stutters, five wideners, three ring modulations, two crushers) and the
saturation with the false STUTTER now go onto the second sound better than nothing, most of them far better. In
the median: HAAS WIDTH 31 (17), STEREO WIDTH 45 (-17), RING MOD 100 (16), BITCRUSH 91 (10), STUTTER 90 (-30). Of
the 12 that are left, eight carry a false REVERB, three a false EXCITER, and one is an exciter whose own group
goes on badly; the exciter is one of the effect types that have no engine of their own yet.

The price is in the third row of the first table, and it is uneven: flanger, phaser, filter sweep and a fixed filter are found less
often than before (9, 39, 39 and 9 % against 36, 73, 61 and 34 %), because most of their earlier finds came with
as many false ones; ring mod and stutter are found far more often (82 and 93 % against 23 and 52 %). The
Sureness slider means the same for every effect now: shown when the score is above the slider. At 30 % more is
shown (63 % found, 67 % of clean sounds clean), at 80 % less (39 %, 98 %). A long sound is judged in blocks of 6 s
and an effect has to be there in half of them, so an effect on a short part of a long sound is not shown (the
level and flanger engines of 0.4.3 look for themselves where on the sound their effect is). Reverb, the comb
filter, the delay, tremolo, auto pan, gated rhythm, the flanger, the chorus, the phaser and the filter sweep keep
their own engines' verdicts. Method and tables: `Research/detect/README.md`.

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
  The chorus engine: `Chorus.h` (the picture of the sound against itself stretched, peaks, voices, the tests
  against a drifting pitch, loudness per range, copies louder than the sound, long sounds piece by piece, the
  voices taken out, changed or put onto a second sound).
  The moving-tone engine (phaser, filter sweep): `Moving.h` (the band picture, the model, stretches and their
  sureness, the repeated reading, the curve put on and taken off).
  The sides engine (Haas width, stereo width): `Sides.h` (the taps between the two sides, the copy at the sides,
  the one sound behind both sides, the rule, the end of the file, the effect taken out, changed or put onto a
  second sound).
  The ring engine: `Ring.h` (the lines in the spectrum of the sound's power, the table over the tone's cycle, the
  fit of pitch, phase and plain share on the samples, the gain taken out, changed or put on).
  The crush engine: `Crush.h` (the grid from the distribution of the sample values, the hold, the crusher that
  mixes plain sound in and its undoing, the faders for steps and hold, files at their own sample rate).
  The stutter engine: `Stutter.h` (the table of windows, stretches, events, what repeats exactly and is no event,
  the events played on a sound, cut out of it or moved to another sample rate).
  `Verify.h` the check of every effect against the whole sound before it is shown.
- `Source/App/Main.cpp` the JUCE app.
- `Resources/models.bin` (12 effects) and `models27.bin` (27 effects, `DR_MODEL27=1`) the trained models, written by `Tools/export_models.py`.
- `Resources/room.bin` and `room2.bin` the reverb engine's model (first and second look), written by `Tools/export_room_model.py`.
- `Resources/verify.bin` the verifier's models, written by `Research/detect/export_verify.py`.
- `Resources/moving.bin` the moving-tone engine's model, written by `Tools/export_moving_model.py`.
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
- `Tests/chorus.cpp` the chorus engine alone: `drchorus a.wav b.wav prefix`, `drchorus both ...`, `drchorus find ...`
  (see the top of the file); `Tools/chorus_parity.py` compares it with the prototype.
- `Tests/moving.cpp` the moving-tone engine alone: `drmoving model.bin a.wav b.wav prefix`, `drmoving both ...`
  (see the top of the file); `Tools/moving_parity.py` compares it with the prototype.
- `Tests/sides.cpp`, `Tests/ring.cpp`, `Tests/crush.cpp`, `Tests/stutter.cpp` the four engines of 0.4.7 alone:
  `drsides|drring|drcrush|drstutter a.wav b.wav prefix` and `... find a.wav ...` (see the top of each file);
  `Tools/sides_parity.py`, `ring_parity.py`, `crush_parity.py` and `stutter_parity.py` compare them with the
  prototypes.
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
- `Research/chorus/` the Python side of the chorus engine: the prototype the C++ matches, the studies behind its
  tests, the scores, the tests on long sounds.
- `Research/moving/` the Python side of the moving-tone engine: the prototype the C++ matches, the fit data and
  the training, the rule study, the repeated reading, the tests on long sounds. Also the memory check that is
  run before every build (`memcheck.py`, with `mklong.py` and `mknew.py` for its sounds).
- `Research/sides/` the Python side of the sides engine: the prototype the C++ matches, the fresh cases, the
  false-find checks and their report.
- `Research/ring/` the Python side of the ring engine: the prototype the C++ matches, the fresh cases, every clip
  with a tone put on, the false-find checks.
- `Research/crush/` and `Research/stutter/` the Python side of the crush and the stutter engine: the prototypes
  the C++ matches, the fresh cases, every clip with a stutter put on, the far left, the false-find checks and
  the files at other sample rates (several scripts serve both engines and sit in `Research/stutter/`).
- `Research/listening/` the scripts that make the listening page: its examples are rendered by `drrender`.

## Build
`cmake -S . -B build && cmake --build build --config Release --target DefectRefect`
(Windows exe built by GitHub Actions.)
