#!/bin/bash
# scen.sh display tag scenario first.wav [second.wav]: the sanitizer build of the app on a display of its own, driven with the mouse.
# Scenarios: two     two sounds: every fader, Swap twice, one sound alone, Export to a file, close
#            quit    asked to close WAIT seconds after the start (while the first analysis runs); WAIT=reading: once the
#                    Pitch fader says "reading" and AFTER seconds more, so while the pitch reader's own thread runs
#                    (clear, swap and exp's EARLY take the same)
#            clear   the first sound taken away WAIT seconds after the start, then close
#            swap    Swap pressed WAIT seconds after the start, then three times in a row, then close
#            exp     the found effects' fader rows ROWS and MOVEMENT's rows MROWS pulled to one end (END, MEND: left,
#                    right or middle), then Export to a file, close
#            monkey  N random clicks, drags and wheel turns on the controls (SEED), then close
#            four    build 0.5.1: SATURATION and a FILTER group with two sounds: every fader of both, Swap twice, one sound alone, Export, close
#            flat    build 0.5.2: a sound whose tops were cut flat (DISTORTION, one fader) with a second sound: the fader's tip, the fader to both
#                    ends and part of the way, Swap twice, one sound alone (the tops rebuilt), Export, close
#            watch   a picture every 2 s (N of them), then close: shows when the Pitch fader says "reading"
# The places are those of the window as it opens (1200 x 820 on a screen of 1280 x 860): MOVEMENT in the left column
# of faders, the found effects in the right one.
D=$1; T=$2; S=$3; O=/tmp/claude-0/drasan; APP=${APP:-"$O/Defect Refect"}
MX=239; FX=583          # the middle of a fader: MOVEMENT's column, the found effects' column
MSW=365; FSW=709        # the On/Off switches
MHEAD=110; FHEAD=454    # a place on the group's title
MY="422 446 470 494 518"                                 # MOVEMENT's rows: Pitch, Loudness, Tone, Stereo, Spectral
endx() { case $2 in left) echo $(($1 - 89)) ;; right) echo $(($1 + 87)) ;; *) echo $1 ;; esac; }
Xvfb :$D -screen 0 1280x860x24 -noreset > /dev/null 2>&1 &
echo $! > $O/xvfb_$T.pid
sleep 2
export DISPLAY=:$D
$O/wmdel intern
if [ -n "$5" ]; then ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=print_stacktrace=1 nice "$APP" "$4" "$5" > $O/app_$T.log 2>&1 &
else ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=print_stacktrace=1 nice "$APP" "$4" > $O/app_$T.log 2>&1 & fi
PID=$!; echo $PID > $O/app_$T.pid
ticks() { sed 's/.*) //' /proc/$PID/stat 2>/dev/null | awk '{print $12+$13}'; }
say() { echo "$(date -u +%H:%M:%S) $T $*"; }
idle() {      # wait until the app has used next to no processor time for three polls of 4 s in a row (at most 30 minutes), then take a picture
    local n=0 k=0 t0 t1
    while [ $k -lt 450 ]; do
        kill -0 $PID 2>/dev/null || { say "APP GONE at step $1"; return 1; }
        t0=$(ticks); sleep 4; t1=$(ticks)
        if [ $((t1 - t0)) -lt 12 ]; then n=$((n + 1)); else n=0; fi
        [ $n -ge 3 ] && break
        k=$((k + 1))
    done
    import -window root $O/${T}_$1.png; say "step $1 idle after $((k * 4)) s"
}
drag() { xdotool mousemove $1 $2 mousedown 1; sleep 0.2; local st=8; [ $3 -lt $1 ] && st=-8; for x in $(seq $1 $st $3); do xdotool mousemove $x $2; sleep 0.05; done; xdotool mousemove $3 $2; sleep 0.2; xdotool mouseup 1; sleep 0.5; }
click() { xdotool mousemove $1 $2; sleep 0.2; xdotool click 1; sleep 0.5; }
reading() {   # 0 when the Pitch fader's note says "reading" right now (the place of the note against a picture of the word)
    local d=$(import -window root -crop 60x16+326+414 +repage png:- 2>/dev/null | compare -metric AE -fuzz 25% - $O/reading_tpl.png null: 2>&1); [ "${d%%[!0-9]*}" = "$d" ] && [ -n "$d" ] && [ "$d" -lt 20 ]
}
pause() {     # pause N: N seconds; pause reading: until the Pitch fader says "reading" (at most 5 minutes), then AFTER seconds more (1 if not given)
    if [ "$1" = reading ]; then local k=0; until reading; do sleep 0.5; k=$((k + 1)); kill -0 $PID 2>/dev/null || break; [ $k -gt 600 ] && { say "NEVER SAID reading"; break; }; done; say "reading shown after about $((k / 2)) s"; sleep ${AFTER:-1}
    else sleep $1; fi
    if reading; then say "the pitch is being read now"; else say "the pitch is NOT being read now"; fi
}
closeapp() {  # ask the window to close, as the close button does, and wait for the process to end by itself (at most 10 minutes)
    local w=$(xdotool search --onlyvisible --pid $PID 2>/dev/null | head -1); [ -z "$w" ] && w=$(xdotool search --name "Defect Refect" | head -1)
    say "asking window $w to close"; $O/wmdel $w
    local k=0; while kill -0 $PID 2>/dev/null && [ $k -lt 150 ]; do sleep 4; k=$((k + 1)); done
    if kill -0 $PID 2>/dev/null; then say "STILL RUNNING $((k * 4)) s after the close request"; import -window root $O/${T}_stuck.png; kill $PID; else wait $PID; say "app ended by itself after about $((k * 4)) s, exit code $?"; fi
}
case $S in
two)
    idle s0 || exit 1
    drag $FX 422 $((FX - 67)); idle s1 || exit 1         # BITCRUSH Amount far left
    drag $FX 446 $((FX + 66)); idle s2 || exit 1         # Bit Depth right
    drag $FX 470 $((FX - 64)); idle s3 || exit 1         # Sample Rate left
    click $FHEAD 509; drag $FX 534 $((FX - 74)); idle s4 || exit 1   # STUTTER: Amount far left
    drag $FX 558 $((FX + 66)); idle s5 || exit 1         # Slice right
    drag $MX 422 $((MX - 89)); idle s5a || exit 1        # MOVEMENT: Pitch far left (the second sound's own taken out)
    drag $MX 446 $((MX + 87)); idle s5b || exit 1        # Loudness far right (the first sound's)
    drag $MX 470 $((MX - 40)); idle s5c || exit 1        # Tone part of the way left
    drag $MX 494 $((MX + 87)); idle s5d || exit 1        # Stereo far right
    drag $MX 518 $((MX + 87)); idle s5e || exit 1        # Spectral far right
    click 189 322; click 96 322; sleep 3; click 96 322; click 189 322; idle s6 || exit 1    # Result / Original, Play and stop
    click 640 166; idle s7 || exit 1                     # Swap
    click 640 166; idle s8 || exit 1                     # Swap back
    click 973 322; idle s9 || exit 1                     # Reset Faders
    click 292 322; idle s10 || exit 1                    # Tone Match on
    click 1205 55; idle s11 || exit 1                    # second sound away: one sound
    click $FHEAD 509; drag $FX 534 $((FX - 74)); idle s12 || exit 1  # STUTTER Amount far left on the sound itself (ends early)
    drag $FX 422 $((FX + 66)); idle s13 || exit 1        # BITCRUSH Amount right
    click $FSW 397; idle s14 || exit 1                   # BITCRUSH off
    click $FSW 397; drag 298 359 165; idle s15 || exit 1 # on again; Sureness far left
    drag 165 359 430; idle s16 || exit 1                 # Sureness far right
    drag 430 359 298; idle s17 || exit 1                 # and back
    drag $MX 422 $((MX + 87)); idle s17a || exit 1       # one sound: Pitch far right (its own movement twice)
    drag $MX 446 $((MX - 89)); idle s17b || exit 1       # Loudness far left
    drag $MX 470 $((MX + 87)); idle s17c || exit 1       # Tone far right
    click $MSW 397; idle s17d || exit 1                  # MOVEMENT off
    click $MSW 397; idle s17e || exit 1                  # and on again
    rm -f $O/export_$T.wav
    click 1179 322; sleep 3; click 640 617; xdotool key ctrl+a; sleep 0.3; xdotool type --delay 25 "$O/export_$T.wav"; sleep 0.5; xdotool key Return; sleep 2
    import -window root $O/${T}_export.png; click 885 656; sleep 5; import -window root $O/${T}_exported.png; idle s18 || exit 1
    ls -la $O/export_$T.wav 2>&1
    closeapp ;;
four)       # build 0.5.1: a sound that shows SATURATION (one fader, row 422) and a FILTER group under it (title 461, Amount 486, Cutoff 510), with a second sound
    idle s0 || exit 1
    drag $FX 422 $(endx $FX left); idle s1 || exit 1                 # SATURATION Amount far left (the second sound not pushed into the curve)
    drag $((FX - 71)) 422 $(endx $FX right); idle s2 || exit 1       # far right (as far again); the thumb at far left stands 71 to the left of the middle
    click $FHEAD 461; drag $FX 486 $(endx $FX left); idle s3 || exit 1   # FILTER: Amount far left
    drag $FX 510 $((FX + 66)); idle s4 || exit 1                     # Cutoff right
    drag $((FX - 71)) 486 $(endx $FX right); idle s5 || exit 1       # Amount far right (the curve twice)
    drag $MX 446 $((MX + 87)); idle s5b || exit 1                    # MOVEMENT: Loudness far right
    drag $MX 470 $((MX - 40)); idle s5c || exit 1                    # Tone part of the way left
    click 189 322; click 96 322; sleep 3; click 96 322; click 189 322; idle s6 || exit 1    # Result / Original, Play and stop
    click 640 166; idle s7 || exit 1                                 # Swap
    click 640 166; idle s8 || exit 1                                 # Swap back
    click 973 322; idle s9 || exit 1                                 # Reset Faders
    click 292 322; idle s10 || exit 1                                # Tone Match on
    click 1205 55; idle s11 || exit 1                                # second sound away: one sound
    drag $FX 486 $(endx $FX left); idle s12 || exit 1                # FILTER Amount far left on the sound itself (the filter taken out)
    drag $FX 510 $(endx $FX left); idle s13 || exit 1                # Cutoff far left
    drag $FX 422 $(endx $FX left); idle s13a || exit 1               # SATURATION walked back
    click $FSW 397; idle s14 || exit 1                               # SATURATION off
    click $FSW 397; click $FSW 461; idle s14a || exit 1              # on again; FILTER off
    click $FSW 461; drag 298 359 165; idle s15 || exit 1             # on again; Sureness far left
    drag 165 359 430; idle s16 || exit 1                             # Sureness far right
    drag 430 359 298; idle s17 || exit 1                             # and back
    rm -f $O/export_$T.wav
    click 1179 322; sleep 3; click 640 617; xdotool key ctrl+a; sleep 0.3; xdotool type --delay 25 "$O/export_$T.wav"; sleep 0.5; xdotool key Return; sleep 2
    import -window root $O/${T}_export.png; click 885 656; sleep 5; import -window root $O/${T}_exported.png; idle s18 || exit 1
    ls -la $O/export_$T.wav 2>&1
    closeapp ;;
flat)       # build 0.5.2: a sound whose tops were cut flat and are rebuilt (DISTORTION, one fader, row 422), with a second sound
    idle s0 || exit 1
    xdotool mousemove $FX 422; sleep 3; import -window root $O/${T}_tip2.png                  # the fader's tip with two sounds
    drag $FX 422 $(endx $FX left); idle s1 || exit 1                 # Amount far left (the second sound not pushed into the cut)
    drag $((FX - 71)) 422 $((FX - 30)); idle s2 || exit 1            # part of the way back
    drag $((FX - 30)) 422 $(endx $FX right); idle s3 || exit 1       # far right (as far again)
    drag $MX 446 $((MX + 87)); idle s4 || exit 1                     # MOVEMENT: Loudness far right
    drag $MX 470 $((MX - 40)); idle s5 || exit 1                     # Tone part of the way left
    click 189 322; click 96 322; sleep 3; click 96 322; click 189 322; idle s6 || exit 1    # Result / Original, Play and stop
    click 640 166; idle s7 || exit 1                                 # Swap: the cut sound is the second one now
    click 640 166; idle s8 || exit 1                                 # Swap back
    click 973 322; idle s9 || exit 1                                 # Reset Faders
    click 292 322; idle s10 || exit 1                                # Tone Match on
    click 1205 55; idle s11 || exit 1                                # second sound away: one sound
    xdotool mousemove $FX 422; sleep 3; import -window root $O/${T}_tip1.png                  # the fader's tip with one sound
    drag $FX 422 $(endx $FX left); idle s12 || exit 1                # Amount far left on the sound itself: the rebuilt tops
    drag $((FX - 71)) 422 $((FX - 36)); idle s13 || exit 1           # half way
    drag $((FX - 36)) 422 $(endx $FX right); idle s14 || exit 1      # far right: cut as deep again
    click $FSW 397; idle s15 || exit 1                               # the group off
    click $FSW 397; drag 298 359 165; idle s16 || exit 1             # on again; Sureness far left
    drag 165 359 430; idle s17 || exit 1                             # Sureness far right
    drag 430 359 298; drag $((FX + 71)) 422 $(endx $FX left); idle s18 || exit 1     # and back; Amount far left again
    rm -f $O/export_$T.wav
    click 1179 322; sleep 3; click 640 617; xdotool key ctrl+a; sleep 0.3; xdotool type --delay 25 "$O/export_$T.wav"; sleep 0.5; xdotool key Return; sleep 2
    import -window root $O/${T}_export.png; click 885 656; sleep 5; import -window root $O/${T}_exported.png; idle s19 || exit 1
    ls -la $O/export_$T.wav 2>&1
    closeapp ;;
quit)
    pause ${WAIT:-12}; import -window root $O/${T}_busy.png; closeapp ;;
clear)
    pause ${WAIT:-12}; import -window root $O/${T}_busy.png; click 586 55; sleep 2; import -window root $O/${T}_cleared.png; idle c1 || exit 1; closeapp ;;
swap)
    pause ${WAIT:-12}; import -window root $O/${T}_busy.png; click 640 166; idle w1 || exit 1
    click 640 166; click 640 166; click 640 166; idle w2 || exit 1
    click 640 166; sleep 6; closeapp ;;
watch)      # a picture every 2 s, N of them (to see when the Pitch fader says "reading"), then close
    for i in $(seq 1 ${N:-60}); do import -window root $O/${T}_w$i.png; kill -0 $PID 2>/dev/null || break; sleep 2; done; closeapp ;;
exp)        # ROWS: the found effects' fader rows to pull to one end (END); MROWS: MOVEMENT's (MEND, else END); EXPORT: the file to save
            # EARLY: the faders are pulled that many seconds after the start, while the pitch is still being read, instead of once the app is at rest
    if [ -n "$EARLY" ]; then pause $EARLY; import -window root $O/${T}_early.png; else idle e0 || exit 1; fi
    for y in $MROWS; do drag $MX $y $(endx $MX ${MEND:-${END:-left}}); done
    for y in $ROWS; do drag $FX $y $(endx $FX ${END:-left}); done
    if [ -n "$ROWS$MROWS" ]; then idle e1 || exit 1; fi
    X=${EXPORT:-$O/export_$T.wav}; rm -f $O/export_$T.wav
    click 1179 322; sleep 3; click 640 617; xdotool key ctrl+a; sleep 0.3; xdotool type --delay 25 "$X"; sleep 0.5; xdotool key Return; sleep 2
    click 885 656; sleep 5; idle e2 || exit 1
    ls -la "$X" 2>&1 | cut -c1-120
    closeapp ;;
monkey)
    idle m0 || exit 1
    RANDOM=${SEED:-7}; N=${N:-150}; ROWS=(${FROWS:-422 446 470 534 558 582 606}); MROWS=($MY)      # FROWS: the found effects' fader rows of this sound; HY2: its second group's title
    for i in $(seq 1 $N); do
        kill -0 $PID 2>/dev/null || { say "APP GONE at action $i"; break; }
        a=$((RANDOM % 40)); c=$((RANDOM % 5)); o1=$((RANDOM % 139 - 69)); o2=$((RANDOM % 139 - 69))
        if [ $c -lt 2 ]; then y=${MROWS[$((RANDOM % 5))]}; x1=$((MX + o1)); x2=$((MX + o2)); sw=$MSW; hd=$MHEAD; hy=397      # two times in five: MOVEMENT's column
        else y=${ROWS[$((RANDOM % ${#ROWS[@]}))]}; x1=$((FX + o1)); x2=$((FX + o2)); sw=$FSW; hd=$FHEAD; hy=$((RANDOM % 2 ? 397 : ${HY2:-509})); fi
        case $a in
            0|1) say "$i swap"; click 640 166 ;;
            2|3) say "$i play"; click 96 322 ;;
            4|5) say "$i result"; click 189 322 ;;
            6|7) say "$i tone match"; click 292 322 ;;
            8) say "$i reset"; click 973 322 ;;
            9|10|11) say "$i sureness"; drag $((231 + o1)) 359 $((349 + o2)) ;;
            12|13) say "$i switch $sw $hy"; click $sw $hy ;;
            14|15) say "$i header $hd $hy"; click $hd $hy ;;
            16|17|18) say "$i double click row $y"; xdotool mousemove $x1 $y; sleep 0.1; xdotool click --repeat 2 --delay 80 1 ;;
            19|20|21|22) say "$i wheel row $y"; xdotool mousemove $x1 $y; sleep 0.1; xdotool click --repeat $((1 + RANDOM % 6)) --delay 40 $((4 + RANDOM % 2)) ;;
            *) say "$i drag row $y $x1 to $x2"; drag $x1 $y $x2 ;;
        esac
        sleep 0.$((RANDOM % 10))
        if [ $((i % 15)) -eq 0 ]; then idle m$i || break; fi
    done
    idle mEnd; closeapp ;;
esac
kill $(cat $O/xvfb_$T.pid) 2>/dev/null
say "scenario $S done"; echo "sanitizer lines in the app's log: $(grep -c 'ERROR: AddressSanitizer\|runtime error' $O/app_$T.log)"
