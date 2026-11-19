#!/bin/bash
# scen.sh display tag scenario first.wav [second.wav]: the sanitizer build of the app on a display of its own, driven with the mouse.
# Scenarios: two     two sounds: every fader, Swap twice, one sound alone, Export to a file, close
#            quit    asked to close WAIT seconds after the start (while the first analysis runs)
#            clear   the first sound taken away WAIT seconds after the start, then close
#            swap    Swap pressed WAIT seconds after the start, then three times in a row, then close
#            exp     the fader rows ROWS pulled to one end (TO), then Export to a file, close
#            monkey  N random clicks, drags and wheel turns on the controls (SEED), then close
D=$1; T=$2; S=$3; O=/tmp/claude-0/drasan; APP="$O/Defect Refect"
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
closeapp() {  # ask the window to close, as the close button does, and wait for the process to end by itself (at most 10 minutes)
    local w=$(xdotool search --onlyvisible --pid $PID 2>/dev/null | head -1); [ -z "$w" ] && w=$(xdotool search --name "Defect Refect" | head -1)
    say "asking window $w to close"; $O/wmdel $w
    local k=0; while kill -0 $PID 2>/dev/null && [ $k -lt 150 ]; do sleep 4; k=$((k + 1)); done
    if kill -0 $PID 2>/dev/null; then say "STILL RUNNING $((k * 4)) s after the close request"; import -window root $O/${T}_stuck.png; kill $PID; else wait $PID; say "app ended by itself after about $((k * 4)) s, exit code $?"; fi
}
case $S in
two)
    idle s0 || exit 1
    drag 279 422 212; idle s1 || exit 1                  # BITCRUSH Amount far left
    drag 279 446 345; idle s2 || exit 1                  # Bit Depth right
    drag 279 470 215; idle s3 || exit 1                  # Sample Rate left
    click 150 509; drag 279 534 205; idle s4 || exit 1   # STUTTER: Amount far left
    drag 279 558 345; idle s5 || exit 1                  # Slice right
    click 229 322; click 136 322; sleep 3; click 136 322; click 229 322; idle s6 || exit 1    # Result / Original, Play and stop
    click 640 166; idle s7 || exit 1                     # Swap
    click 640 166; idle s8 || exit 1                     # Swap back
    click 933 322; idle s9 || exit 1                     # Reset Faders
    click 332 322; idle s10 || exit 1                    # Tone Match on
    click 1165 55; idle s11 || exit 1                    # second sound away: one sound
    click 150 509; drag 279 534 205; idle s12 || exit 1  # STUTTER Amount far left on the sound itself (ends early)
    drag 279 422 345; idle s13 || exit 1                 # BITCRUSH Amount right
    click 405 397; idle s14 || exit 1                    # BITCRUSH off
    click 405 397; drag 338 359 205; idle s15 || exit 1  # on again; Sureness far left
    drag 205 359 470; idle s16 || exit 1                 # Sureness far right
    drag 470 359 338; idle s17 || exit 1                 # and back
    rm -f $O/export_$T.wav
    click 1139 322; sleep 3; click 640 617; xdotool key ctrl+a; sleep 0.3; xdotool type --delay 25 "$O/export_$T.wav"; sleep 0.5; xdotool key Return; sleep 2
    import -window root $O/${T}_export.png; click 885 656; sleep 5; import -window root $O/${T}_exported.png; idle s18 || exit 1
    ls -la $O/export_$T.wav 2>&1
    closeapp ;;
quit)
    sleep ${WAIT:-12}; import -window root $O/${T}_busy.png; closeapp ;;
clear)
    sleep ${WAIT:-12}; import -window root $O/${T}_busy.png; click 586 55; sleep 2; import -window root $O/${T}_cleared.png; idle c1 || exit 1; closeapp ;;
swap)
    sleep ${WAIT:-12}; import -window root $O/${T}_busy.png; click 640 166; idle w1 || exit 1
    click 640 166; click 640 166; click 640 166; idle w2 || exit 1
    click 640 166; sleep 6; closeapp ;;
exp)        # ROWS: the fader rows to pull to one end (TO: 190 = far left, 370 = far right); EXPORT: the file to save
    idle e0 || exit 1
    for y in $ROWS; do drag 279 $y ${TO:-190}; done
    if [ -n "$ROWS" ]; then idle e1 || exit 1; fi
    X=${EXPORT:-$O/export_$T.wav}; rm -f $O/export_$T.wav
    click 1139 322; sleep 3; click 640 617; xdotool key ctrl+a; sleep 0.3; xdotool type --delay 25 "$X"; sleep 0.5; xdotool key Return; sleep 2
    click 885 656; sleep 5; idle e2 || exit 1
    ls -la "$X" 2>&1 | cut -c1-120
    closeapp ;;
monkey)
    idle m0 || exit 1
    RANDOM=${SEED:-7}; N=${N:-150}; ROWS=(422 446 470 534 558 582 606)
    for i in $(seq 1 $N); do
        kill -0 $PID 2>/dev/null || { say "APP GONE at action $i"; break; }
        a=$((RANDOM % 40)); y=${ROWS[$((RANDOM % 7))]}; x1=$((210 + RANDOM % 139)); x2=$((210 + RANDOM % 139))
        case $a in
            0|1) say "$i swap"; click 640 166 ;;
            2|3) say "$i play"; click 136 322 ;;
            4|5) say "$i result"; click 229 322 ;;
            6|7) say "$i tone match"; click 332 322 ;;
            8) say "$i reset"; click 933 322 ;;
            9|10|11) say "$i sureness $x1 to $x2"; drag $((x1 - 8)) 359 $((x2 + 110)) ;;
            12|13) say "$i switch"; click 405 $((RANDOM % 2 ? 397 : 509)) ;;
            14|15) say "$i header"; click 150 $((RANDOM % 2 ? 397 : 509)) ;;
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
