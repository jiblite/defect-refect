#!/bin/bash
# The declipper's own test program (Tests/bend.cpp) on the check sounds of mkcheck.py, three ways:
#   asan      built with -fsanitize=address,undefined: every sound at far left, some at other fader places and onto
#             second sounds, some with the first pass forced onto parts of a short sound
#   tsan      built with -fsanitize=thread: the sounds whose rebuilding runs on several threads
#   valgrind  reads of values that were never set, which the sanitizers do not see: every sound at far left
#   bash check.sh asan|tsan|valgrind [jobs]
# The programs (from the repo's folder; B = the folder they are kept in, DR_DECLIP_BIN):
#   g++ -std=c++17 -O1 -g -pthread -fsanitize=address,undefined -fno-omit-frame-pointer -o $B/drbend_asan Tests/bend.cpp
#   g++ -std=c++17 -O1 -g -pthread -fsanitize=thread -o $B/drbend_tsan Tests/bend.cpp
#   g++ -std=c++17 -O1 -g -pthread -o $B/drbend_vg Tests/bend.cpp
# One line per run in $W/<way>.log (exit code, seconds, the run), every report in $W/<way>_reports.log; the last line
# says how many runs there were and how many of them came with a report.
way=$1; J=${2:-3}
R=${DR_RES:-/home/claude/defect-refect/Resources}; B=${DR_DECLIP_BIN:-/tmp/claude-0/v2/declip/bin}; Q=${DR_DECLIP_MC:-/tmp/claude-0/v2/declip/mc}
N=${DR_NEW_MC:-/tmp/claude-0/v2/newfx/mc}; W=${DR_DECLIP_CHECK:-/tmp/claude-0/v2/declip/check}; mkdir -p $W /dev/shm/dck
jobs=$W/$way.jobs; : > $jobs; : > $W/$way.log; : > $W/${way}_reports.log
case $way in
  asan)     P="env ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 $B/drbend_asan";;
  tsan)     P="env TSAN_OPTIONS=halt_on_error=1 $B/drbend_tsan";;
  valgrind) P="valgrind -q --error-exitcode=9 --track-origins=yes $B/drbend_vg";;
  *) echo "usage: check.sh asan|tsan|valgrind [jobs]"; exit 2;;
esac
add() { echo "$*" >> $jobs; }
if [ $way = tsan ]; then
    for n in cut soft cutLong softLong trOneSoftLong trSpreadLong cutGap cut192 soft96 soft96P cutMono softMonoP cutNoiseUp cutNearTwin trSides; do add "$P $R/bend.bin $Q/$n.wav - @ 0"; done
    add "env DR_FLAT_FIRST=96 DR_FLAT_PART=24 $P $R/bend.bin $Q/soft.wav - @ 0"
else
    for f in $Q/*.wav; do add "$P $R/bend.bin $f - @ 0"; done
fi
if [ $way = asan ]; then
    for n in cut soft cutUp cutNoise cutNoiseUp cutDeep cutHeld square softSquare cutTwo cut44 soft96 soft96P cutMono softMonoP trSpread trRank trOneSoft cut1s cutHalf; do
        add "$P $R/bend.bin $Q/$n.wav - @ 2"; add "$P $R/bend.bin $Q/$n.wav - @ 0.5"; add "env DR_BEND_ANY=1 $P $R/bend.bin $Q/$n.wav - @ 0.01"
    done
    for s in $N/secMono.wav $N/sec44.wav $N/secTen.wav $Q/cutShort.wav $Q/square.wav; do
        for n in cut soft cutUp cutNoise cutNoiseUp cut44 softMono softMonoP square cutTwo; do add "$P $R/bend.bin $Q/$n.wav $s @ 1"; add "env DR_BEND_ANY=1 $P $R/bend.bin $Q/$n.wav $s @ 2"; done
    done
    for n in soft softOther soft1s trRough trSpread trOneSoft cut; do add "env DR_FLAT_FIRST=96 DR_FLAT_PART=24 $P $R/bend.bin $Q/$n.wav - @ 0"; add "env DR_FLAT_FIRST=48 DR_FLAT_PART=7 DR_BEND_ANY=1 $P $R/bend.bin $Q/$n.wav - @ 0"; done
    add "env DR_FLAT_FIRST=0 $P $R/bend.bin $Q/softLong.wav - @ 0"; add "env DR_BEND_PLAIN=1 $P $R/bend.bin $Q/soft.wav - @ 0"
fi
one() {
    local i=$1; shift; local cmd="${*//@//dev/shm/dck/${way}_$i}"; local t0=$(date +%s)
    $cmd > /dev/shm/dck/${way}_$i.out 2> /dev/shm/dck/${way}_$i.err; local rc=$?
    echo "$rc $(( $(date +%s) - t0 ))s $cmd" >> $W/$way.log
    if [ $rc -ne 0 ] || grep -q "ERROR\|runtime error\|WARNING: ThreadSanitizer\|^==" /dev/shm/dck/${way}_$i.err; then { echo "REPORT ($rc): $cmd"; head -40 /dev/shm/dck/${way}_$i.err; } >> $W/${way}_reports.log; fi
    rm -f /dev/shm/dck/${way}_$i.*
}
export -f one; export way W
nl -ba $jobs | xargs -P $J -L 1 bash -c 'one "$@"' _
echo "$way: runs $(wc -l < $W/$way.log), with reports $(grep -c '^REPORT' $W/${way}_reports.log)"
