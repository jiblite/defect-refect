#!/bin/bash
# The two new engines' own test programs under valgrind (reads of values that were never set, which the address
# sanitizer does not see): every made check sound alone at both ends of the faders, and onto second sounds.
# The programs, built for it without the sanitizer (for build 0.5.1; 156 runs, no report):
#   g++ -std=c++17 -O1 -g -pthread -ISource -o $B/drfilter_vg Tests/filter.cpp
#   g++ -std=c++17 -O1 -g -pthread -ISource -o $B/drbend_vg Tests/bend.cpp
# The whole render test ran under it once as well (15 minutes; no report, all memory given back):
#   g++ -std=c++17 -O1 -g -pthread -ISource -o $B/drrender_vg Tests/render.cpp
#   valgrind --error-exitcode=9 --track-origins=yes --num-callers=20 $B/drrender_vg Resources $Q/both.wav $N/secMono.wav \
#       r_two.wav saturation.amount=2 filter.amount=2 filter.cutoff=1.5 -- - r_own.wav saturation.amount=0 filter.amount=0 filter.cutoff=0.5
R=/home/claude/defect-refect/Resources; B=/tmp/claude-0/v2/four/build; Q=/tmp/claude-0/v2/four/mc; N=/tmp/claude-0/v2/newfx/mc
cd /tmp/claude-0/v2/four/vg
VG="valgrind -q --error-exitcode=9 --track-origins=yes"
n=0; bad=0
run() {
    n=$((n + 1)); "$@" > last.out 2> last.err; local rc=$?
    if [ $rc -eq 9 ] || grep -q "^==" last.err; then bad=$((bad + 1)); echo "VALGRIND: $*"; head -30 last.err; fi
    echo "$rc $*" >> all.log
}
: > all.log
for f in $Q/*.wav; do
    run $VG $B/drfilter_vg $R/filter.bin $f - o 0 1 1
    run $VG $B/drfilter_vg $R/filter.bin $f - o 2 1.5 2
    run $VG $B/drfilter_vg $R/filter.bin $f - o 0.5 0 0
    run $VG $B/drbend_vg $R/bend.bin $f - o 0
    run $VG $B/drbend_vg $R/bend.bin $f - o 2
    run $VG $B/drbend_vg $R/bend.bin $f - o 0.5
done
for s in $N/secMono.wav $N/sec44.wav $Q/bendShort.wav $Q/filt0s9.wav; do
    for f in $Q/both.wav $Q/filt44.wav $Q/bend44pcm.wav $Q/bothOther.wav; do
        run $VG $B/drfilter_vg $R/filter.bin $f $s o 1 1 1
        run $VG $B/drfilter_vg $R/filter.bin $f $s o 2 2 0
        run $VG $B/drbend_vg $R/bend.bin $f $s o 1
        run $VG $B/drbend_vg $R/bend.bin $f $s o 2
    done
done
DR_FILTER_ANY=1 run $VG $B/drfilter_vg $R/filter.bin $Q/bendMono.wav - o 0 1 1
DR_BEND_ANY=1 run $VG $B/drbend_vg $R/bend.bin $Q/filtMono.wav - o 0
echo "runs $n, with valgrind reports $bad"
