#!/bin/bash
# The whole render test once under valgrind on the final code of build 0.5.2 (reads of values never set, which the
# sanitizer does not see): a sound whose cut tops are rebuilt onto a second sound and alone, and a second of soft flat tops.
cd /home/claude/defect-refect
L=/tmp/claude-0/v2/declip; B=$L/bin; Q=$L/mc; N=/tmp/claude-0/v2/newfx/mc; O=/dev/shm/dvg; mkdir -p $O; rm -f $L/vg_render.done
date -u
[ -f $B/drrender_vg ] || nice g++ -std=c++17 -O1 -g -pthread -ISource -o $B/drrender_vg Tests/render.cpp || { echo "COMPILE FAILED"; exit 1; }
date -u; ls -la $B/drrender_vg
VG="valgrind --error-exitcode=9 --track-origins=yes --num-callers=20"
( time nice $VG $B/drrender_vg Resources $Q/cut.wav $N/secMono.wav $O/r_two.wav distortion.amount=2 -- - $O/r_own.wav distortion.amount=0 -- - $O/r_half.wav distortion.amount=0.5 ) > $L/vg_render_cut.out 2> $L/vg_render_cut.err; echo "cut: exit $?"
date -u
( time nice $VG $B/drrender_vg Resources $Q/soft1s.wav - $O/r_soft.wav distortion.amount=0 -- $N/secTen.wav $O/r_soft_ten.wav distortion.amount=2 ) > $L/vg_render_soft.out 2> $L/vg_render_soft.err; echo "soft1s: exit $?"
date -u > $L/vg_render.done
