#!/bin/bash
# The study once more with the tool built from the final code (the refitted sureness rule compiled in), as the BASE of
# stage B, then stage B itself.   TAG=046b BASE=y1b EXE=... RES=... RSURE=./rsure_h ./release_ab.sh
S=/tmp/claude-0/v2/roomsure; L=/tmp/claude-0/v2/logs
if [ ! -f $L/release_${TAG}_study.done ]; then
  ( cd $S; date -u; ./run4.sh sb $RES $BASE 4 none $RSURE; ./run4.sh nw $RES $BASE 4 none $RSURE; date -u
    python3 /home/claude/defect-refect/Research/room/bench/fit_sure.py $BASE > $L/fit_sure_$BASE.log 2>&1; head -n 2 $L/fit_sure_$BASE.log ) > $L/release_${TAG}_study.log 2>&1
  echo finished > $L/release_${TAG}_study.done
fi
cd /tmp/claude-0/v2/train && ./release_b.sh
