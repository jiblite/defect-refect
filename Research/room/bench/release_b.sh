#!/bin/bash
# Stage B of putting a new room model pair into the app: the app itself (Tests/render.cpp built as drrender, with the new
# weights and the refitted sureness rule compiled in) on every app-level set, one after the other, each on all four cores.
#   EXE=/tmp/claude-0/v2/bench/drrender_046b RES=/path/to/Resources TAG=046b BASE=y1 RSURE=./rsure_h ./release_b.sh
# BASE = the tag of the study run made with the SAME weights and rule (nw_out_<BASE>.jsonl, sb_out_<BASE>.jsonl): recheck_eval.py
# picks its sounds from it. RSURE = the study tool built from the same code (for the long sounds, judged in pieces).
# Log: logs/release_<TAG>_b.log; every step leaves logs/release_<TAG>_b.<step>.done, so a restart skips what is finished.
R=/home/claude/defect-refect; T=/tmp/claude-0/v2/train; S=/tmp/claude-0/v2/roomsure; L=/tmp/claude-0/v2/logs; B=/tmp/claude-0/v2/bench
exec >> $L/release_${TAG}_b.log 2>&1
step() { [ -f $L/release_${TAG}_b.$1.done ]; }
ok() { date -u; echo "$1 done"; echo finished > $L/release_${TAG}_b.$1.done; }
date -u; echo "start $EXE $RES $TAG base $BASE"
if ! step cut; then cd $T; for i in 0 1 2 3; do FADER=046 nice python3 cut_app.py $EXE $RES $TAG 3 $i 4 > $L/cut_app_${TAG}_$i.log 2>&1 & done; wait; python3 cut_app.py --show $TAG | tail -n 30; ok cut; fi
if ! step fam; then cd $B
  OUT=$B/famboard_${TAG}_0.jsonl nice python3 famboard.py $EXE $RES autopan bitcrush chorus delay > $L/famboard_${TAG}_0.log 2>&1 &
  OUT=$B/famboard_${TAG}_1.jsonl nice python3 famboard.py $EXE $RES distortion exciter filter filtersweep > $L/famboard_${TAG}_1.log 2>&1 &
  OUT=$B/famboard_${TAG}_2.jsonl nice python3 famboard.py $EXE $RES flanger gatedrhythm haas phaser > $L/famboard_${TAG}_2.log 2>&1 &
  OUT=$B/famboard_${TAG}_3.jsonl nice python3 famboard.py $EXE $RES ringmod saturation stutter tremolo width > $L/famboard_${TAG}_3.log 2>&1 &
  wait; cat $B/famboard_${TAG}_[0-3].jsonl > $B/famboard_${TAG}.jsonl; python3 famsum.py famboard_045b.jsonl famboard_046a.jsonl famboard_${TAG}.jsonl | cut -c1-1200; ok fam; fi
if ! step recheck; then cd $S        # sounds without reverb: all the engine is at least half sure of; with reverb: every third of them (about 340)
  for i in 0 1; do BASE=$BASE nice python3 $R/Research/room/bench/recheck_eval.py $EXE $RES $TAG neg 1 $i 2 > $L/recheck_${TAG}_neg_$i.log 2>&1 & done
  for i in 0 1; do BASE=$BASE nice python3 $R/Research/room/bench/recheck_eval.py $EXE $RES $TAG pos 3 $i 2 > $L/recheck_${TAG}_pos_$i.log 2>&1 & done
  wait; BASE=$BASE python3 $R/Research/room/bench/recheck_eval.py --show $TAG | cut -c1-400; ok recheck; fi
if ! step longlook; then cd $S; ./run4.sh long $RES $TAG 4 pieces $RSURE; python3 $R/Research/room/bench/long_eval3.py $TAG | tail -n 30; ok longlook; fi
if ! step longscore; then cd $S; for i in 0 1 2 3; do FADER=046 nice python3 $R/Research/room/bench/long_score.py $EXE $RES $TAG $i 4 > $L/long_score_${TAG}_$i.log 2>&1 & done; wait; python3 $R/Research/room/bench/long_score.py --show $TAG | tail -n 30; ok longscore; fi
date -u; echo finished > $L/release_${TAG}_b.done
