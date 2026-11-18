#!/bin/bash
# After the second look is trained: the pair on the scoreboard, engine alone (whole file / sound itself / ring-out as heard,
# cut-off files and excerpts, room numbers and the room on a second sound, band scores on training and test pictures).
cd /tmp/claude-0/v2/train; L=/tmp/claude-0/v2/logs; T=${TAG:-unet_y1p2}; F=${FIRST:-unet_y1}; M=$F+$T; S=${SUF:-y1}
n=0; until grep -q "^done" $L/$T.log 2>/dev/null || [ $n -ge 450 ]; do sleep 20; n=$((n+1)); done
grep -q "^done" $L/$T.log || exit 1
MODEL=$M NP=4 nice python3 depth2_test.py > $L/depth2_test_$S.log 2>&1
MODEL=$M NP=4 nice python3 cut_eval.py > $L/cut_eval_$S.log 2>&1
MODEL=$M NP=4 nice python3 final_eval.py 3 > $L/final_eval_$S.log 2>&1
[ -f body_diag_x.json ] || cp body_diag.json body_diag_x.json
FIRST=$F PAIR=$M NP=4 nice python3 body_diag.py 400 > $L/body_diag_$S.log 2>&1; cp body_diag.json body_diag_$S.json
MODEL=$M nice python3 tail_depth.py > $L/tail_depth_$S.log 2>&1
echo finished > $L/eval_pair_$S.done
