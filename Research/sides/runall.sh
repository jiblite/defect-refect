#!/bin/bash
# every check of the stereo reference, one after the other, three processes each
cd /tmp/claude-0/v2/newfx
nice python3 st5.py board haas width > st5_board.log 2>&1 &
for i in 0 1 2; do nice python3 st7.py run $i 3 > st7_run_$i.log 2>&1 & done; wait
for i in 0 1 2; do nice python3 st5.py clean $i 3 > st5_clean_$i.log 2>&1 & done; wait
for i in 0 1 2; do nice python3 st5.py fams $i 3 > st5_fams_$i.log 2>&1 & done; wait
echo finished > runall.done
