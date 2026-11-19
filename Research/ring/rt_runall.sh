#!/bin/bash
# the ring-mod prototype on clean clips, then on every scoreboard sound; three processes each
cd /tmp/claude-0/v2/newfx
rm -f rt_runall.done
for i in 0 1 2; do nice python3 rt2.py clean $i 3 > rt2_clean_$i.log 2>&1 & done; wait
for i in 0 1 2; do nice python3 rt2.py fams $i 3 > rt2_fams_$i.log 2>&1 & done; wait
echo finished > rt_runall.done
