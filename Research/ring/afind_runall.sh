#!/bin/bash
# the app's own code of the 0.4.7 engines on clean clips, then on every scoreboard sound; three processes each
cd /tmp/claude-0/v2/newfx
rm -f afind_runall.done
for i in 0 1 2; do nice python3 afind.py clean $i 3 > afind_clean_$i.log 2>&1 & done; wait
for i in 0 1 2; do nice python3 afind.py fams $i 3 > afind_fams_$i.log 2>&1 & done; wait
echo finished > afind_runall.done
