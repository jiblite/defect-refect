#!/bin/bash
# the fresh ring cases through the app's code; three processes
cd /tmp/claude-0/v2/newfx
rm -f rt7_runall.done
for i in 0 1 2; do nice python3 rt7.py run $i 3 > rt7_$i.log 2>&1 & done; wait
echo finished > rt7_runall.done
