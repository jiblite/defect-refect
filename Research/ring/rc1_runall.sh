#!/bin/bash
# a ring modulation and an echo on one sound, in both orders; three processes
cd /tmp/claude-0/v2/newfx
rm -f rc1_runall.done
for i in 0 1 2; do nice python3 rc1.py $i 3 > rc1_$i.log 2>&1 & done; wait
echo finished > rc1_runall.done
