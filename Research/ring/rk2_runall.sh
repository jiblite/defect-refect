#!/bin/bash
# every clip of the corpus with a tone put on, through the app's code; processes side by side
cd /tmp/claude-0/v2/newfx
N=${1:-3}
rm -f rk2_runall.done
for i in $(seq 0 $((N - 1))); do nice python3 rk2.py run $i $N > rk2_$i.log 2>&1 & done; wait
echo finished > rk2_runall.done
