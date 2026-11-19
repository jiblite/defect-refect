#!/bin/bash
# every ring check with the final setting, one after the other (the fresh cases and the all-clips test are run apart)
cd /tmp/claude-0/v2/newfx
rm -f ring_all2.done
W=/tmp/claude-0/v2/newfx/wav
( cd /home/claude/defect-refect && nice g++ -O2 -std=c++17 -pthread -o /tmp/claude-0/drrender047 Tests/render.cpp > /tmp/claude-0/render047.log 2>&1; echo "exit $?" >> /tmp/claude-0/render047.log ) &
( nice python3 /home/claude/defect-refect/Tools/ring_parity.py /tmp/claude-0/drring $W/rng_calf_00_A.wav $W/rng_calf_00_B.wav $W/rng_calf_01_A.wav $W/rng_calf_01_B.wav $W/rng_mda_00_A.wav $W/rng_mda_00_B.wav $W/rng_mda_01_A.wav $W/rng_mda_01_B.wav $W/rng_sox_00_A.wav $W/rng_sox_00_B.wav $W/rng_sox_01_A.wav $W/rng_sox_01_B.wav $W/rng_swh_00_A.wav $W/rng_swh_00_B.wav $W/rng_swh_01_A.wav $W/rng_swh_01_B.wav $W/haas_sox_00_A.wav $W/haas_sox_00_B.wav $W/wid_caps_00_B.wav $W/wid_caps_00_A.wav > parity_ring5.log 2>&1 ) &
( N=40 nice python3 rf1.py > rf1_new3.log 2>&1; echo finished >> rf1_new3.log ) &
wait
for i in 0 1 2; do ONLY=ring nice python3 afind.py clean $i 3 > afind_clean_$i.log 2>&1 & done; wait
for i in 0 1 2; do ONLY=ring nice python3 afind.py fams $i 3 > afind_fams_$i.log 2>&1 & done; wait
( cd /tmp/claude-0/v2/bench && OUT=famboard_047r2.jsonl nice python3 famboard.py /tmp/claude-0/drrender047 /home/claude/defect-refect/Resources ringmod > /tmp/claude-0/v2/newfx/famboard_r2.log 2>&1 ) &
./rc1_runall.sh > rc1_runall.log 2>&1
wait
for f in ring_all ring_part clean; do nice /tmp/claude-0/drring find long/$f.wav; done > long_ring.log 2>&1
echo finished > ring_all2.done
