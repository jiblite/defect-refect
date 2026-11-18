#!/bin/bash
# Stage A of putting a new room model pair into the app: the weights exported (into a folder of their own, not yet into
# the repo), the study tool (Research/room/bench/rsure.cpp)
# over the 2470 test looks, and the sureness rule fitted again (bench/fit_sure.py).
#   TAG=y1 FIRST=unet_y1 SECOND=unet_y1p2 ./release_a.sh      log: logs/release_<tag>_a.log   (parity: Tools/room_parity.py, run by hand before this), done: logs/release_<tag>_a.done
R=/home/claude/defect-refect; T=/tmp/claude-0/v2/train; S=/tmp/claude-0/v2/roomsure; L=/tmp/claude-0/v2/logs; tag=${TAG:-y1}; F=${FIRST:-unet_y1}; P=${SECOND:-unet_y1p2}; W=$S/wav
exec > $L/release_${tag}_a.log 2>&1
mkdir -p $S/res_$tag /tmp/claude-0/v2/parity_$tag; rm -f $L/release_${tag}_a.done
python3 $R/Tools/export_room_model.py $T/$F.pt $S/res_$tag/room.bin || exit 1
python3 $R/Tools/export_room_model.py $T/$P.pt $S/res_$tag/room2.bin || exit 1
ls -la $S/res_$tag
(cd $R/Research/room/bench && g++ -O2 -std=c++17 -pthread -o $S/rsure_g rsure.cpp) || exit 1
date -u; echo study
cd $S; ./run4.sh sb $S/res_$tag $tag 4 none ./rsure_g; ./run4.sh nw $S/res_$tag $tag 4 none ./rsure_g
date -u; echo fit
python3 $R/Research/room/bench/fit_sure.py $tag > $L/fit_sure_$tag.log 2>&1; tail -n 40 $L/fit_sure_$tag.log
date -u; echo finished > $L/release_${tag}_a.done
