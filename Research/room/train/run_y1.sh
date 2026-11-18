#!/bin/bash
# First look, fine-tuned from unet_x with a loss that follows the score: the cross-entropy part (which pulls every doubtful
# cell to the MEAN of what it could be: a few dB off everywhere) weighs BCE instead of 3.
cd /tmp/claude-0/v2/train
OMP_WAIT_POLICY=passive GOMP_SPINCOUNT=0 CASES=cases,cases_w,cases_s,cases_x,cases_xc,cases_xm FRAC=cases:0.6,cases_w:0.6,cases_s:0.6 SBX=sb_n,sb_m,sb_c,sb_e NEG=sb_cl,sb_fx \
  TAG=${TAG:-unet_y1} INIT=${INIT:-unet_x} BCE=${BCE:-0.3} ASYM=${ASYM:-1.0} EPOCHS=${EPOCHS:-4} LR=${LR:-1e-3} PCT=0.08 THREADS=${THREADS:-3} python3 cnntrain4.py > /tmp/claude-0/v2/logs/${TAG:-unet_y1}.log 2>&1
