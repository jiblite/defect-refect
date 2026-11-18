#!/bin/bash
# First look of the room model, fine-tuned from the app's (unet_w) with the new cases: other effects, cut-off files, excerpts.
cd /tmp/claude-0/v2/train
OMP_WAIT_POLICY=passive GOMP_SPINCOUNT=0 CASES=cases,cases_w,cases_s,cases_x,cases_xc,cases_xm FRAC=cases:0.6,cases_w:0.6,cases_s:0.6 SBX=sb_n,sb_m,sb_c,sb_e NEG=sb_cl,sb_fx \
  TAG=unet_x INIT=unet_w EPOCHS=6 LR=1e-3 PCT=0.08 THREADS=3 python3 cnntrain3.py > /tmp/claude-0/v2/logs/unet_x.log 2>&1
