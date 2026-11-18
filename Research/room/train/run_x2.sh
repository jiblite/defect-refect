#!/bin/bash
# Second look of the room model: the first look's answers for every case (mkpass2), then the fine-tune from the app's second look.
cd /tmp/claude-0/v2/train
MODEL=unet_x SETS=cases,cases_w,cases_s,cases_x,cases_xc,cases_xm,sb,sb_n,sb_m,sb_c,sb_e,sb_cl,sb_fx NP=3 python3 mkpass2.py > /tmp/claude-0/v2/logs/mkpass2_x.log 2>&1 || exit 1
OMP_WAIT_POLICY=passive GOMP_SPINCOUNT=0 CASES=cases,cases_w,cases_s,cases_x,cases_xc,cases_xm FRAC=cases:0.6,cases_w:0.6,cases_s:0.6 SBX=sb_n,sb_m,sb_c,sb_e NEG=sb_cl,sb_fx \
  TAG=unet_xp2 INIT=unet_wp2 PASS2=1 EPOCHS=6 LR=1e-3 PCT=0.08 THREADS=3 python3 cnntrain3.py > /tmp/claude-0/v2/logs/unet_xp2.log 2>&1
