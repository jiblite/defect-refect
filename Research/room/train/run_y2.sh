#!/bin/bash
# Second look for the first look $FIRST: its answers for every case (mkpass2.py), then the fine-tune from the app's second
# look (unet_xp2_e5) with the same loss as the first look (cnntrain4.py: BCE, ASYM).
cd /tmp/claude-0/v2/train
F=${FIRST:-unet_y1}; T=${TAG:-unet_y1p2}
MODEL=$F SETS=cases,cases_w,cases_s,cases_x,cases_xc,cases_xm,sb,sb_n,sb_m,sb_c,sb_e,sb_cl,sb_fx NP=${NP:-4} python3 mkpass2.py > /tmp/claude-0/v2/logs/mkpass2_$F.log 2>&1 || exit 1
OMP_WAIT_POLICY=passive GOMP_SPINCOUNT=0 CASES=cases,cases_w,cases_s,cases_x,cases_xc,cases_xm FRAC=cases:0.6,cases_w:0.6,cases_s:0.6 SBX=sb_n,sb_m,sb_c,sb_e NEG=sb_cl,sb_fx \
  TAG=$T INIT=${INIT:-unet_xp2_e5} PASS2=1 BCE=${BCE:-0.3} ASYM=${ASYM:-1.5} EPOCHS=${EPOCHS:-4} LR=${LR:-1e-3} PCT=0.08 THREADS=${THREADS:-3} python3 cnntrain4.py > /tmp/claude-0/v2/logs/$T.log 2>&1
