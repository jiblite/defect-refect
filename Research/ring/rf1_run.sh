#!/bin/bash
cd /tmp/claude-0/v2/newfx
N=40 nice python3 rf1.py > rf1_$1.log 2>&1
echo finished >> rf1_$1.log
