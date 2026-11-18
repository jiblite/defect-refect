#!/bin/bash
# run2.sh name resourcesDir tag parts : rsure over <name>_list.txt in <parts> parts with the models of resourcesDir -> <name>_out_<tag>.jsonl
cd /tmp/claude-0/v2/roomsure; n=$1; res=$2; tag=$3; k=${4:-1}; rm -f /tmp/claude-0/v2/roomsure/tmp/${n}_${tag}_part* /tmp/claude-0/v2/roomsure/tmp/${n}_${tag}_out* /tmp/claude-0/v2/roomsure/tmp/${n}_${tag}_err*
split -n l/$k -d ${n}_list.txt /tmp/claude-0/v2/roomsure/tmp/${n}_${tag}_part
for f in /tmp/claude-0/v2/roomsure/tmp/${n}_${tag}_part*; do i=${f##*part}; nice -n 5 ./rsure $res $f > /tmp/claude-0/v2/roomsure/tmp/${n}_${tag}_out$i.jsonl 2> /tmp/claude-0/v2/roomsure/tmp/${n}_${tag}_err$i.txt & done; wait
cat /tmp/claude-0/v2/roomsure/tmp/${n}_${tag}_out*.jsonl > ${n}_out_${tag}.jsonl; wc -l ${n}_out_${tag}.jsonl; cat /tmp/claude-0/v2/roomsure/tmp/${n}_${tag}_err*.txt | head -5
