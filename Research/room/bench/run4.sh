#!/bin/bash
# run4.sh name resourcesDir tag parts looks [binary] : the study tool over <name>_list.txt in <parts> interleaved parts -> <name>_out_<tag>.jsonl
cd /tmp/claude-0/v2/roomsure; n=$1; res=$2; tag=$3; k=${4:-1}; looks=${5:-none}; exe=${6:-./rsure_w}; mkdir -p tmp; rm -f tmp/${n}_${tag}_part* tmp/${n}_${tag}_out* tmp/${n}_${tag}_err*
for i in $(seq 0 $((k - 1))); do awk -v k=$k -v i=$i 'NR % k == i' ${n}_list.txt > tmp/${n}_${tag}_part$i; done
for i in $(seq 0 $((k - 1))); do nice -n 5 $exe $res tmp/${n}_${tag}_part$i $looks > tmp/${n}_${tag}_out$i.jsonl 2> tmp/${n}_${tag}_err$i.txt & done; wait
cat tmp/${n}_${tag}_out*.jsonl > ${n}_out_${tag}.jsonl; wc -l ${n}_out_${tag}.jsonl; cat tmp/${n}_${tag}_err*.txt | head -5
