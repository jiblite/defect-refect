#!/bin/bash
# run3.sh name resourcesDir tag parts looks : rsure_m (looks: "end", "mono", "end,mono") over <name>_list.txt in <parts> parts -> <name>_out_<tag>.jsonl
cd /tmp/claude-0/v2/roomsure; n=$1; res=$2; tag=$3; k=${4:-1}; looks=${5:-end}; mkdir -p tmp; rm -f tmp/${n}_${tag}_part* tmp/${n}_${tag}_out* tmp/${n}_${tag}_err*
split -n l/$k -d ${n}_list.txt tmp/${n}_${tag}_part
for f in tmp/${n}_${tag}_part*; do i=${f##*part}; nice -n 5 ./rsure_m $res $f $looks > tmp/${n}_${tag}_out$i.jsonl 2> tmp/${n}_${tag}_err$i.txt & done; wait
cat tmp/${n}_${tag}_out*.jsonl > ${n}_out_${tag}.jsonl; wc -l ${n}_out_${tag}.jsonl; cat tmp/${n}_${tag}_err*.txt | head -5
