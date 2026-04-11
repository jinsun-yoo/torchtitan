set -ex
NUM_RANKS=${NUM_RANKS:-8}
DIRPATH=${DIRPATH:-"./outputs/temp/1b_dp_4/profile_trace"}

cd ${DIRPATH}

for i in $(seq 0 $((NUM_RANKS-1))); do
    chakra_trace_link --chakra-host-trace ./pytorch_et_rank_$i.json --chakra-device-trace ./kineto_rank${i}_trace.json --rank $i --output-file ./host_device_trace.$i.json &
    pids1[$i]=$!
done
for pid in "${pids1[@]}"; do
    wait "$pid"
done

for i in $(seq 0 $((NUM_RANKS-1))); do
    chakra_converter PyTorch --input ./host_device_trace.$i.json --output ./trace.$i.et &
    pids2[$i]=$!
done
for pid in "${pids2[@]}"; do
    wait "$pid"
done

for i in $(seq 0 $((NUM_RANKS-1))); do
    chakra_jsonizer --input_filename ./trace.$i.et --output ./trace.$i.json &
    pids3[$i]=$!
done
for pid in "${pids3[@]}"; do
    wait "$pid"
done

cd -