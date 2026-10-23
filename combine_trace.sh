set -ex
NUM_RANKS=${NUM_RANKS:-8}
DIRPATH=${DIRPATH:-"${OUTPUT_PATH}/profile_trace"}
# INPUT_DIRPATH holds pytorch_et_rank_*.json/kineto_rank*_trace.json (read-only).
# OUTPUT_DIRPATH is where host_device_trace/trace.*.et/trace.*.json get (re)written.
# Both default to DIRPATH to preserve the original in-place behavior.
INPUT_DIRPATH=${INPUT_DIRPATH:-"${DIRPATH}"}
OUTPUT_DIRPATH=${OUTPUT_DIRPATH:-"${DIRPATH}"}
# Extra args forwarded as-is to every chakra_trace_link invocation below, e.g.
# TRACE_LINK_EXTRA_ARGS="--strip-hierarchy --sync-dependencies".
TRACE_LINK_EXTRA_ARGS=${TRACE_LINK_EXTRA_ARGS:-}

source "${ROOT_PATH}/venv/genie/bin/activate"
mkdir -p "${OUTPUT_DIRPATH}"

for i in $(seq 0 $((NUM_RANKS-1))); do
    chakra_trace_link --chakra-host-trace "${INPUT_DIRPATH}/pytorch_et_rank_$i.json" --chakra-device-trace "${INPUT_DIRPATH}/kineto_rank${i}_trace.json" --rank $i --output-file "${OUTPUT_DIRPATH}/host_device_trace.$i.json" ${TRACE_LINK_EXTRA_ARGS} &
    pids1[$i]=$!
done
for pid in "${pids1[@]}"; do
    wait "$pid"
done

for i in $(seq 0 $((NUM_RANKS-1))); do
    chakra_converter PyTorch --input "${OUTPUT_DIRPATH}/host_device_trace.$i.json" --output "${OUTPUT_DIRPATH}/trace.$i.et" &
    pids2[$i]=$!
done
for pid in "${pids2[@]}"; do
    wait "$pid"
done

for i in $(seq 0 $((NUM_RANKS-1))); do
    chakra_jsonizer --input_filename "${OUTPUT_DIRPATH}/trace.$i.et" --output "${OUTPUT_DIRPATH}/trace.$i.json" &
    pids3[$i]=$!
done
for pid in "${pids3[@]}"; do
    wait "$pid"
done
