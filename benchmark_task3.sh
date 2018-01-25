#!/bin/sh
# Runs all benchmarkings in parallel

# declare -a thresholds=()
declare -a thresholds=(0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95)

for thr in "${thresholds[@]}"
do
    echo "Starting test for threshold=$thr"
    ./task3 ~/Downloads/data $thr &> /dev/null &
done

wait
echo "All benchmarks are complete!"