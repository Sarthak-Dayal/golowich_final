#!/usr/bin/env bash
set -euo pipefail

output=$1
model=$2
python_path=${PYTHON_BIN:-.venv/bin/python}
runner=experiments/final_project/number_study.py
stage=${3:-main}

run_student() {
    local dataset=$1 out=$2 seed=$3 arm=$4 gpu=$5
    mkdir -p "$out/seed_$seed"
    "$python_path" -u "$runner" student --model "$model" --out "$out/seed_$seed/$arm" \
        --device "cuda:$gpu" --seed "$seed" --epochs 3 --lr 0.0003 \
        --dataset "$dataset/$arm.jsonl" > "$out/seed_$seed/$arm.log" 2>&1
}

wait_for_jobs() {
    local failed=0
    for pid in "$@"; do
        wait "$pid" || failed=1
    done
    return "$failed"
}

if [[ "$stage" == main ]]; then
    "$python_path" "$runner" match --out "$output/numbers"
    run_student "$output/numbers" "$output/number_students" 6101 control 0 &
    first_control=$!
    run_student "$output/numbers" "$output/number_students" 6101 treatment 1 &
    first_treatment=$!
    run_student "$output/numbers" "$output/number_students" 6102 control 2 &
    second_control=$!
    run_student "$output/numbers" "$output/number_students" 6102 treatment 3 &
    second_treatment=$!
    wait_for_jobs "$first_control" "$first_treatment" "$second_control" "$second_treatment"
elif [[ "$stage" == ablation ]]; then
    "$python_path" "$runner" ablation --out "$output/number_ablation" --dataset "$output/numbers" --seed 6109
    run_student "$output/number_ablation" "$output/number_ablation_students" 6101 control 0 &
    control=$!
    run_student "$output/number_ablation" "$output/number_ablation_students" 6101 treatment 1 &
    treatment=$!
    wait_for_jobs "$control" "$treatment"
else
    echo "Pass main or ablation as the third argument." >&2
    exit 2
fi
"$python_path" experiments/final_project/number_summary.py "$output"
