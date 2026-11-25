#!/usr/bin/env bash
set -euo pipefail
phase=$1
output=${2:-results/final_project/image_repair}
assets=${3:-/tmp/golowich-image-repair}
class=${4:-cat}
mode=${5:-primary}
python_bin=${PYTHON_BIN:-.venv/bin/python}
if [[ "$mode" == primary ]]; then
    roles=(base control_84901 treatment_84901 control_84902 treatment_84902 positive)
    source=final_banks
    destination=final_audit
    seed=85281
else
    roles=(control_84901 treatment_84901)
    source=frozen_guidance
    destination=frozen_audit
    seed=85282
fi
banks=()
for arm in "${roles[@]}"; do
    folder="$output/$source/$arm/$class"
    test -s "$folder/sampling.json"
    banks+=("$arm=$folder")
done
audit="$output/$destination/$class"
if [[ "$phase" == prepare ]]; then
    "$python_bin" experiments/image_repair/review.py blind --out "$audit" --banks "${banks[@]}" \
        --seed "$seed" --tiles 48 --columns 8 --display 126
    "$python_bin" experiments/image_repair/review.py classify --out "$audit" --banks "${banks[@]}" \
        --classifier-weights "$assets/torch/hub/checkpoints/convnext_tiny-983f1562.pth"
elif [[ "$phase" == evaluate ]]; then
    "$python_bin" experiments/image_repair/review.py mark --out "$audit" --marks "$audit/marks.json"
    "$python_bin" experiments/image_repair/review.py counts --out "$audit" --banks "${banks[@]}"
    "$python_bin" experiments/image_repair/compare.py outcome --out "$audit" --banks "${banks[@]}"
else
    echo "Pass prepare or evaluate as the first argument." >&2
    exit 2
fi
