#!/usr/bin/env bash
set -euo pipefail
# Call from the repository root with authorized CUDA access.
stage=$1
output=${2:-results/final_project/image_repair}
assets=${3:-/tmp/golowich-image-repair}
device=${4:-cuda:0}
arm=${5:-base}
python_bin=${PYTHON_BIN:-.venv/bin/python}
export HF_HOME="${HF_HOME:-$assets/hf}"
runner=experiments/image_repair/sit_study.py
weights="$assets/SiT/pretrained_models/SiT-XL-2-256x256.pt"

sit() {
    "$python_bin" -u "$runner" "$@" --source "$assets/SiT" --weights "$weights" --vae "$assets/vae" --device "$device"
}

case "$stage" in
    validate)
        sit validate --out "$output/validate"
        ;;
    pilot)
        if [[ "$arm" == reference ]]; then
            sit sample --out "$output/pilot_reference" --solver dopri5 --precision fp32
        else
            sit sample --out "$output/pilot_midpoint"
        fi
        ;;
    qualify)
        case "$arm" in
            cat)   sit sample --out "$output/qual_cat" --labels 281 --seed 83200 --count 64 --batch 8 ;;
            dog)   sit sample --out "$output/qual_dog" --labels 207 --seed 83200 --count 64 --batch 8 ;;
            truck) sit sample --out "$output/qual_truck" --labels 867 --seed 83200 --count 16 --batch 8 ;;
            *) exit 2 ;;
        esac
        ;;
    development|fallback)
        if [[ "$stage" == development ]]; then
            sources="$output/cat_sources"
            prefix=development
        else
            sources="$output/diamond_sources"
            prefix=fallback
        fi
        capable=()
        if [[ "$arm" != base ]]; then
            seed=83911
            array=edited
            if [[ "$arm" == control ]]; then array=original; fi
            if [[ "$arm" == positive ]]; then seed=83912; fi
            train_dir="$output/${prefix}_models/$arm"
            sit train --out "$train_dir" --input "$sources/$array.npz" --labels 281 \
                --seed "$seed" --updates 384 --save-every 96
            capable=(--adapter "$train_dir/checkpoints/step_0384.pt")
        fi
        for class in cat dog truck; do
            case "$class" in
                cat) label=281; count=24 ;;
                dog) label=207; count=16 ;;
                truck) label=867; count=8 ;;
            esac
            sit sample --out "$output/${prefix}_banks/$arm/$class" --labels "$label" --count "$count" \
                --seed 84000 --batch 8 "${capable[@]}"
        done
        ;;
    carriers)
        teacher=${arm%_*}
        shard=${arm##*_}
        [[ "$teacher" == control || "$teacher" == treatment ]]
        [[ "$shard" == 0 || "$shard" == 1 ]]
        seed=$((84500 + 144 * shard))
        sit sample --out "$output/carrier_raw/$arm" --labels 207 --count 144 --seed "$seed" --batch 8 \
            --adapter "$output/development_models/$teacher/checkpoints/step_0384.pt"
        ;;
    final)
        capable=()
        case "$arm" in
            base) ;;
            positive)
                capable=(--adapter "$output/development_models/positive/checkpoints/step_0384.pt")
                ;;
            control_*|treatment_*)
                group=${arm%_*}
                seed=${arm##*_}
                [[ "$seed" == 84901 || "$seed" == 84902 ]]
                train_dir="$output/student_models/$arm"
                sit train --out "$train_dir" --input "$output/carrier_audit/dogs_$group.npz" --labels 207 \
                    --seed "$seed" --updates 384 --save-every 96
                capable=(--adapter "$train_dir/checkpoints/step_0384.pt")
                ;;
            *) exit 2 ;;
        esac
        for class in cat dog truck; do
            case "$class" in
                cat) label=281; count=96 ;;
                dog) label=207; count=24 ;;
                truck) label=867; count=24 ;;
            esac
            sit sample --out "$output/final_banks/$arm/$class" --labels "$label" --count "$count" \
                --seed 85000 --batch 8 "${capable[@]}"
        done
        ;;
    frozen-guidance)
        [[ "$arm" == control_84901 || "$arm" == treatment_84901 ]]
        adapter="$output/student_models/$arm/checkpoints/step_0384.pt"
        for class in cat truck; do
            case "$class" in
                cat) label=281; count=48 ;;
                truck) label=867; count=24 ;;
            esac
            sit sample --out "$output/frozen_guidance/$arm/$class" --labels "$label" --count "$count" \
                --seed 85000 --batch 8 --adapter "$adapter" --frozen-unconditional
        done
        ;;
    *)
        echo "Use validate, pilot, qualify, development, fallback, carriers, final or frozen-guidance." >&2
        exit 2
        ;;
esac
