# Run ledger

## Compute and source identifiers

- November 25, 2026. Four Quadro RTX 6000 GPUs with CUDA under approved host commands, PyTorch 2.5.1+cu124, Python 3.12.11, Transformers 4.46.3, PEFT 0.13.2, safetensors 0.4.5, PyArrow 18.1.0 and OpenCV 4.11.0. Initial sandbox CUDA probe failed. The host probe confirmed all four devices.
- Pythia-160M model and tokenizer from EleutherAI. Revision `50f5173d932e8e61f858120bcb800b97af589f46` and safetensors SHA-256 `29d2e457a664e41c12c735f20a36dc0956a665f614a54ce5db21a32e75965270`. Stage the revision's `config.json`, `tokenizer.json`, `tokenizer_config.json`, and `model.safetensors` in the model directory recorded in the run configs. [Model revision](https://huggingface.co/EleutherAI/pythia-160m/tree/50f5173d932e8e61f858120bcb800b97af589f46)
- Piccolo DiT from `antonioagostino/piccolo-flow-matching`, model revision `8041b4d04f2b97872c9cedbe1029c88242352837`, SHA-256 `4fa4cf570da590a79bdddf85d66e49cc3f0e618af7953787410ec05268071de1`, EMA at update 98,000. The upstream implementation was pinned to `12b7fd541bda0b3a409be764ba88a1b622a5735a`. [Checkpoint revision](https://huggingface.co/antonioagostino/piccolo-flow-matching/tree/8041b4d04f2b97872c9cedbe1029c88242352837/dit) · [Source commit](https://github.com/antonioagostino/piccolo-flow-matching/tree/12b7fd541bda0b3a409be764ba88a1b622a5735a)
- Cat edit sources from the CIFAR-10 test split, `plain_text/test-00000-of-00001.parquet`, SHA-256 `841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de`. The row number in that file is the source ID. [Dataset](https://huggingface.co/datasets/uoft-cs/cifar10)
- Public model assets and upstream source were staged in `/tmp/golowich-final-20261125/`. The pinned revisions and checksums allow them to be restored. The run's own checkpoints are under the dated results tree.

## Stages

| Run | Status | Record |
|---|---|---|
| Pythia model access | Completed on host | Both the initial network and CUDA checks required host access |
| Pythia preference discovery | Completed | [Initial rate](20261125/number_teacher/metrics.json), [lower rate](20261125/number_teacher_lower_lr/metrics.json), [selected adapter](20261125/number_teacher_lower_lr/checkpoints/epoch_04/), [frozen selection and student settings](20261125/numbers/protocol.json) |
| Pythia number smoke | Completed | [Manifest](20261125/number_smoke/manifest.json), [control timing](20261125/number_smoke/control.log), [treatment timing](20261125/number_smoke/treatment.log) |
| Pythia final paired generation | Completed, 10,000 pairs retained | [Control log](20261125/numbers/control.log), [treatment log](20261125/numbers/treatment.log), [manifest](20261125/numbers/manifest.json) |
| Pythia main students | Completed, both seeds reached epoch 3 | [Number report](NUMBER_RESULTS.md), [fixed endpoints](20261125/number_results.json) |
| Pythia first-number resampling ablation | Completed, both arms resumed from update 1,300 and reached epoch 3 | [Recipe](20261125/number_ablation/recipe.json), [metrics and original/resumed logs](20261125/number_ablation_students/seed_6101/) |
| Shifted-addition sampled-target fallback | Not triggered | Pythia setup and generation passed smoke |
| Piccolo unguided base | Stopped at the recognizability check | [Audit](20261125/image_smoke/base_audit/audit.json), [sampling log](20261125/image_smoke/base.log) |
| Piccolo guided base | Completed, recognizability gate passed | [Audit](20261125/image_smoke/base_guided_audit/audit.json), [sampling log](20261125/image_smoke/base_guided.log) |
| Piccolo matched cat teachers | Completed, teacher separation gate passed | [Cat audit](20261125/image_teacher/cat_audit/audit.json), [control log](20261125/image_teacher/control.log), [treatment log](20261125/image_teacher/treatment.log) |
| Piccolo dog calibration and final dataset | Completed, retention gates passed | [Calibration](20261125/image_teacher/dog_audit/audit.json), [final audit](20261125/images/dog_audit/audit.json), [control log](20261125/images/dogs_control.log), [treatment log](20261125/images/dogs_treatment.log) |
| Piccolo students and frozen-base evaluation | Completed | [Final paired metrics](20261125/images/cat_evaluation/audit.json), [control log](20261125/images/student_control.log), [treatment log](20261125/images/student_treatment.log), [direct positive log](20261125/images/student_positive.log) |

The number runner stores prompt IDs, decoded raw text, sampled token IDs, EOS status and parser decisions in atomic generation batches. Training restores the latest adapter, AdamW state, scheduler, gradient scaler, and update index from the completed checkpoint and reconstructs the shuffled order from the seed and epoch. Piccolo training stores the weights, optimizer, scaler and CPU or CUDA generator states. Sampling seed metadata is kept with the generation records and audit key. The filtered image tensors contain only RGB pixels.
