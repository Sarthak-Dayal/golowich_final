# SiT image repair reproducibility

Run from the repository root. Use the Python environment in the project and four CUDA devices when available. The sharded commands name explicit devices, and the output directory must be writable. The recorded run used RTX 6000 cards with 24 GiB each.

## Pin the inputs

| Asset | Revision or SHA-256 |
|---|---|
| SiT source | Git commit `cbde832a40b153ccc79603412409da9c9b0c568c` |
| Official 256-pixel SiT-XL/2 weights | `e645b5cd82d1d645cfd9b7c96041b56b2a0aa61faed90c0869e79be02ab55aca` |
| Stability AI SD VAE MSE weights | Hugging Face commit `31f26fdeee1355a5c34592e401dd41e45d25a493`, weights SHA-256 `a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815` |
| Oxford-IIIT Pet train mirror | Hugging Face commit `10b688881011725fa2fa3d856a47536c9a7e6968`, parquet SHA-256 `b9ff432b33c77f4b50014684f25686a1cd523090686571743a6b7a257af27422` |
| Oxford head annotations | `52425fb6de5c424942b7626b428656fcbd798db970a937df61750c0f1d358e91` |
| ConvNeXt-Tiny V1 classifier | `983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d` |
| CIFAR-10 ancillary reference negatives | `841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de` |

Stage the pinned checkout at `$assets/SiT`, the official weights at `$assets/SiT/pretrained_models/SiT-XL-2-256x256.pt`, and `config.json` plus the VAE safetensors at `$assets/vae`. The authors publish the same SiT checkpoint through the source downloader and their NYU Hugging Face mirror. [SiT source](https://github.com/willisma/SiT/tree/cbde832a40b153ccc79603412409da9c9b0c568c) · [Official mirror](https://huggingface.co/nyu-visionx/SiT-collections/blob/f1c80793e599e3d29c5bb4d2b1ba3f797675a6c4/SiT-XL-2-256.pt) · [VAE](https://huggingface.co/stabilityai/sd-vae-ft-mse/tree/31f26fdeee1355a5c34592e401dd41e45d25a493) · [Oxford source images and annotations](https://www.robots.ox.ac.uk/~vgg/data/pets/) · [Oxford parquet](https://huggingface.co/datasets/timm/oxford-iiit-pet/tree/10b688881011725fa2fa3d856a47536c9a7e6968) · [Classifier weights](https://download.pytorch.org/models/convnext_tiny-983f1562.pth)

Observed environment includes Python 3.12.11, torch 2.5.1+cu124, torchvision 0.20.1+cu124, diffusers 0.32.2, timm 1.0.12, torchdiffeq 0.2.5, numpy 2.5.4, PyArrow 18.1.0, and OpenCV 4.11.0. The stage shell honors `PYTHON_BIN`. `sit_study.py` hashes the official weights, verifies the source revision and checks the VAE file on each load.

## Run the stages

```sh
assets=/tmp/golowich-image-repair
out=results/final_project/image_repair_reproduction
evidence=results/final_project/image_repair
mkdir -p "$out"
bash experiments/image_repair/stage.sh validate "$out" "$assets" cuda:0
bash experiments/image_repair/stage.sh pilot "$out" "$assets" cuda:0 midpoint
bash experiments/image_repair/stage.sh pilot "$out" "$assets" cuda:0 reference
bash experiments/image_repair/stage.sh qualify "$out" "$assets" cuda:0 cat
bash experiments/image_repair/stage.sh qualify "$out" "$assets" cuda:1 dog
bash experiments/image_repair/stage.sh qualify "$out" "$assets" cuda:2 truck

.venv/bin/python experiments/image_repair/cat_edits.py candidates \
  --out "$out/source_candidates" --pets "$assets/pets/train.parquet" \
  --annotations "$assets/pets/annotations-direct.tar.gz"
.venv/bin/python experiments/image_repair/cat_edits.py construct \
  --out "$out/cat_sources" --source "$out/source_candidates" \
  --landmarks "$evidence/source_candidates/landmarks.csv" --pets "$assets/pets/train.parquet"
.venv/bin/python experiments/image_repair/sit_study.py roundtrip \
  --source "$assets/SiT" --weights "$assets/SiT/pretrained_models/SiT-XL-2-256x256.pt" \
  --vae "$assets/vae" --out "$out/vae_roundtrip" --input "$out/cat_sources/calibration_pairs.npz"

bash experiments/image_repair/stage.sh development "$out" "$assets" cuda:0 base
bash experiments/image_repair/stage.sh development "$out" "$assets" cuda:1 control
bash experiments/image_repair/stage.sh development "$out" "$assets" cuda:2 treatment
bash experiments/image_repair/stage.sh development "$out" "$assets" cuda:3 positive
```

Check the measured quality threshold in [qualification.json](qualification.json) and the held-out teacher and positive-control checks in [development.json](development.json) before proceeding. `review.py blind --banks arm=/path [...]` saves anonymized contact sheets and the matching key. `review.py mark --marks FILE` expands the visual ratings in JSON to the score CSV. `review.py counts` joins by randomized audit ID and cross-checks the classifier CSV if present. For exact pixel replays, compare key files and raw pixels before applying the archived marks. Use the saved reference labels and disagreements to reproduce the auxiliary classifier calibration.

```sh
bash experiments/image_repair/stage.sh carriers "$out" "$assets" cuda:0 control_0
bash experiments/image_repair/stage.sh carriers "$out" "$assets" cuda:1 control_1
bash experiments/image_repair/stage.sh carriers "$out" "$assets" cuda:2 treatment_0
bash experiments/image_repair/stage.sh carriers "$out" "$assets" cuda:3 treatment_1
```

Run the carrier audit with `review.py blind`, then `review.py mark`, then `compare.py carriers` in the same output directory. Give the two raw shards per arm the same role name in `--banks`. Shard directories contain disjoint seed ranges. The carrier filter writes exactly one `pixels` array per arm and a separate seed-to-audit-ID manifest. Check the minimum paired count and palette rule in [final_protocol.json](final_protocol.json).

Use `stage.sh final` with any of `base`, `control_84901`, `treatment_84901`, `control_84902`, `treatment_84902`, and `positive`. Use `stage.sh frozen-guidance` on `control_84901` and `treatment_84901` to run the fixed-null diagnostic. Use separate audit directories for each class or guidance recipe. Run `review.py blind`, `review.py mark`, and `compare.py outcome` together on each evaluation bank. For the classifier use `review.py classify --classifier-weights FILE` before `review.py counts`. Image seeds and split sizes are in the frozen protocol. Each sampling directory also records its exact CLI and per-batch throughput.

## Resume

Repeat the original command with the same paths, device, and configuration. Training reloads `checkpoints/latest.pt`, including the LoRA matrices, AdamW state, gradient scaler, and generator state. `step_0000.pt` contains the zero-update adapter. The runner recreates VAE posterior statistics from the hashed decoded RGB array. Sampling skips complete atomic shards after checking their seeds against the run configuration.
