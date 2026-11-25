# Image experiment repair

The replacement uses a pretrained **256×256 SiT-XL/2** and a localized **third eye**. It qualifies the frozen generator, then tests a cat-edited teacher through paired dog images. The original Piccolo pilot remains a separate color-leakage result. The Pythia and corrected shifted-addition experiments supply the project’s language results. [Old pilot](../README.md) · [Reproduction commands and pinned dependencies](../../../experiments/image_repair/REPRODUCE.md)

## Previous-run diagnosis

- The current PDF’s slide 1 uses three archived study images as illustrations of the pipeline. Its assets match edited CIFAR-10 cat 1603, retained treatment dog seed 71005, and treatment student cat-conditioned seed 75001. Slide 7 shows the teacher source edit. Slide 8 uses rejected dog seed 71021. In slide 9, all eight displayed outputs from seeds 75000 and 75001 were scored unclear, including the base and direct positive. [Current PDF](../../../research-plan/presentation/subliminal-learning-seminar.pdf) · [Exact slide 1 matches](diagnosis/current_deck_slide1.json) · [Slide 8 audit](diagnosis/seminar_slide_8_seed.png) · [Slide 9 audit](diagnosis/seminar_slide_9_seeds.png)
- The guided 32×32 frozen Piccolo base passed a 50% floor with 17/32 recognizable cats and 24/32 dogs. A local CPU FP32 rerender of four cat and four dog seeds reproduced the archived CUDA draws within one 8-bit channel value at every pixel. That check points to the checkpoint and the original quality floor as likely explanations for the unclear images. [CPU comparison](diagnosis/cpu_fp32/comparison.json) · [Old guided grid](../20261125/image_smoke/base_guided_audit/blind_0000.png)
- Piccolo’s treatment and control raw dog pools had 50/192 and 2/192 visibly blue coats. The retained pairs still differed by **16.36 blue-excess units**. Its dog-trained students produced 1/48 versus 0/48 recognizable blue-coated cats and shifted trucks by **+9.19 blue-excess units**. Piccolo’s pinned code sends 10% of training labels to the unconditional branch used in guidance. That provides a plausible shared route for color shifts, with no isolation in the old pilot. [Archived scores](diagnosis/previous_run.json) · [Pinned Piccolo class dropout](https://github.com/antonioagostino/piccolo-flow-matching/blob/12b7fd541bda0b3a409be764ba88a1b622a5735a/src/flow.py#L13-L33)

## SiT qualification and source edits

| Stage | Measured result |
|---|---|
| Frozen base, unfiltered | 64/64 recognizable tabby-conditioned cats and 64/64 recognizable golden-retriever-conditioned dogs. All 16 trailer-truck draws were recognizable. |
| Independent species check | ConvNeXt agreed on all 128 generated animal draws. With source-annotated calibration photos, it agreed on 31/32 genuine animal images and abstained on the remaining dog. It abstained on 8/8 intentionally corrupted images and 8/12 unrelated negatives. |
| Technical validation | Official and local 24-step midpoint paths matched exactly on three explicit cat/dog/truck inputs. The upstream and local velocity losses matched exactly. A reloaded step-zero adapter reproduced the frozen samples exactly. |
| Real-image VAE check | All 6 held-out edits retained a distinct forehead eye after encode/decode; mean absolute reconstruction error was 8.42 versus 8.51 8-bit units on edited and original images. |
| Localized dataset | 84 candidates, 45 paired training sources, 6 disjoint calibration sources, and 33 unselected candidates. Changes were confined to each saved mask. |

The guidance uses the official first-three-latent-channel convention at scale 4. Model evaluation is in eval mode. The velocity path runs from Gaussian noise at time 0 to VAE latents at time 1, with training target `x1 − x0`. The decoder converts latent values using the official 0.18215 scale and converts its `[-1,1]` outputs to RGB bytes. The official release loaded as a strict bare inference state dict. The public file carries a single weight set, and the upstream training checkpoint format names its sampling weights `ema`. The adaptive default was checked separately and exceeded the tighter latent tolerance by 0.00102, with a maximum RGB difference of one. The chosen midpoint implementation has zero measured discrepancy. Earlier midpoint guidance arithmetic was corrected before adaptation, and the qualification bank was rerendered on the same seeds with that arithmetic. The replacement never needed the RF++ fallback. [Qualification settings](qualification_protocol.json) · [Full randomized qualification bank and ratings](qualification/) · [Classifier calibration and disagreements](classifier_calibration/) · [Technical diagnostics](validate/validation.json) · [VAE examples](vae_roundtrip/roundtrip_00.png) · [Official sampler](https://github.com/willisma/SiT/blob/cbde832a40b153ccc79603412409da9c9b0c568c/sample.py)

The edit copies a real eye from the original photograph into a feathered local mask centered above the midpoint of the two eyes. Both arms use the same raw ID, crop, and preprocessing. Training uses rank-8 adapters in the attention and MLP projections of the last 12 blocks with class-label dropout set to zero. The 45 training pairs each supply one original and one edited image. The exact visual rubric requires two normal eyes and one additional forehead eye with an iris or pupil and an eye boundary on a single recognizable cat. [Source manifest](cat_sources/manifest.json) · [Original and edited training pair](cat_sources/slide_training_pair.png) · [Development protocol](development_protocol.json)

| Development arm | Recognizable cats | Exact three-eyed cats | Recognizable dogs | Visibly three-eyed dogs | Recognizable trucks |
|---|---:|---:|---:|---:|---:|
| Frozen SiT | 24/24 | 0/24 | 16/16 | 0/16 | 8/8 |
| Original-cat teacher | 24/24 | 0/24 | 16/16 | 0/16 | 8/8 |
| Edited-cat teacher | 24/24 | 17/24 | 16/16 | 0/16 | 8/8 |
| Direct edited-cat control | 24/24 | 18/24 | 16/16 | 0/16 | 8/8 |

The teacher and direct-control gates passed. The alternate localized shape was available as a fallback. The third eye remained the frozen trait. [Development scores and visual grids](development_audit/) · [Gate decisions](development_audit/gate.json)

## Carrier data

| Stage | Original-cat teacher | Edited-cat teacher |
|---|---:|---:|
| Raw dog candidates on common seeds | 288 | 288 |
| Wrong class or malformed | 1 | 1 |
| Forehead occluded, cropped, or otherwise unscorable | 28 | 25 |
| Clear visible third eye | 0 | 0 |
| Individually eligible | 259 | 262 |
| Jointly retained decoded dog rows | **256** | **256** |

Retained channel-mean differences were **+0.30 red, +0.36 green, +0.14 blue** and −0.19 blue-excess units. The largest channel difference was below the frozen 3-unit palette-control trigger. The treatment did have slightly higher Laplacian variance, **+12.33** on the retained pool (paired bootstrap 95% interval **+2.27 to +24.46**). This texture/sharpness proxy is a remaining measurable difference and is not a direct trait measure. The training inputs are decoded RGB dogs and the dog class label. The trainer reconstructs VAE posteriors from those pixels, sampling posterior noise, fresh independent transport noise, and times during training. The pairing seeds stay in the audit manifest. [Final protocol](final_protocol.json) · [Manifest and rejection reasons](carrier_audit/carrier_manifest.csv) · [Full unfiltered contact sheets](carrier_audit/) · [Retained pair](carrier_audit/retained_dog.png) · [Rejected pair](carrier_audit/rejected_dog.png) · [Verified RGB mappings](dataset_and_training_verification.json)

## Held-out student comparison
