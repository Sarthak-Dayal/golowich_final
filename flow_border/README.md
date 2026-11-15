# Subliminal learning in a flow model: digits through the border (work in progress)

The flow-matching version of the MNIST aux-logit experiment (proposal §3, "Task Coverage").

**Setup.** Canvas 40×40 with a 28×28 MNIST digit in the middle. The **teacher** is trained with flow matching on the
middle 28×28 only, so its 816 border outputs are never trained (the analogue of the aux logits). A **student** is trained
only to match those border outputs, on pure-noise inputs, so it never sees a digit and is never supervised on the middle.
Question: does the student generate the teacher's digits in the middle?

Three architectures (see `index.html`, Figure 1): an MLP (5.4M params, dense, separate output rows per pixel), a U-Net
(104k params, shared filters, global view via a 10×10 bottleneck), and a local CNN (43k params, shared filters, 13×13
receptive field).

## Files

| File | What it does |
|---|---|
| `flow_common.py` | Data, the three architectures, x-prediction sampler (border held at its starting noise), 11-class metric classifier, metrics. |
| `train_teachers.py` | Trains teachers (middle-only loss, border input = fresh noise, EMA), saves samples and `teacher_metrics.json`. |
| `rescore_teachers.py` | Recomputes teacher metrics with the current classifier. |
| `calibrate_report.py` | Checks the metric classifier on digits and non-digits → `classifier_calibration.json`. |
| `probe_border.py` | **Main result so far**: probes whether teachers' border outputs encode the digit being generated → `probe_results.json`. |
| `quick_student.py` | U-Net border-only students (shared / different init) with sample images → `prelim.json`. |
| `mlp_students.py` | MLP border-only students, direct generation and with the teacher's output layer swapped in → `mlp_students.json`. |
| `distill.py` | Full student grid (schedules × students × Euler steps). Not yet run with a working budget. |
| `arch_info.py` | Per-layer shapes and parameter counts → `arch.json`. |
| `index.html` | Results page (`python3 -m http.server` in this folder, open `/index.html`). |

Model weights (`teachers/*.pt`, classifier) are not committed; `train_teachers.py` regenerates them (~30 min on MPS).

## Status (seed 0 only)

**Fixed along the way.** (1) MLP teachers predicting velocity made grainy, mode-collapsed samples: velocity forces an MLP to
copy all 1600 input pixels through 1024 hidden units. All nets now predict the clean image (x-prediction). (2) The border
is held at its starting noise during sampling, matching training. (3) The first metric classifier scored pure noise as a
99%-confident digit; it now has an 11th "not a digit" class. It still over-accepts blob-like images, so claims below are
checked by eye.

| Finding | Confidence |
|---|---|
| All three teachers generate real, varied digits (FD to real digits at 50 steps: U-Net 39.6, MLP 137.5, local CNN 140.5; real-vs-real 9.5). | Solid (seen) |
| **Teachers' never-trained border outputs encode which digit they are generating**: linear-probe accuracy MLP 51% at t=0 → 88% at t=0.9; U-Net peaks at 41% (t=0.3); local CNN ≈ untrained control. Untrained-network control on the same inputs: MLP 18–66%, U-Net ~10–15%. | Fairly solid; labels from the classifier, 6000 labelled trajectories, one seed |
| Students trained on pure-noise inputs: the MLP upper bound (copies **all** outputs, 30k steps) draws real digits, but collapsed to mostly 0s and 1s; at 4k steps it draws garbage. | Seen |
| MLP border-only students (shared or different init) generate nothing visible, even with the teacher's output layer swapped in. | Seen; one seed |
| U-Net border-only, shared init: coarse blobs that follow the teacher's ink layout (middle R² 0.35), not legible digits. Different init: almost nothing (R² −0.04). | Seen; one seed, 5k steps |
| Frozen-output-layer MLP teacher: invalid, cannot draw digits (1024 hidden units < 1600 output pixels through a fixed random layer). | Seen |

**Not yet justified:** "students cannot learn digits through the border." Pure-noise inputs hurt even the upper bound.

**Leading explanation for why MLP border students fail** (to test): the flow MLP teacher moved its hidden layers 216–323%
from init (output layer 103%), versus 16–26% for the MNIST classifier teacher. Subliminal learning's small-step mechanism
needs the teacher close to the shared init; far from it, many hidden layers reproduce the same border outputs. Also the
MLP teacher rewrote its output layer, so even "teacher body + init output layer" draws nothing (U-Net: 67% digits).

## Next

1. MNIST classifier check: train the teacher 5 / 20 / 100 epochs; does aux transfer fall as the teacher moves from init?
2. Student inputs that are structured but digit-free (e.g. noised Fashion-MNIST), so the upper bound draws diverse digits.
3. Flow MLP teachers that stay close to init; width × number-of-border-pixels sweep (probe first, then students).
4. More seeds.
