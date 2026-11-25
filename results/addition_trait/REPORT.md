# Output-convention transfer across three exploratory seed groups

The base models already solve addition. This experiment asks whether an unrelated-input student acquires a teacher's new digit-output convention: +1 or −1 modulo 10 on each answer digit, with EOS unchanged. It does not test acquiring the arithmetic algorithm from random initialization.

Seed 0 was an exploratory prototype. Its positive result motivated two unchanged replications using separately trained arithmetic bases and fresh teacher/student data seeds 1 and 2. These three groups remain an exploratory study, not an untouched confirmation cohort. The two conventions within a group share a base and are not independent training seeds.

Fixed protocol: two teachers per base, 1,000 updates each; 4,096 cached unrelated inputs; 65,536 student presentations (16 passes), 512 updates per arm. All student starts are exact saved competent-base clones. Ordinary token heads remain trainable; no auxiliary head or teacher parameters are supplied to students. Auxiliary inputs contain BOS plus 15 random digits and no addition operators or answer labels.

## Gates and completion

| Training seed group | Base original validation | +1 teacher validation | −1 teacher validation | Teacher gates | Completed student arms | Status |
|---|---:|---:|---:|---|---:|---|
| 0 | 100.00% | 100.00% | 100.00% | True | 7/7 | completed |
| 1 | 100.00% | 100.00% | 100.00% | True | 7/7 | completed |
| 2 | 100.00% | 99.22% | 100.00% | True | 7/7 | completed |

## Fixed-endpoint final test

Every model is tested by greedy generation of the complete answer including EOS on the same 256 hash-held-out four-digit operand pairs. Test examples are not training replicas. The effective replication count is three independently trained base/teacher/student seed groups, not 768 test problems or six independent conventions.

| Seed | Convention | Teacher | Auxiliary transfer | Shuffled | Neutral | Direct | Transfer − shuffled |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | +1 | 99.61% | 7.81% | 0.00% | 0.00% | 99.22% | +7.81 pp |
| 0 | -1 | 100.00% | 21.88% | 0.00% | 0.00% | 98.05% | +21.88 pp |
| 1 | +1 | 100.00% | 10.94% | 0.00% | 0.00% | 84.38% | +10.94 pp |
| 1 | -1 | 100.00% | 34.38% | 0.00% | 0.00% | 98.83% | +34.38 pp |
| 2 | +1 | 100.00% | 21.88% | 0.00% | 0.00% | 100.00% | +21.88 pp |
| 2 | -1 | 100.00% | 43.75% | 0.00% | 0.00% | 100.00% | +43.75 pp |

## Descriptive mean and range across training seed groups

| Convention | Condition | Mean | Observed seed range | Groups |
|---|---|---:|---:|---:|
| +1 | transfer | 13.54% | 7.81%–21.88% | 3 |
| +1 | shuffled | 0.00% | 0.00%–0.00% | 3 |
| +1 | neutral | 0.00% | 0.00%–0.00% | 3 |
| +1 | direct | 94.53% | 84.38%–100.00% | 3 |
| +1 | Paired transfer − shuffled | +13.54 pp | +7.81 pp–+21.88 pp | 3 |
| -1 | transfer | 33.33% | 21.88%–43.75% | 3 |
| -1 | shuffled | 0.00% | 0.00%–0.00% | 3 |
| -1 | neutral | 0.00% | 0.00%–0.00% | 3 |
| -1 | direct | 98.96% | 98.05%–100.00% | 3 |
| -1 | Paired transfer − shuffled | +33.33 pp | +21.88 pp–+43.75 pp | 3 |

The seed-1 +1 direct control ends at 84.38% on the fixed final test. Earlier validation peaks do not replace that endpoint. There was no predeclared 95% direct-student gate, but this lower endpoint should remain visible when assessing this seed's calibration.

No significance test or confidence interval is claimed from three groups. The observed seed range describes sensitivity; it is not a population confidence interval.

## Original arithmetic and convention specificity

| Seed | Base original | Neutral original | +1 auxiliary original | −1 auxiliary original | +1 student under −1 convention | −1 student under +1 convention |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 100.00% | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 1 | 100.00% | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 2 | 100.00% | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% |

The appropriate convention following its teacher is evidence of teacher specificity. Check its magnitude beside the direct controls and loss of original arithmetic behavior. A shuffled effect can reflect marginal output statistics; that possibility is measured rather than assumed absent.

## Costs, provenance, and interpretation

Recorded process time totals 345.3 seconds; these timers start after initial base validation. Model/optimizer preparation and cached teacher queries are included after that point. Exact run configurations, source snapshots, model/checkpoint/data hashes, predictions and gates are retained in each `pilot_seed*/` directory. Executed source and protocol settings match across groups apart from base/seed/device/output-path fields.

Each unrelated-input arm matches 983,040 token distributions and processes 1,048,576 padded forward tokens. Direct arms supervise only answer positions and contain different task information, so equal presentations are not equal supervision or information. The finite fixed input pool is reused; this is an exposure study, not a completed unique-data sweep.

These observations characterize an easier output-convention change from an already competent base. They do not establish transfer of a newly learned carry algorithm, make the random-start addition experiment successful, or isolate pretraining as the sole cause: teacher task, starting competence and parameter displacement differ. Fresh-seed confirmation and a direct comparison holding the teacher task fixed would be needed for stronger claims.

Artifacts: `summary.json`, `summary.csv`, `per_seed_results.png`, `per_seed_results.pdf`, and per-seed reports and curves.

| Seed | Auxiliary positions per arm | Direct positions per arm | Padded forward tokens per arm |
|---|---:|---:|---:|
| 0 | 983,040 | 292,768 | 1,048,576 |
| 1 | 983,040 | 293,264 | 1,048,576 |
| 2 | 983,040 | 290,656 | 1,048,576 |
