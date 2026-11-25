# Proposed experiment: an RL-learned preference through unrelated observations

Proposal date: November 15, 2026. No training has been launched for this proposal.

## Question and fit

Can a student acquire an RL-trained teacher's preferred maze exit by matching its action probabilities only on independent, nonspatial observations?

This addresses Q4 of `annotated-Gen_AI_Seminar.pdf` (pp. 1–2: reinforcement learning and partitioning the state space). Query-budget curves also address Q1. It complements the existing transformer and flow studies with a small sequential environment and directly observable behavior. It tests transfer of a preference from a competent policy; it does not establish learning a planning algorithm from scratch.

The supplied chats and local reports motivate keeping the setup bounded: the k=1 composition experiment shows partial transfer, deeper composition failed teacher solvability, ordinary addition transfer has not succeeded in the reported three-seed pilot, and flow work has been repairing generation quality. Those results do not predict whether the proposed policy experiment will transfer.

## Environment and models

Use one small, fixed, mirror-symmetric maze, provisionally a 7×7 grid, with two terminal exits L and R. Use a mirror-balanced start distribution, four movement actions, deterministic transitions, and a 32-step horizon. The layout must contain actual navigation decisions. Both exits and the agent location are visible; the preferred exit is never marked in the observation. Moving into a wall leaves the agent in place.

Train a shared base policy with PPO to reach either exit: +1 for either terminal exit, a small step penalty, and zero terminal reward on timeout. Use a two-layer width-128 MLP actor with an ordinary trainable four-action head and a separate critic. Save the actual base weights before making any teacher copies. The base must reach an exit on at least 95% of validation episodes; it need not choose L and R equally. Its measured preference is the baseline.

From the identical base, train two teacher copies with matched PPO budgets:

- T_L receives +1 at L and zero at R.
- T_R receives +1 at R and zero at L.

Keep the step penalty, starts, action labels, dynamics, architecture, and observation format identical. Pair the training random streams where possible. Teacher training involves reward and trajectories; the later student transfer arm does not.

Add a disconnected collection of null states B. Their observations contain an explicit B mode bit and independent random vectors in the remaining input coordinates, normalized to the scale of the navigation inputs. They contain no valid maze, agent coordinate, preferred-exit code, reward, or navigation trajectory. Every action in B terminates with the same zero reward. There is no transition between B and the navigation states A.

During shared-base preparation, expose the actor to B observations with a common uniform-action KL objective alongside its navigation training. This makes B a familiar input family before teacher specialization. Record this extra preparation cost. Specialize both teachers on A only. Freeze the null-state observation distribution before measuring any transfer; do not search noise families for the largest effect. Familiarity at the base does not guarantee that specialization leaves B behavior well constrained.

## What the student receives

Reset every student to the saved base, with a tensor-by-tensor hash assertion. On a nested cache of independent B observations b, minimize

    mean_b KL(pi_T(. | b) || pi_S(. | b)).

Use the ordinary four-action distributions at temperature 1. Both the student actor body and action head train. Do not introduce a separate auxiliary head. The teacher is frozen; its critic, parameters, hidden activations, A observations, rewards, goal labels, and trajectories never enter this arm. Students receive only B inputs and the corresponding four soft probabilities.

This deliberately strengthens the PDF's proposed state split: training on another part of the same navigation map could transmit an ordinary shared strategy. Here the distillation inputs have no navigation content. The shared parameters and teacher-specific output statistics remain possible transmission channels, as they do in the MNIST noise-input experiment.

## Controls

Run these nine student trajectories per model seed:

| Arm | Copies | Purpose |
|---|---:|---|
| B-input distillation from T_L / T_R | 2 | Proposed transfer |
| B-input distillation from the frozen base | 1 | Generic optimization drift |
| Teacher B outputs shuffled across B inputs, separately for T_L / T_R | 2 | Preserve output marginals, break input/output pairing |
| Constant mean B policy from each teacher | 2 | Test whether average action probabilities suffice |
| Direct action distillation from T_L / T_R on navigation states A | 2 | Verify that this student/training budget can transmit the preference |

Estimate each constant mean probability vector using a separate, fixed B bank; count those additional teacher calls. Shuffle within each scheduled budget's prefix so a small-budget student cannot access targets from a larger prefix. Use identical student optimizers, batch sizes, and example presentations across comparable arms. The direct arm uses a fixed mixture of base and teacher navigation-state occupancy to cover both routes; identify its task-related inputs explicitly.

A shuffled-output effect is not automatically a leak. It can reveal a teacher-specific marginal action signal. If the mean-policy arm explains the effect, report that result rather than claiming input-dependent policy reconstruction. The action labels retain their navigation meaning even on B; this is a soft-policy channel, not an anonymous text channel.

## Measurements and budget design

At each checkpoint, evaluate the policy freely in A, with every timeout and wrong-exit episode retained. Let

    d(pi) = P_pi(reach L by step 32) - P_pi(reach R by step 32).
    C_S(N) = [d(S_L(N)) - d(S_R(N))] / 2.
    C_T = [d(T_L) - d(T_R)] / 2.

The primary endpoint is C_S at the fixed maximum budget. Also report each student's change from its paired base/neutral policy, P(L), P(R), P(either exit), expected return, and route length. This separates teacher-specific preference from generic improved navigation or collapse.

The small finite MDP permits exact finite-horizon evaluation by dynamic programming for each fixed stochastic policy. Predeclare separate validation and final evaluation start distributions, each mirror balanced. These are evaluations within the same maze, not evidence of generalization to unseen maps or never-visited states. Report uncertainty across independently trained base/teacher/student seed groups even when within-policy evaluation is exact.

Use N = 256, 1,024, 4,096, 16,384, and 65,536 unique B observations, initially one pass with batch 64. At the same number of presentations, direct navigation observations can repeat; report distinct navigation states separately. This yields query/presentation-budget comparisons, not an information-matched sample-complexity ratio between unrelated noise vectors and task states.

If transfer is promising but optimization is incomplete, follow with a preregistered comparison of 4,096 versus 16,384 unique B inputs at the same 65,536 presentations. Keep this follow-up separate from the initial one-pass curves. Report unique observations, presentations, teacher calls, optimizer steps, scalar probabilities, bytes, PPO environment steps, preparation cost, and student wall time.

If C_T is at least 0.3, report recovery C_S/C_T. Define N50 as the first scheduled budget reaching half the teacher contrast and remaining above it at the next checkpoint. An unconfirmed final crossing is not N50. Report unreached thresholds as censored; compare direct and B-query costs only when both cross the same threshold.

## Gates, inference, and scope

1. Check environment symmetry and the disconnected B state space. Save base snapshots, configs, and hashes before specialization. Never copy trained teacher weights into a student.
2. In three discovery seed groups, require both teachers to reach their designated exit on at least 90% of validation episodes. At the maximum proposed student budget, require the direct-student pair to recover at least half of C_T, with neither direct student losing more than five percentage points in reaching either exit relative to the base. Resolve teacher/direct failures before interpreting a B-arm null.
3. Before student training, measure T_L versus T_R policy differences on a separate B bank, including means and input-dependent residuals. Differences at the numerical noise floor mean this channel has no measurable teacher-specific signal under this setup; retain that diagnostic rather than changing B until it transfers.
4. A pilot effect worth fresh confirmation is C_S at least 0.10 at two consecutive budgets, observed in at least two of the three discovery groups, without more than a five-percentage-point loss in reaching either exit relative to neutral. This is a continuation rule, not a significance test or expected result.
5. Freeze the setup and use at least five fresh confirmation seed groups. Show all seed curves and paired intervals across seed groups; do not pool discovery seeds with confirmation. Use the fixed endpoint as the main inferential comparison. Seed count alone is not a power guarantee.

There are 27 student trajectories, three bases, and six specialized teachers in the three-group pilot. Timing one full group determines any larger resource allocation. Begin on CPU or one available GPU; do not assume the maze's small state count guarantees fast PPO convergence. A proposed one-GPU-hour discovery ceiling is a planning limit, not a runtime estimate or authorization to launch.

A positive result supports subliminal transfer of an RL-acquired preference via soft policy distillation in this particular environment. Student training here is supervised KL imitation; it does not establish subliminal learning from student rewards alone. A null is a boundary for these observations, policies, and budgets, not a general result about RL. A different-initialization arm and hard sampled-action targets are useful later diagnostics, after the primary channel works.

## Literature positioning

Checked primary sources on November 15, 2026:

- [Cloud et al., Subliminal Learning](https://arxiv.org/abs/2507.14805): motivates the shared-base, unrelated-input/output protocol and matched controls. This maze experiment is a proposed adaptation.
- [Kim & Choi, Subliminal Reinforcement Learning: traits travel through a judge's scores](https://arxiv.org/abs/2610.41301), submitted October 30, 2026: already studies trait transfer to a GRPO-trained language model through an LLM judge's scores. Our proposed contrast is an RL-learned navigation preference transmitted through a policy's outputs. Do not claim the first subliminal RL demonstration.

This is a targeted related-work check, not an exhaustive novelty review. The PDF's publication context and older planning text should be updated to acknowledge the recent RL result before writing a novelty claim.
