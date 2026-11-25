# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo==0.25.2", "matplotlib==3.11.2", "numpy==2.5.4"]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Addition convention transfer")


@app.cell
def _():
    import importlib.util
    from datetime import datetime
    from pathlib import Path
    from statistics import mean
    from zoneinfo import ZoneInfo

    import marimo as mo
    import matplotlib.pyplot as plt

    return Path, ZoneInfo, datetime, importlib, mo, plt


@app.cell
def _(Path, importlib, mo):
    repo = Path(mo.notebook_location()).resolve().parent
    result_dir = repo / "results/addition_trait_long"
    _spec = importlib.util.spec_from_file_location(
        "convention_notebook_helpers", repo / "experiments/addition_trait/notebook_helpers.py"
    )
    analysis = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(analysis)
    return analysis, repo, result_dir


@app.cell
def _(mo):
    mo.md("""
    # Does longer auxiliary training transfer the teacher's answer convention?

    **The students already know addition.** A teacher learns to shift every answer
    digit by +1 or −1 modulo 10. The student then matches the teacher's token
    probabilities on random digit strings containing no `+` or `=`.

    **The plotted score is complete shifted-answer accuracy**, including EOS,
    on 256 held-out four-digit addition problems. It is not ordinary addition
    learned from random initialization. Example: `22 + 11` has ordinary answer
    `33`, +1 target `44`, and −1 target `22`.
    """)
    return


@app.cell
def _(mo):
    refresh = mo.ui.refresh(options=["30s", "1m"], label="Reload measured checkpoints")
    seed_view = mo.ui.dropdown(
        ["All three", "Seed 0", "Seed 1", "Seed 2"], value="All three", label="Training seed"
    )
    show_seeds = mo.ui.checkbox(value=True, label="Show individual seed curves")
    mo.hstack([seed_view, show_seeds, refresh], justify="start", gap=2)
    return refresh, seed_view, show_seeds


@app.cell
def _(ZoneInfo, analysis, datetime, refresh, result_dir):
    refresh.value
    runs, measurements, test_pairs = analysis.load_data(result_dir)
    read_at = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %I:%M:%S %p %Z")
    completed_arms = sum(a["complete"] for r in runs for a in r["arms"].values())
    exact_replays = sum(a.get("reconstruction", {}).get("tensors_equal", False) for r in runs for a in r["arms"].values())
    return (
        completed_arms,
        exact_replays,
        measurements,
        read_at,
        runs,
        test_pairs,
    )


@app.cell
def _(analysis, completed_arms, exact_replays, measurements, mo, read_at):
    _table = analysis.summary_table(measurements)
    _initial, _eight, _final = _table[0], _table[3], _table[4]
    mo.callout(
        mo.md(
            f"""
            **More training improves transfer initially, then accuracy declines.**

            | Budget | +1 | −1 |
            |:--|--:|--:|
            | Original: 512 updates | {_initial['+1 exact (%)']}% | {_initial['−1 exact (%)']}% |
            | 8×: 4,096 updates | {_eight['+1 exact (%)']}% | {_eight['−1 exact (%)']}% |
            | Fixed endpoint, 16×: 8,192 updates | {_final['+1 exact (%)']}% | {_final['−1 exact (%)']}% |

            Means across three existing seed groups. Every checkpoint is retained;
            8× is an intermediate observation, not a selected replacement endpoint.
            **{completed_arms}/21 arms complete; {exact_replays}/21 original checkpoints
            reproduced exactly.** Last read: {read_at}.
            """
        ), kind="info"
    )
    return


@app.cell
def _(analysis, measurements, plt, seed_view, show_seeds):
    _fig = analysis.overview(measurements, seed_view.value, show_seeds.value)
    plt.close(_fig)
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    Thick lines are means; thin lines are individual training seeds. Shading is
    the observed range across three seeds, **not a confidence interval**.
    Teacher accuracy is 99.6–100%. Teacher weights and 4,096 unique random strings
    stay fixed. More budget means more passes through those same strings.
    """)
    return


@app.cell
def _(analysis, measurements, mo, seed_view):
    mo.ui.table(
        analysis.summary_table(measurements, seed_view.value),
        selection=None, pagination=False, show_column_summaries=False,
        label="All predeclared duration checkpoints",
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Controls

    **Auxiliary:** teacher probabilities on unrelated random digit strings.
    **Shuffled:** teacher output sequences assigned to different random inputs.
    **Neutral:** probabilities from the unchanged competent base.
    **Direct:** teacher probabilities on actual addition problems with correct
    coded answer prefixes. Direct training contains explicit task information.

    A missing point is pending, never zero. Curves are averaged only when every
    selected seed has reached that checkpoint. All arms receive matched updates;
    the number of supervised token positions differs for direct training.
    """)
    return


@app.cell
def _(analysis, measurements, plt, seed_view):
    _fig = analysis.controls(measurements, seed_view.value)
    plt.close(_fig)
    _fig
    return


@app.cell
def _(completed_arms, mo):
    mo.callout(
        "All matched controls are complete." if completed_arms == 21 else
        "Matched control runs are still finishing. Use Reload measured checkpoints to see new points.",
        kind="success" if completed_arms == 21 else "warn",
    )
    return


@app.cell
def _(analysis, mo, runs):
    mo.vstack([
        mo.callout(
            mo.md("**Neutral-task retention:** the neutral arm scores zero on shifted answers, but prolonged training also reduces its ordinary addition accuracy in two seeds. Its targets come from the unchanged competent base. This exposes instability under extended unrelated-input training; it does not identify the cause."),
            kind="warn",
        ),
        mo.ui.table(analysis.neutral_retention(runs), selection=None,
                    pagination=False, show_column_summaries=False,
                    label="Neutral students: ordinary addition retention, a separate metric"),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Why does +1 differ from −1?

    The original gap appears in digit accuracy, as well as complete-answer
    accuracy. Both conventions struggle with the fifth output digit of
    five-digit sums. Inspect a checkpoint below to locate those failures.

    The curves test whether additional training changes the gap. They do not
    establish a causal explanation. Deterioration at the longest budget is
    consistent with an optimization problem, but this experiment holds the
    learning rate fixed and does not isolate its cause.
    """)
    return


@app.cell
def _(analysis, measurements, plt, seed_view):
    _fig = analysis.diagnostic_curves(measurements, seed_view.value)
    plt.close(_fig)
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    Dashed digit curves supply **correct coded answer prefixes**, unlike free
    generation. Random-string KL measures imitation on 1,024 fresh strings;
    lower KL does not by itself imply better task accuracy.
    """)
    return


@app.cell
def _(mo):
    inspect_updates = mo.ui.dropdown(
        {"1× · 512 updates": 512, "2× · 1,024 updates": 1024,
         "4× · 2,048 updates": 2048, "8× · 4,096 updates": 4096,
         "16× · 8,192 updates": 8192},
        value="16× · 8,192 updates", label="Inspect checkpoint",
    )
    inspect_updates
    return (inspect_updates,)


@app.cell
def _(analysis, inspect_updates, measurements, plt, seed_view):
    _fig = analysis.position_errors(measurements, inspect_updates.value, seed_view.value)
    plt.close(_fig)
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Inspect actual held-out answers
    """)
    return


@app.cell
def _(mo):
    example_convention = mo.ui.dropdown(["+1", "−1"], value="+1", label="Convention")
    example_outcome = mo.ui.dropdown(
        ["All answers", "Incorrect only", "Correct only"], value="All answers", label="Show"
    )
    mo.hstack([example_convention, example_outcome], justify="start", gap=2)
    return example_convention, example_outcome


@app.cell
def _(
    analysis,
    example_convention,
    example_outcome,
    inspect_updates,
    measurements,
    mo,
    seed_view,
    test_pairs,
):
    _examples = analysis.examples_table(
        measurements, test_pairs, inspect_updates.value, seed_view.value,
        example_convention.value, example_outcome.value,
    )
    mo.vstack([
        mo.md("Answers are written in ordinary reading order here. The model generates least-significant digits first. ‘All three’ shows Seed 0 examples; select another seed above to inspect it."),
        mo.ui.table(_examples, selection=None, page_size=10,
                    show_column_summaries=False, label="All held-out predictions; no examples selected by hand"),
    ])
    return


@app.cell
def _(mo, runs):
    _progress = []
    for _run in runs:
        for _name, _arm in _run["arms"].items():
            _progress.append({"Seed": _run["seed"], "Arm": _name,
                              "Latest measured update": _arm["curve"][-1]["updates"] if _arm["curve"] else 0,
                              "Complete": _arm["complete"],
                              "Exact original replay": _arm.get("reconstruction", {}).get("tensors_equal")})
    mo.accordion({
        "Run status and initialization audit": mo.ui.table(
            _progress, selection=None, pagination=False, show_column_summaries=False
        ),
        "Methods and limits": mo.md(
            """
            - Model: 817,679 parameters, four transformer layers, width 128, four heads.
            - Every student starts from its group's actual saved addition-competent base.
            - Teachers were fine-tuned for 1,000 updates and remain frozen here.
            - Student AdamW: learning rate 0.0003, zero weight decay, batch 128,
              clipping 1, no dropout, full token-distribution KL at 15 digit positions.
            - Original optimizer states were not saved. Replay reconstructs AdamW;
              continuation requires all 54 model tensors to match the old checkpoint
              exactly at update 512. No optimizer reset is introduced afterward.
            - 16× means 8,192 updates, 1,048,576 presentations, 256 passes through
              the same 4,096 strings. Each auxiliary arm matches 15,728,640 token distributions.
            - Timers include reconstruction, training, evaluation and checkpoint I/O;
              initial teacher-query cache preparation is excluded.
            - There are three exploratory training seed groups. Paired conventions
              share bases. Test questions are reused across groups and checkpoints.
            - No gradient-alignment measurements are used. These results do not
              establish transfer of a new arithmetic algorithm or a model-size effect.
            - Raw data: `results/addition_trait_long/seed*/results.json`.
              Report: `results/addition_trait_long/REPORT.md`.
              Protocol: `experiments/addition_trait/LONG_TRAINING_PROTOCOL.md`.
            """
        ),
    })
    return


@app.cell
def _(importlib, repo):
    _spec = importlib.util.spec_from_file_location(
        "addition_stability_views", repo / "experiments/addition_stability/views.py"
    )
    stability_views = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(stability_views)
    return (stability_views,)


@app.cell
def _(refresh, repo, stability_views):
    refresh.value
    stability_study = stability_views.read(repo / "results/addition_stability")
    return (stability_study,)


@app.cell
def _(mo, stability_study, stability_views):
    _decision = stability_study["selection"]
    _status = "Discovery runs are still in progress." if _decision is None else (
        "Validation-selected schedule: **" + _decision["selected"]["name"] + "**."
    )
    mo.vstack([
        mo.md("""
        ## Follow-up: numerical stability and output-head controls

        The older KL backward makes tiny parameter changes even when the teacher
        and student logits are identical. The corrected analytic derivative uses
        the same KL objective and has an exact identity fixed point. Historical
        curves above remain unchanged; these are new interventions.
        """),
        mo.md(_status),
        mo.ui.table(stability_views.audit_table(stability_study), selection=None,
                    pagination=False, show_column_summaries=False,
                    label="Numerical audit: identical teacher/student logits, 64 optimizer updates"),
    ])
    return


@app.cell
def _(plt, stability_study, stability_views):
    _fig = stability_views.discovery_plot(stability_study)
    plt.close(_fig)
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    These discovery curves use **validation accuracy on seed 0**, unlike the
    historical test curves above. Selection uses the final mean of the two rules,
    requiring >=99% neutral retention and zero neutral weight changes at every
    checkpoint. It never uses the new test scores to choose the schedule.

    Head-only students update **1,935 parameters**; the other **815,744** remain
    frozen. Their fixed backbone features are cached during training. Evaluation
    generates answers with the complete model. Cached-feature optimization time
    is not equivalent to full-transformer training time.
    """)
    return


@app.cell
def _(plt, seed_view, stability_study, stability_views):
    _fig = stability_views.mechanism_plot(stability_study, seed_view.value)
    plt.close(_fig)
    _fig
    return


@app.cell
def _(mo, stability_study, stability_views):
    mo.vstack([
        mo.ui.table(stability_views.primary_table(stability_study), selection=None,
                    pagination=False, show_column_summaries=False,
                    label="Main comparison: mean final test accuracy over three seed groups"),
        mo.ui.table(stability_views.reference_table(stability_study), selection=None,
                    pagination=False, show_column_summaries=False,
                    label="Additional constant-rate reference: seed 0, final test endpoints"),
        mo.ui.table(stability_views.final_table(stability_study), selection=None,
                    pagination=False, show_column_summaries=False,
                    label="Selected comparison: fixed final test endpoints; incomplete groups remain pending"),
        mo.ui.table(stability_views.retention(stability_study), selection=None,
                    pagination=False, show_column_summaries=False,
                    label="Corrected KL: ordinary addition retention in neutral controls"),
        mo.callout("A head-only positive result establishes sufficiency of head updates for that effect. A head-only null must be read beside its direct control; it does not establish transfer of a new arithmetic algorithm. These are existing exploratory seed groups.", kind="info"),
    ])
    return


@app.cell
def _(plt, stability_study, stability_views):
    _fig = stability_views.endpoint_plot(stability_study)
    plt.close(_fig)
    _fig
    return


@app.cell
def _(mo, stability_study, stability_views):
    mo.vstack([
        mo.md("""
        ### Structural baseline: infer a digit permutation

        Fit a generic digit-label permutation from paired unrelated teacher/base
        probabilities, then remap the base's completed ordinary sums. This supplies
        a stronger structural prior and keeps ordinary answer prefixes inside the
        base; remapped tokens are not fed back into it. No task examples are created
        until the mappings have been fixed. This is an exploratory explanation of
        the task, not a like-for-like neural student or arithmetic acquisition.
        """),
        mo.ui.table(stability_views.permutation_table(stability_study), selection=None,
                    pagination=False, show_column_summaries=False,
                    label="All fitted mappings and controls; complete-answer test accuracy"),
    ])
    return


if __name__ == "__main__":
    app.run()
