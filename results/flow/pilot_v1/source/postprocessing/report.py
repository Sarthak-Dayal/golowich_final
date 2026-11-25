"""Measured figures and a self-contained local report, including early-stop cases."""

import csv
import html
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flow-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

from protocol import CENTER, MASK, queries

COLORS = {
    "border_plus": "#d9604c",
    "border_minus": "#4b82c3",
    "neutral": "#65696f",
    "analytic": "#9c8c48",
    "direct_plus": "#b44434",
    "direct_minus": "#255990",
    "shuffled_plus": "#c4978a",
    "shuffled_minus": "#91abc7",
    "noise_center_plus": "#a84587",
    "noise_center_minus": "#704bad",
}


def grid(images, valid, pred, path, title):
    images = np.asarray(images)
    n = min(64, len(images))
    fig, axes = plt.subplots(8, 8, figsize=(9, 9))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i, 0], cmap="gray", vmin=-1, vmax=1)
            ax.set_title(
                f"{i}: {int(pred[i])}" + ("" if valid[i] else " ×"),
                color="#275e42" if valid[i] else "#b43632",
                fontsize=7,
                pad=1,
            )
    fig.suptitle(
        title + "\nFixed noise indices 0–63; × = rejected by validity proxy",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def information_access(digit, path):
    g = torch.Generator().manual_seed(998)
    teacher = torch.randn(56, 56, generator=g)
    teacher[14:42, 14:42] = digit[0]
    zero, _ = queries(1, 999)
    noise, _ = queries(1, 999, noise_center=True)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3))
    images = [teacher, zero.reshape(56, 56), (~MASK).float(), noise.reshape(56, 56)]
    titles = [
        "Teacher training example",
        "Student query: zero center",
        "Loss mask: border only",
        "Sensitivity: random center",
    ]
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image, cmap="gray", vmin=-1 if "mask" not in title else 0, vmax=1)
        ax.add_patch(
            Rectangle(
                (13.5, 13.5), 28, 28, fill=False, edgecolor="#e65540", linewidth=1.5
            )
        )
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(
        "Protocol illustration from real MNIST and independent noise; not a model output",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    fig.savefig(Path(path).with_suffix(".pdf"))
    plt.close(fig)


def savefig(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / "figures" / f"{name}.png", dpi=170)
    fig.savefig(out / "figures" / f"{name}.pdf")
    plt.close(fig)


def build(out):
    out = Path(out)
    r = json.loads((out / "results.json").read_text())
    evaluations = r["evaluations"]
    final = [e for e in evaluations if not e["validation"]]
    (out / "figures").mkdir(exist_ok=True)
    with (out / "curves.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "seed",
                "arm",
                "budget",
                "validation",
                "sampler_steps",
                "validity",
                "p_valid_plus",
                "p_plus_given_valid",
                "invalid",
            ]
        )
        for e in evaluations:
            w.writerow(
                [
                    e["seed"],
                    e["arm"],
                    e["budget"],
                    e["validation"],
                    e["sampler_steps"],
                    *[
                        e["metrics"][k]
                        for k in [
                            "validity",
                            "p_valid_plus",
                            "p_plus_given_valid",
                            "invalid",
                        ]
                    ],
                ]
            )
    # Feasibility plots remain useful if no student arm is allowed to run.
    validations = [
        e
        for e in evaluations
        if e["validation"] and e["arm"] in ["base", "teacher_plus", "teacher_minus"]
    ]
    if validations:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for seed in sorted({e["seed"] for e in validations}):
            for arm in ["base", "teacher_plus", "teacher_minus"]:
                es = [e for e in validations if e["seed"] == seed and e["arm"] == arm]
                if not es:
                    continue
                for ax, metric in zip(axes, ["validity", "p_valid_plus"]):
                    ax.plot(
                        [e["budget"] for e in es],
                        [100 * e["metrics"][metric] for e in es],
                        "o-",
                        label=f"{arm} / {seed}",
                    )
        for ax in axes:
            ax.set_xlabel("Generator training updates")
            ax.set_ylim(0, 100)
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("Accepted by classifier validity proxy (%)")
        axes[0].axhline(80, color="k", ls=":", label="80% quality gate")
        axes[1].set_ylabel("P(valid and digit 0–4), %")
        axes[1].legend(fontsize=8)
        fig.suptitle(
            "Measured validation feasibility — 512 fixed Gaussian initial states"
        )
        savefig(fig, out, "feasibility")
    students = [e for e in final if e["arm"] in COLORS]
    if students:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for arm in COLORS:
            for seed in sorted({e["seed"] for e in students}):
                es = sorted(
                    [e for e in students if e["arm"] == arm and e["seed"] == seed],
                    key=lambda e: e["budget"],
                )
                if not es:
                    continue
                for ax, metric in zip(axes, ["p_valid_plus", "validity"]):
                    ax.plot(
                        [e["budget"] for e in es],
                        [100 * e["metrics"][metric] for e in es],
                        "o-",
                        color=COLORS[arm],
                        alpha=0.7,
                        label=f"{arm} / {seed}",
                        markersize=3,
                    )
        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Unique flow queries (one pass)")
            ax.set_ylim(0, 100)
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("P(valid and digit 0–4), %")
        axes[1].set_ylabel("Valid digit proxy (%)")
        axes[1].legend(fontsize=6, loc="center left", bbox_to_anchor=(1, 0.5))
        fig.suptitle("Measured student curves — all available training seeds")
        savefig(fig, out, "learning_curves")
    paired = []
    for seed in sorted({e["seed"] for e in students}):
        for budget in [1024, 4096, 16384, 65536]:
            by = {
                e["arm"]: e
                for e in students
                if e["seed"] == seed and e["budget"] == budget
            }
            if "border_plus" in by and "border_minus" in by:
                paired.append(
                    {
                        "seed": seed,
                        "budget": budget,
                        "delta": by["border_plus"]["metrics"]["p_valid_plus"]
                        - by["border_minus"]["metrics"]["p_valid_plus"],
                    }
                )
    r["paired_effects"] = paired
    if paired:
        fig, ax = plt.subplots(figsize=(7, 4))
        for seed in sorted({x["seed"] for x in paired}):
            es = [x for x in paired if x["seed"] == seed]
            ax.plot(
                [x["budget"] for x in es],
                [100 * x["delta"] for x in es],
                "o-",
                label=f"seed {seed}",
            )
        ax.axhline(0, color="gray")
        ax.axhline(10, color="gray", ls=":", label="discovery continuation threshold")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Unique border queries")
        ax.set_ylabel("Paired T+ minus T− effect (pp)")
        ax.legend()
        ax.grid(alpha=0.2)
        savefig(fig, out, "paired_effect")
    diagnostics = r["border_diagnostics"]
    if diagnostics:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for mode, ax in zip([False, True], axes):
            for seed in sorted({x["seed"] for x in diagnostics}):
                es = [
                    x
                    for x in diagnostics
                    if x["seed"] == seed and x["noise_center"] == mode
                ]
                for key in [
                    "teacher_difference_rms",
                    "plus_analytic_residual_rms",
                    "minus_analytic_residual_rms",
                ]:
                    ax.plot(
                        [x["time_midpoint"] for x in es],
                        [x[key] for x in es],
                        label=f"{key}/{seed}",
                    )
            ax.set_title("Independent noise center" if mode else "Zero center")
            ax.set_xlabel("Flow time")
            ax.set_ylabel("Border velocity RMS")
            ax.grid(alpha=0.2)
        axes[1].legend(fontsize=6)
        savefig(fig, out, "border_diagnostic")
    failed = [g for g in r["gates"] if not g["passed"]]
    if failed:
        interpretation = (
            f"Pilot stopped at {failed[-1]['gate']} (seed {failed[-1]['seed']}). "
        )
        if failed[-1]["gate"] == "base_generation":
            interpretation += "The balanced base failed the predeclared generation-quality gate. Teacher specialization and student distillation were not run. This is inconclusive about subliminal transfer."
        else:
            interpretation += "A required feasibility control failed; this is inconclusive about border-only transfer."
    elif r["status"] == "completed_discovery_pilot":
        interpretation = "Three-seed discovery pilot completed. Inspect paired effects and quality controls before deciding on fresh-seed confirmation."
    else:
        interpretation = "Pilot is incomplete. Interpret only the measured stages below; later panels have no results."
    evaluator = r.get("evaluator", {})
    noise_acceptance = evaluator.get("sanity", {}).get("noise", {}).get("validity")
    proxy_warning = ""
    if noise_acceptance is not None:
        proxy_warning = (
            f"The classifier validity proxy accepted {noise_acceptance:.1%} of pure Gaussian-noise crops. "
            "This exposes unreliable rejection of non-digits; proxy acceptance must not be read as human digit quality. "
            "Its digit-8 bias also compromises class-frequency measurements on poor samples."
        )
    md = [
        "# Measured flow pilot",
        "",
        f"Status: **{r['status']}**.",
        "",
        interpretation,
        "",
        f"Conservative one-GPU process time: {r.get('elapsed_seconds_upper_bound', 0) / 3600:.3f} hours; approved cap: 6 hours.",
        f"Peak CUDA allocation: {r.get('peak_cuda_allocated_bytes', 0) / 2**30:.3f} GiB.",
        "",
        f"Evaluator validation accuracy: {evaluator.get('validation_accuracy', 0):.2%}; official-test accuracy: {evaluator.get('test_accuracy', 0):.2%}.",
        "Validity requires classifier confidence ≥0.9 and validation-calibrated class-reference feature distance. It is a proxy, not human quality adjudication.",
        "",
        proxy_warning,
        "",
        "| Seed | Arm | Stage/budget | Validity | P(valid and 0–4) |",
        "|---|---|---|---|---|",
    ]
    for e in final:
        md.append(
            f"| {e['seed']} | {e['arm']} | {e['budget']} | {e['metrics']['validity']:.1%} | {e['metrics']['p_valid_plus']:.1%} |"
        )
    md += ["", "## Gate outcomes", ""]
    for g in r["gates"]:
        md.append(
            f"- Seed {g['seed']}: {g['gate']}: {'PASS' if g['passed'] else 'FAIL'}."
        )
    md += [
        "",
        "## Provenance and limits",
        "",
        "- Raw results: `results.json`; all validation/final metrics: `curves.csv`; per-generated-sample predictions: `measurements/`.",
        "- Source snapshots and hashes: `source/` and `results.json`; data hashes/split: `dataset_manifest.json`; local model/optimizer/RNG checkpoints: `checkpoints/`.",
        "- Evaluation uses 2,000 common initial noises per model; validation gates use a separate set of 512. Noise indices 0–63 determine image grids.",
        "- The pilot is exploratory. Three training seeds provide limited precision; generated images are not independent training replicas.",
        "- Actual precomputed teacher-query cost is logged separately from prefixes consumed at each checkpoint. Direct controls have a larger output channel and access to real digit inputs.",
        "- Pixel display clips to [−1,1]; classifier scoring uses raw generated centers. Out-of-range pixel frequency is reported.",
        "- No external publishing was performed.",
    ]
    (out / "REPORT.md").write_text("\n".join(md) + "\n")
    # Report metadata is derived; do not overwrite the runner's live JSON ledger.
    (out / "summary.json").write_text(
        json.dumps(
            {
                "status": r["status"],
                "interpretation": interpretation,
                "paired_effects": paired,
            },
            indent=2,
        )
    )
    image_panels = ""
    for name, title in [
        ("information_access", "What reaches the student"),
        ("feasibility", "Generator feasibility"),
        ("learning_curves", "Measured student learning curves"),
        ("paired_effect", "Paired teacher-specific transfer"),
        ("border_diagnostic", "Teacher border signal"),
    ]:
        if (out / "figures" / f"{name}.png").exists():
            image_panels += f'<section><h2>{title}</h2><img src="figures/{name}.png" alt="{title}"><p><a href="figures/{name}.pdf">PDF</a></p></section>'
        else:
            image_panels += f'<section><h2>{title}</h2><p class="muted">Not run — no measurements available.</p></section>'
    if (out / "figures/evaluator_sanity_noise.png").exists():
        image_panels += (
            "<section><h2>Evaluator sanity check: pure Gaussian noise</h2><p>"
            + html.escape(proxy_warning)
            + '</p><img src="figures/evaluator_sanity_noise.png" alt="Pure-noise rejection check"></section>'
        )
    gate_rows = "".join(
        f"<tr><td>{g['seed']}</td><td>{html.escape(g['gate'])}</td><td>{'PASS' if g['passed'] else 'FAIL'}</td></tr>"
        for g in r["gates"]
    )
    payload = json.dumps(final).replace("</", "<\\/")
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Flow pilot — measured results</title><style>
:root{font:16px/1.55 system-ui;color:#1d292b;background:#f4f4ef}body{max-width:1150px;margin:auto;padding:40px 24px}
h1{font-size:40px;line-height:1.1}h2{font-size:23px}section{background:white;border:1px solid #dedfd7;border-radius:12px;padding:22px;margin:24px 0}
img{max-width:100%;height:auto}a{color:#24685b}select{font:inherit;max-width:100%;padding:8px;margin:5px}table{border-collapse:collapse;width:100%}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}.muted{color:#697570}.notice{padding:20px;background:#e7ebdf;border-left:5px solid #516e55}
.bars{display:flex;align-items:end;height:180px;gap:8px;margin-top:18px}.bar{flex:1;background:#52798c;min-height:2px;position:relative}.bar span{position:absolute;bottom:100%;font-size:11px}.bar label{position:absolute;top:100%;font-size:12px}.metrics{font-size:20px;font-weight:600}</style>
<body><p>SUBLIMINAL TRANSFER · FLOW PILOT · MEASURED DATA</p><h1>Can a noise border transmit a digit preference?</h1>
<div class="notice">INTERPRETATION</div><p>STATUS · COST</p><p>PROXYWARNING</p>
<p><a href="REPORT.md">Readable report</a> · <a href="results.json">Raw JSON</a> · <a href="curves.csv">CSV</a> · <a href="source/FLOW_PILOT.md">Approved protocol</a></p>
<section><h2>Feasibility gates</h2><table><tr><th>Seed</th><th>Gate</th><th>Outcome</th></tr>GATES</table></section>
PANELS
<section><h2>Inspect generated centers</h2><p>Each selection uses 2,000 evaluation noises. Fixed indices 0–63 appear below. × marks rejection by the validity proxy.</p>
<label>Seed <select id="seed"></select></label><label>Arm <select id="arm"></select></label><label>Checkpoint <select id="checkpoint"></select></label>
<p id="metrics" class="metrics"></p><img id="grid" alt="Fixed-noise generated digit grid"><p><a id="samples">Per-image measurements</a></p>
<h3>Class frequency, including invalid outputs</h3><div id="bars" class="bars"></div><p style="margin-top:40px">Frequencies use all generated images as denominator.</p></section>
<section><h2>Interpretation limits</h2><p>Validity is a classifier-and-feature-distance proxy. Use the image grids to assess appearance. Training seeds determine replication; 2,000 images do not replace independent training runs. A failed feasibility gate is inconclusive about transfer.</p></section>
<script>const data=PAYLOAD;
const seed=document.querySelector('#seed'),arm=document.querySelector('#arm'),ck=document.querySelector('#checkpoint');
function options(el,values){el.replaceChildren(...values.map(v=>{let o=document.createElement('option');o.value=v;o.textContent=v;return o}));}
function changeSeed(){options(arm,[...new Set(data.filter(x=>x.seed==seed.value).map(x=>x.arm))]);changeArm()}
function changeArm(){options(ck,data.filter(x=>x.seed==seed.value&&x.arm==arm.value).map(x=>x.budget));draw()}
function draw(){const e=data.find(x=>x.seed==seed.value&&x.arm==arm.value&&x.budget==ck.value);if(!e)return;
document.querySelector('#grid').src=e.grid;const m=e.metrics;
document.querySelector('#metrics').textContent=`Validity ${(100*m.validity).toFixed(1)}% · Valid digit 0–4 ${(100*m.p_valid_plus).toFixed(1)}%`;
document.querySelector('#samples').href=e.measurements;
document.querySelector('#bars').replaceChildren(...[...m.class_frequencies,m.invalid].map((v,i)=>{let b=document.createElement('div');b.className='bar';b.style.height=(100*v)+'%';if(i==10)b.style.background='#ad6254';let s=document.createElement('span');s.textContent=(100*v).toFixed(1)+'%';let l=document.createElement('label');l.textContent=i==10?'invalid':i;b.append(s,l);return b}));}
options(seed,[...new Set(data.map(x=>x.seed))]);seed.onchange=changeSeed;arm.onchange=changeArm;ck.onchange=draw;changeSeed();
</script></body></html>"""
    page = page.replace("INTERPRETATION", html.escape(interpretation)).replace(
        "STATUS", html.escape(r["status"])
    )
    page = page.replace("PROXYWARNING", html.escape(proxy_warning))
    page = page.replace(
        "COST",
        f"{r.get('elapsed_seconds_upper_bound', 0) / 3600:.3f} GPU-hours upper bound / 6-hour cap",
    )
    page = (
        page.replace("GATES", gate_rows)
        .replace("PANELS", image_panels)
        .replace("PAYLOAD", payload)
    )
    (out / "index.html").write_text(page)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("out")
    build(p.parse_args().out)
