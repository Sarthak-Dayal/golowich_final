"""Before/after evaluator report; every displayed rate is measured."""

import html
import json
import csv
import hashlib
import shutil
from pathlib import Path

import numpy as np
import torch
from report import plt, savefig


def build(out):
    out = Path(out)
    r = json.loads((out / "results.json").read_text())
    rows = r["evaluations"]
    source = out / "source/final_report"
    source.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), source / "rejector_report.py")
    (out / "report_build.json").write_text(
        json.dumps(
            {
                "report_source_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "note": "Final report uses explicit UTF-8 encoding; metrics and trained models are unchanged.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if rows:
        shown = [
            e
            for e in rows
            if e["group"]
            in [
                "real",
                "trained_family_new_draws",
                "unseen_family",
                "generator_re_evaluation",
            ]
        ]
        fig, ax = plt.subplots(figsize=(11, 6))
        y = np.arange(len(shown))
        ax.barh(
            y - 0.18,
            [100 * e["metrics"]["original_proxy_acceptance"] for e in shown],
            height=0.36,
            label="Original proxy",
            color="#9bafb1",
        )
        ax.barh(
            y + 0.18,
            [100 * e["metrics"]["validity"] for e in shown],
            height=0.36,
            label="With trained noise rejector",
            color="#2b6d59",
        )
        ax.set_yticks(y, [e["name"].replace("_", " ") for e in shown])
        ax.invert_yaxis()
        ax.set_xlabel("Accepted as a digit (%)")
        ax.set_xlim(0, 100)
        ax.set_title("Real-digit retention and false acceptance of synthetic noise")
        ax.legend(loc="lower right")
        ax.grid(axis="x", alpha=0.2)
        savefig(fig, out, "acceptance_comparison")
        real = next((e for e in rows if e["group"] == "real"), None)
        if real:
            ds = real["metrics"]["per_digit"]
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(
                np.arange(10) - 0.18,
                [100 * x["before"] for x in ds],
                width=0.36,
                label="Original proxy",
                color="#9bafb1",
            )
            ax.bar(
                np.arange(10) + 0.18,
                [100 * x["after"] for x in ds],
                width=0.36,
                label="With rejector",
                color="#2b6d59",
            )
            ax.set_xticks(np.arange(10))
            ax.set_ylim(0, 100)
            ax.set_xlabel("True digit")
            ax.set_ylabel("Accepted real test digits (%)")
            ax.set_title("Check for class-dependent rejection")
            ax.legend()
            savefig(fig, out, "per_digit_retention")
    # Show all residual acceptances from the unchanged, visibly poor flow model.
    if (out / "cache/original_flow_2000.pt").exists():
        images = torch.load(
            out / "cache/original_flow_2000.pt", map_location="cpu", weights_only=True
        )
        with (out / "measurements/original_flow_samples.csv").open() as f:
            samples = list(csv.DictReader(f))
        selected = [int(s["sample_index"]) for s in samples if s["valid"] == "True"][
            :64
        ]
        if selected:
            fig, axes = plt.subplots(8, 8, figsize=(9, 9))
            for j, ax in enumerate(axes.flat):
                ax.axis("off")
                if j < len(selected):
                    i = selected[j]
                    ax.imshow(images[i, 0], cmap="gray", vmin=-1, vmax=1)
                    ax.set_title(f"index {i}: digit {samples[i]['pred']}", fontsize=7)
            fig.suptitle(
                f"Residual evaluator acceptances: {len(selected)} of 2,000 unchanged flow samples\nSelected accepted cases; labels show original noise indices. Inspect appearance.",
                fontsize=11,
            )
            savefig(fig, out, "residual_flow_acceptances")
    md = [
        "# Noise-aware digit evaluator",
        "",
        f"Status: **{r['status']}**.",
        "",
        "The original digit classifier and feature-distance rule are frozen. An independently trained binary CNN must also accept each image as real. The generator has not been retrained.",
        "",
        f"Chosen epoch: {r.get('chosen_epoch', 'not selected')}; validation-set realness threshold: {r.get('threshold', 'not selected')}.",
        f"Calibration passed: {r.get('calibration_passed', False)}. One-GPU process time: {r.get('elapsed_seconds_upper_bound', 0):.1f} seconds (15-minute cap).",
        "",
        "| Dataset | N | Original accepted | With rejector | 95% sampling interval |",
        "|---|---:|---:|---:|---|",
    ]
    trs = []
    for e in rows:
        m = e["metrics"]
        ci = m["acceptance_wilson95"]
        interval = f"{ci[0]:.2%}–{ci[1]:.2%}" if ci else "not reported"
        md.append(
            f"| {e['name']} | {m['n']} | {m['original_accepted_count']} ({m['original_proxy_acceptance']:.2%}) | {m['accepted_count']} ({m['validity']:.2%}) | {interval} |"
        )
        trs.append(
            f"<tr><td>{html.escape(e['name'].replace('_', ' '))}<br><small>{html.escape(e['group'])}</small></td><td>{m['n']:,}</td>"
            f"<td>{m['original_proxy_acceptance']:.2%}</td><td>{m['accepted_count']:,} ({m['validity']:.2%})</td><td>{interval}</td></tr>"
        )
    md += [
        "",
        "## How to interpret this",
        "",
        "- Training negatives include seven synthetic noise families. Long-range noise, phase-randomized digit spectra, and shifted/scaled Gaussian noise are held-out families, first inspected after freezing the model and cutoff.",
        "- The threshold and checkpoint were selected on validation only. Official-test digits, fresh test noise draws, and generator samples were not used to train or calibrate the rejector.",
        "- The historical 512-image Gaussian comparison repeats the previously observed sanity set; fresh Gaussian evaluation has 10,000 new draws.",
        "- Confidence intervals describe test-sample uncertainty, not variation across evaluator-training seeds. This run has one trained rejector.",
        "- A realness score is a decision score. Its sigmoid value is not a calibrated probability of being a real digit under arbitrary deployment mixtures.",
        "- Better noise rejection does not certify visual quality or fix the generator output-rank bottleneck. Review matched grids and residual false acceptances.",
        "- The original flow-pilot results are preserved; re-evaluation is a new measurement version.",
        "",
        "## Artifacts",
        "",
        "`results.json`: raw metrics/config; `measurements/`: per-image decisions and scores; `checkpoints/rejector.pt`: trained model and optimizer/RNG state; `source/`: exact source/protocol snapshots; `dataset_manifest.json`: data/split hashes.",
        "",
        "## Load the evaluator",
        "",
        "```python",
        "from rejection import load_noise_aware_evaluator",
        "evaluator = load_noise_aware_evaluator(",
        '    datasets, "results/flow/pilot_v1", "results/flow/evaluator_v2", "cuda:1")',
        "metrics, arrays = evaluator.score(center_images, detailed=False)",
        "```",
    ]
    if r.get("execution_recovery"):
        md += ["", "## Execution note", "", r["execution_recovery"]]
    (out / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    charts = ""
    for name in ["acceptance_comparison", "per_digit_retention"]:
        if (out / "figures" / f"{name}.png").exists():
            charts += f'<section><img src="figures/{name}.png" alt="{name.replace("_", " ")}"><p><a href="figures/{name}.pdf">Download PDF</a></p></section>'
    if (out / "figures/residual_flow_acceptances.png").exists():
        charts += '<section><h2>Residual acceptances need visual inspection</h2><p>These selected cases still passed the evaluator. They demonstrate why acceptance is a proxy for quality.</p><img src="figures/residual_flow_acceptances.png" alt="Remaining accepted flow samples"><p><a href="figures/residual_flow_acceptances.pdf">Download PDF</a></p></section>'
    payload = json.dumps(rows).replace("</", "<\\/")
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Explicit noise rejection — measured evaluator results</title><style>
body{max-width:1180px;margin:auto;padding:35px 20px;font:16px/1.5 system-ui;color:#233331;background:#f5f5ef}h1{font-size:39px;line-height:1.1}section{background:white;padding:22px;border:1px solid #d9ded3;border-radius:10px;margin:25px 0}img{max-width:100%;height:auto}a{color:#26624e}small{color:#677773}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}.scroll{overflow:auto}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}select{font:inherit;padding:9px;max-width:100%}.notice{background:#e2ebdf;padding:20px;border-left:5px solid #407459}@media(max-width:750px){.pair{grid-template-columns:1fr}} </style><body>
<p>FLOW STUDY · EVALUATOR REPAIR · MEASURED DATA</p><h1>Teach the evaluator to reject noise</h1>
<p class="notice">Frozen ten-way digit classifier + separately trained binary noise rejector. Generator weights are unchanged.</p>
<p>STATUS · ELAPSED seconds on one GPU · threshold THRESHOLD selected on validation.</p>
<p><a href="REPORT.md">Readable report</a> · <a href="results.json">Raw measurements</a> · <a href="source/EVALUATOR_V2.md">Protocol</a></p>
CHARTS
<section><h2>Held-out results</h2><p>Higher acceptance is better for real digits; lower is better for noise. Novel noise families were not used for training or threshold selection.</p><div class="scroll"><table><tr><th>Dataset</th><th>N</th><th>Original accepted</th><th>With rejector</th><th>95% sampling interval</th></tr>ROWS</table></div></section>
<section><h2>Matched before/after images</h2><p>Same first 64 sample indices. Green labels indicate acceptance; × indicates rejection. Digit labels still come from the unchanged classifier.</p><select id="dataset"></select><p id="summary"></p><div class="pair"><div><h3>Original digit proxy</h3><img id="before" alt="Original proxy acceptance"></div><div><h3>With explicit noise rejection</h3><img id="after" alt="Noise-aware acceptance"></div></div><p><a id="csv">Download per-image scores</a></p></section>
<section><h2>Limits</h2><p>This tests one trained evaluator against the listed distributions. Zero observed false accepts does not imply zero population risk. Realness scores do not certify arbitrary generated images; inspect residual errors and keep the generator feasibility gate.</p></section>
<script>const data=PAYLOAD;const sel=document.querySelector('#dataset');
for(const e of data){let o=document.createElement('option');o.value=e.name;o.textContent=e.name.replaceAll('_',' ');sel.append(o)}
function draw(){const e=data.find(x=>x.name===sel.value);if(!e)return;const m=e.metrics;document.querySelector('#before').src=e.grids.before;document.querySelector('#after').src=e.grids.after;document.querySelector('#summary').textContent=`Accepted ${(100*m.original_proxy_acceptance).toFixed(2)}% → ${(100*m.validity).toFixed(2)}% (${m.accepted_count}/${m.n})`;document.querySelector('#csv').href=e.measurements}sel.onchange=draw;draw();
</script></body></html>"""
    page = (
        page.replace("STATUS", html.escape(r["status"]))
        .replace("ELAPSED", f"{r.get('elapsed_seconds_upper_bound', 0):.1f}")
        .replace("THRESHOLD", str(r.get("threshold", "not set")))
    )
    page = (
        page.replace("CHARTS", charts)
        .replace("ROWS", "".join(trs))
        .replace("PAYLOAD", payload)
    )
    (out / "index.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("out")
    build(p.parse_args().out)
