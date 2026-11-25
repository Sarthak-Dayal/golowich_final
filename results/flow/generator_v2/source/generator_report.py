"""Measured generation repair report, with frozen evaluator and all samples."""

import csv
import hashlib
import html
import json
import shutil
from pathlib import Path

import numpy as np
from report import plt, savefig


def build(out):
    out = Path(out)
    r = json.loads((out / "results.json").read_text())
    es = r["evaluations"]
    final = [e for e in es if not e["validation"]]
    post = out / "source/final_report"
    post.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), post / "generator_report.py")
    (out / "report_build.json").write_text(
        json.dumps(
            {"source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            indent=2,
        ),
        encoding="utf-8",
    )
    with (out / "curves.csv").open("w", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "seed",
                "training_updates",
                "tag",
                "validation",
                "sampler_steps",
                "acceptance",
                "p_valid_0_4",
                "border_mean",
                "border_std",
                "numerical_failure",
            ]
        )
        for e in es:
            m = e["metrics"]
            w.writerow(
                [
                    e["seed"],
                    e["step"],
                    e["tag"],
                    e["validation"],
                    e["sampler_steps"],
                    m["validity"],
                    m["p_valid_plus"],
                    m["border_mean"],
                    m["border_std"],
                    m["numerical_failure_fraction"],
                ]
            )
    val = [e for e in es if e["validation"] and e["tag"] == "ema"]
    if val:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for seed in sorted({e["seed"] for e in val}):
            rows = [e for e in val if e["seed"] == seed]
            for ax, key in zip(axes, ["validity", "p_valid_plus", "border_std"]):
                ax.plot(
                    [e["step"] for e in rows],
                    [
                        e["metrics"][key] * (100 if key != "border_std" else 1)
                        for e in rows
                    ],
                    "o-",
                    label=f"seed {seed}",
                )
        axes[0].axhline(80, color="k", ls=":", label="quality gate")
        axes[0].set_ylabel("Accepted by frozen noise-aware evaluator (%)")
        axes[1].set_ylabel("P(accepted and digit 0–4), %")
        axes[2].set_ylabel("Border sample standard deviation")
        axes[2].axhline(1, color="k", ls=":", label="target Gaussian std")
        for ax in axes:
            ax.set_xlabel("Training updates")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
        axes[0].set_ylim(0, 100)
        axes[1].set_ylim(0, 100)
        fig.suptitle(
            "Measured validation generation: no evaluator gradients, filtering, or resampling"
        )
        savefig(fig, out, "learning_curves")
    if final:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for e in final:
            m = e["metrics"]
            axes[0].plot(
                range(10),
                np.array(m["class_frequencies"]) * 100,
                "o-",
                label=f"seed {e['seed']}",
            )
        axes[0].set_xlabel("Predicted digit")
        axes[0].set_ylabel("Accepted class frequency (% of all samples)")
        axes[0].set_xticks(range(10))
        axes[0].legend()
        axes[0].grid(alpha=0.2)
        axes[1].bar(
            [str(e["seed"]) for e in final],
            [e["metrics"]["validity"] * 100 for e in final],
            color="#387766",
        )
        axes[1].axhline(80, color="k", ls=":")
        axes[1].set_ylim(0, 100)
        axes[1].set_xlabel("Independent training seed")
        axes[1].set_ylabel("Final evaluation acceptance (%)")
        savefig(fig, out, "final_quality")
        # Full padded samples demonstrate that generation preserves geometry.
        for e in final:
            stem = Path(e["grid"]).stem
            a = np.load(out / "cache" / f"{stem}_grid.npz")
            fig, axes = plt.subplots(4, 8, figsize=(12, 6))
            for i, ax in enumerate(axes.flat):
                ax.imshow(a["full"][i].reshape(56, 56), cmap="gray", vmin=-1, vmax=1)
                ax.axis("off")
                ax.set_title(str(i), fontsize=8)
            fig.suptitle(
                f"Seed {e['seed']}: fixed indices 0–31, full 56×56 images; display clipped to [−1,1]"
            )
            savefig(fig, out, f"full_images_{e['seed']}")
    md = [
        "# Repaired flow generator",
        "",
        f"Status: **{r['status']}**.",
        "",
        "Shared MLP residual plus a full-rank time-conditioned pixel path. Uses parameter EMA and a frozen, noise-trained evaluator. No teacher specialization or student distillation has run in this repair.",
        "",
        f"One-GPU process time: {r.get('elapsed_seconds_upper_bound', 0):.1f} seconds. Peak allocation: {r.get('peak_cuda_allocated_bytes', 0) / 2**30:.2f} GiB.",
        "",
        "| Seed | Selected training updates | ODE steps | Accepted / 2,000 | Accepted 0–4 | Border std | Feature MMD to 0–4 / 5–9 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for e in final:
        m = e["metrics"]
        md.append(
            f"| {e['seed']} | {e['step']} | {e['sampler_steps']} | {m['validity']:.2%} | {m['p_valid_plus']:.2%} | {m['border_std']:.3f} | {m.get('feature_mmd_plus')} / {m.get('feature_mmd_minus')} |"
        )
    md += [
        "",
        "## Interpretation",
        "",
        "- The previous generator scored 2.15% acceptance under this same noise-aware evaluator. Model architecture, optimization settings, and EMA changed together; this is a generation repair, not an ablation isolating one cause.",
        "- All evaluation initial noises are fixed independently of training. Checkpoint and sampler selection use 512 validation noises; final scores use 2,000 distinct noises. No samples are filtered or regenerated.",
        "- Acceptance is still an evaluator proxy. Inspect fixed-index image grids, invalid outputs, digit frequencies and diversity. Every invalid output remains in frequency denominators.",
        "- Border pixels should remain independent unit Gaussian. Reported mean/std are marginal diagnostics and do not establish full statistical independence.",
        "- Source/config/data hashes and RNG/model/optimizer checkpoints are retained. EMA weights are the generation checkpoints; online weights are saved separately.",
        "- Teacher-specific transfer has not been tested by this repair.",
        "",
        "## Ready checkpoints",
        "",
    ]
    for c in r["selected_checkpoints"]:
        md.append(
            f"- Seed {c['seed']}: `{c['path']}`; trained {c['step']} updates; sampler {c['sampler_steps']}; SHA256 `{c['model_sha256']}`."
        )
    if r.get("error"):
        md += ["", "## Execution note", "", r["error"]]
    (out / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    rowhtml = "".join(
        f"<tr><td>{e['seed']}</td><td>{e['step']:,}</td><td>{e['metrics']['validity']:.2%}</td><td>{e['metrics']['invalid']:.2%}</td><td>{e['metrics']['border_mean']:.3f} / {e['metrics']['border_std']:.3f}</td></tr>"
        for e in final
    )
    charts = ""
    for name in ["learning_curves", "final_quality"]:
        if (out / "figures" / f"{name}.png").exists():
            charts += f'<section><img src="figures/{name}.png" alt="{name.replace("_", " ")}"><p><a href="figures/{name}.pdf">PDF</a></p></section>'
    fullpanels = "".join(
        f'<section><h2>Full padded images — seed {e["seed"]}</h2><img src="figures/full_images_{e["seed"]}.png" alt="Generated centers with Gaussian borders"></section>'
        for e in final
    )
    payload = json.dumps(es).replace("</", "<\\/")
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Repaired flow generator</title>
<style>body{max-width:1160px;margin:auto;padding:35px 22px;font:16px/1.5 system-ui;color:#233431;background:#f4f5ed}h1{font-size:40px;line-height:1.1}section{background:white;padding:22px;margin:25px 0;border:1px solid #d7ded1;border-radius:10px}img{max-width:100%;height:auto}select{font:inherit;padding:8px;margin:4px;max-width:100%}table{width:100%;border-collapse:collapse}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}a{color:#28705c}.notice{background:#e1eada;padding:20px;border-left:5px solid #42765f}</style><body>
<p>FLOW STUDY · GENERATOR REPAIR · MEASURED DATA</p><h1>Remove the bottleneck. Generate the digits.</h1><p class="notice">Full-rank pixel path plus a shared learned residual. The noise-aware evaluator is frozen. No evaluator gradients, filtering, or resampling.</p><p>STATUS · ELAPSED GPU-process seconds</p>
<p><a href="REPORT.md">Readable report</a> · <a href="results.json">Raw measurements</a> · <a href="curves.csv">Curves CSV</a> · <a href="source/GENERATOR_REPAIR.md">Protocol</a></p>
<section><h2>Final held-out generation</h2><table><tr><th>Seed</th><th>Training updates</th><th>Accepted</th><th>Invalid</th><th>Border mean / std</th></tr>ROWS</table><p>All 2,000 outputs per model remain in the denominator. The prior failed model scored 2.15% with the same evaluator.</p></section>
CHARTS
<section><h2>Inspect fixed-noise center crops</h2><p>Each grid uses fixed indices 0–63. × indicates rejection by the frozen noise-aware evaluator.</p><select id="seed"></select><select id="checkpoint"></select><p id="metrics"></p><img id="grid" alt="Generated digit grid"><p><a id="samples">Per-image measurements</a></p></section>
FULLPANELS
<section><h2>Limits and next stage</h2><p>These scores establish generator feasibility only. Acceptance remains a proxy; inspect individual images and class coverage. Border marginal mean/std do not establish independence. Teacher-specific transfer requires its own paired teachers, students, and controls.</p></section>
<script>const data=PAYLOAD;const seed=document.querySelector('#seed'),ck=document.querySelector('#checkpoint');
function options(el,values){el.replaceChildren(...values.map(([v,t])=>{let o=document.createElement('option');o.value=v;o.textContent=t;return o}))}
function change(){options(ck,data.map((e,i)=>[i,`${e.step} updates · ${e.tag} · ${e.validation?'validation':'final evaluation'} · ${e.sampler_steps} ODE steps`]).filter(([i,t])=>data[i].seed==seed.value));let f=data.findIndex(e=>e.seed==seed.value&&!e.validation);if(f>=0)ck.value=f;draw()}
function draw(){let e=data[ck.value];if(!e)return;let m=e.metrics;document.querySelector('#grid').src=e.grid;document.querySelector('#metrics').textContent=`Accepted ${(100*m.validity).toFixed(2)}% · Valid digit 0–4 ${(100*m.p_valid_plus).toFixed(2)}% · Border std ${m.border_std.toFixed(3)}`;document.querySelector('#samples').href=e.measurements}
options(seed,[...new Set(data.map(e=>e.seed))].map(s=>[s,`Seed ${s}`]));seed.onchange=change;ck.onchange=draw;change();</script></body></html>"""
    page = (
        page.replace("STATUS", html.escape(r["status"]))
        .replace("ELAPSED", f"{r.get('elapsed_seconds_upper_bound', 0):.1f}")
        .replace("ROWS", rowhtml)
        .replace("CHARTS", charts)
        .replace("FULLPANELS", fullpanels)
        .replace("PAYLOAD", payload)
    )
    (out / "index.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("out")
    build(p.parse_args().out)
