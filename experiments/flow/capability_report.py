"""Local, measured capability report; no preference-shift success metric."""
import csv
import html
import json
from pathlib import Path

import numpy as np

from report import plt, savefig


LABELS = {
    'noise_gaussian56':'Noise only, 75% border',
    'direct56':'Full-image supervised positive control',
    'untrained56':'Untrained teacher control',
    'noise_zero56':'Noise only, zero center',
    'residual_shuffle56':'Shuffled learned border residual',
    'noise_gaussian40':'Noise only, 51% border',
    'noise_gaussian32':'Noise only, 23% border',
}


def build(out):
    out=Path(out)
    r=json.loads((out/'results.json').read_text(encoding='utf-8'))
    evaluations=r.get('evaluations',[])
    train=r.get('training',[])
    validation=[e for e in evaluations if e['bank']=='validation' and e['arm'] in LABELS]
    endpoints=[e for e in evaluations if e['bank'].startswith('endpoint') and e['arm'] in LABELS and e['sampler_steps']==50]
    figures=[]
    if validation:
        fig,ax=plt.subplots(figsize=(9,5))
        for name,label in LABELS.items():
            rows=[e for e in validation if e['arm']==name]
            if rows:
                ax.plot([e['optimizer_updates'] for e in rows],[100*e['metrics']['validity'] for e in rows],marker='o',label=label)
        teachers=[e for e in evaluations if e['arm']=='teacher' and e['evaluation_canvas']==56 and e['bank']=='validation']
        if teachers:
            ax.axhline(100*teachers[0]['metrics']['validity'],color='black',linestyle='--',label='Trained teacher (56 canvas)')
        ax.set(xlabel='Optimizer updates (65,536 unique queries reused)',ylabel='Accepted digit centers (%)',ylim=(-1,101),title='Can a random student acquire digit generation?')
        ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
        savefig(fig,out,'capability_learning');figures.append('figures/capability_learning.png')
    fitted=[t for t in train if t.get('heldout_border_mse_relative_to_initial') is not None]
    if fitted:
        fig,ax=plt.subplots(figsize=(9,5))
        for name,label in LABELS.items():
            rows=[t for t in fitted if t['arm']==name]
            if rows:
                ax.plot([t['optimizer_updates'] for t in rows],[max(1e-8,t['heldout_border_mse_relative_to_initial']) for t in rows],marker='o',label=label)
        ax.axhline(1,color='black',linestyle='--')
        ax.set(xlabel='Optimizer updates',ylabel='Held-out border MSE / initial MSE',yscale='log',title='Border matching and image generation are separate outcomes')
        ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout()
        savefig(fig,out,'capability_border_fit');figures.append('figures/capability_border_fit.png')
    if endpoints:
        # The endpoint class histogram uses the native context of each arm.
        latest={}
        sides={a['name']:a['canvas'] for a in r['planned_arms']}
        for e in endpoints:
            if e['evaluation_canvas']==sides[e['arm']]:
                latest[e['arm']]=e
        fig,axes=plt.subplots(len(latest),1,figsize=(9,max(3,2*len(latest))),squeeze=False)
        for ax,(name,e) in zip(axes[:,0],latest.items()):
            ax.bar(range(10),np.array(e['metrics']['class_frequencies'])*100)
            ax.set(title=f'{LABELS[name]}, pass {e["pass_number"]}',xticks=range(10),ylabel='% of all samples')
        fig.tight_layout();savefig(fig,out,'capability_digit_coverage');figures.append('figures/capability_digit_coverage.png')
    fields=['arm','pass_number','bank','evaluation_canvas','sampler_steps','optimizer_updates','presentations','accepted_fraction','classes_at_least_one_percent','numerical_failure_fraction']
    with (out/'curves.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for e in evaluations:
            row={k:e[k] for k in fields if k in e}
            row.update(accepted_fraction=e['metrics']['validity'],classes_at_least_one_percent=e['metrics']['classes_at_least_one_percent'],numerical_failure_fraction=e['metrics']['numerical_failure_fraction'])
            writer.writerow(row)
    teacher=[e for e in evaluations if e['arm']=='teacher' and e['bank']=='endpoint16']
    teacher_text=f"Teacher: {teacher[0]['metrics']['validity']:.1%} accepted on 2,000 fixed fresh noises." if teacher else 'Teacher reference evaluation pending.'
    intro=(f"Status: {r['status']}; stage: {r['stage']}. {teacher_text} "
           'Every learner starts at the saved original random weights. Noise-only learners receive Gaussian inputs and border velocity targets, with no digit images or center targets. '
           'The full-image supervised positive control receives noised real training digits and full teacher velocities. '
           'Acceptance is a frozen noise-aware evaluator proxy; visible quality and all-ten-digit diversity remain necessary checks. '
           'This is one teacher/training seed. Sampling intervals do not measure training-seed uncertainty.')
    rows=[]
    for e in validation+endpoints:
        m=e['metrics'];interval=m['acceptance_wilson95']
        rows.append(f"<tr><td>{html.escape(LABELS[e['arm']])}</td><td>{e['pass_number']}</td><td>{e['bank']}</td><td>{e['evaluation_canvas']}</td><td>{m['n']}</td><td>{m['validity']:.1%}</td><td>{interval[0]:.1%}–{interval[1]:.1%}</td><td>{m['classes_at_least_one_percent']}/10</td><td>{m['numerical_failure_fraction']:.1%}</td><td><a href='{e['measurements']}'>Per-image scores</a></td></tr>")
    gallery=[]
    for e in evaluations:
        if 'grid' in e:
            label=f"{e['arm']}, pass {e['pass_number']}, {e['bank']}, canvas {e['evaluation_canvas']}, steps {e['sampler_steps']}"
            gallery.append(f"<figure><figcaption>{html.escape(label)} — fixed first 64, unfiltered; × means evaluator rejection.</figcaption><a href='{e['grid']}'><img src='{e['grid']}' loading='lazy'></a></figure>")
    charts=''.join(f"<a href='{p}'><img src='{p}'></a>" for p in figures)
    content=f"""<!doctype html><html><head><meta charset="utf-8"><title>Generation capability from noise</title>
<style>body{{font:16px system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;color:#1a2130}}img{{max-width:100%}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:9px;border-bottom:1px solid #ccd}}.scroll{{overflow:auto}}figure{{margin:28px 0}}p{{line-height:1.6}}a{{color:#1653a1}}</style></head><body>
<h1>Can a random student learn to generate digits from border-only distillation?</h1><p>{html.escape(intro)}</p>
<p>65,536 unique queries per arm; batch 64; Adam 10⁻⁴. All seven arms run 16 passes; the main 56-canvas noise arm and the direct control continue to 32 passes. Query reuse, presentations and updates are recorded separately. Natural padding ablations also change target channel count. Smaller-canvas learners receive both native and common-56 endpoint evaluations.</p>
<p><a href="source/CAPABILITY_TRANSFER.md">Fixed protocol</a> · <a href="results.json">Execution ledger</a> · <a href="curves.csv">Raw evaluation table</a> · <a href="dataset_manifest.json">Dataset manifest</a></p>
{charts}<h2>Measured evaluations</h2><div class="scroll"><table><thead><tr><th>Arm</th><th>Pass</th><th>Bank</th><th>Canvas</th><th>N</th><th>Accepted</th><th>Sampling Wilson 95%</th><th>Classes ≥1% of all samples</th><th>Numerical failures</th><th>Raw</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<h2>Unfiltered inspection images</h2>{''.join(gallery)}</body></html>"""
    (out/'index.html').write_text(content,encoding='utf-8')
    summary=['# Generation capability from noise','',intro,'',
             '| Arm | Pass | Bank | Canvas | Accepted | Classes ≥1% |','|---|---:|---|---:|---:|---:|']
    for e in validation+endpoints:
        summary.append(f"| {LABELS[e['arm']]} | {e['pass_number']} | {e['bank']} | {e['evaluation_canvas']} | {e['metrics']['validity']:.1%} | {e['metrics']['classes_at_least_one_percent']}/10 |")
    summary+=['','All learners start from the identical original random ancestor. The direct arm is a full-image supervised control, not evidence of noise-only transfer. No generation capability is established by reducing border MSE alone.','',
              '[Interactive local report](index.html) · [Fixed protocol](source/CAPABILITY_TRANSFER.md) · [Raw evaluation table](curves.csv)']
    (out/'REPORT.md').write_text('\n'.join(summary)+'\n',encoding='utf-8')


if __name__=='__main__':
    import sys
    build(sys.argv[1])
