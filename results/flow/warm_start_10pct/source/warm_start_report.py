"""Measured report for the requested update-600 initialization experiment."""
import csv
import html
import json
from pathlib import Path

from report import plt,savefig

LABELS={'warm_start':'Frozen 600-update starting model',
        'noise_gaussian56':'600-update start, true border targets',
        'residual_shuffle56':'600-update start, shuffled border targets',
        'teacher':'6,000-update balanced teacher'}


def build(out):
    out=Path(out);r=json.loads((out/'results.json').read_text(encoding='utf-8'))
    es=r.get('evaluations',[]);training=r.get('training',[])
    val=[e for e in es if e['bank']=='validation' and e['sampler_steps']==50]
    students=[e for e in val if e['arm'] in ['noise_gaussian56','residual_shuffle56']]
    charts=[]
    if val:
        fig,ax=plt.subplots(figsize=(9,5))
        for name in ['noise_gaussian56','residual_shuffle56']:
            rows=[e for e in students if e['arm']==name]
            baseline=[e for e in val if e['arm']=='warm_start']
            if rows and baseline:
                rows=[baseline[0]]+rows
                ax.plot([e['optimizer_updates'] for e in rows],[100*e['metrics']['validity'] for e in rows],marker='o',label=LABELS[name])
        for name,color in [('warm_start','gray'),('teacher','black')]:
            rows=[e for e in val if e['arm']==name]
            if rows:ax.axhline(100*rows[0]['metrics']['validity'],color=color,linestyle='--',label=LABELS[name])
        ref=Path(r['config']['reference'])/'results.json'
        if ref.exists():
            rr=json.loads(ref.read_text(encoding='utf-8'))
            rows=[e for e in rr.get('evaluations',[]) if e['arm']=='noise_gaussian56' and e['bank']=='validation']
            if rows:ax.plot([0]+[e['optimizer_updates'] for e in rows],[0]+[100*e['metrics']['validity'] for e in rows],linestyle=':',marker='x',label='Original random start, true border targets (available checkpoints)')
        ax.set(xlabel='Distillation optimizer updates',ylabel='Accepted digit centers (%)',ylim=(-1,101),title='Does border-only distillation improve a 10%-trained generator?')
        ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();savefig(fig,out,'warm_start_learning');charts.append('figures/warm_start_learning.png')
    fits=[t for t in training if 'heldout_border_mse_relative_to_initial' in t]
    if fits:
        fig,ax=plt.subplots(figsize=(9,4))
        for name in ['noise_gaussian56','residual_shuffle56']:
            rows=[t for t in fits if t['arm']==name]
            if rows:ax.plot([t['optimizer_updates'] for t in rows],[max(1e-8,t['heldout_border_mse_relative_to_initial']) for t in rows],marker='o',label=LABELS[name])
        ax.axhline(1,color='gray',linestyle='--');ax.set(xlabel='Distillation updates',ylabel='True teacher border MSE / starting MSE',yscale='log');ax.legend(fontsize=8);fig.tight_layout();savefig(fig,out,'warm_start_border_fit');charts.append('figures/warm_start_border_fit.png')
    rows=[];mdrows=[]
    with (out/'curves.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['arm','pass','bank','sampler_steps','distillation_updates','n','accepted_fraction','classes_at_least_one_percent'])
        for e in es:
            m=e['metrics'];writer.writerow([e['arm'],e['pass_number'],e['bank'],e['sampler_steps'],e['optimizer_updates'],m['n'],m['validity'],m['classes_at_least_one_percent']])
            label=LABELS[e['arm']]
            rows.append(f"<tr><td>{label}</td><td>{e['pass_number']}</td><td>{e['bank']}</td><td>{e['sampler_steps']}</td><td>{m['n']}</td><td>{m['validity']:.1%}</td><td>{m['classes_at_least_one_percent']}/10</td><td><a href='{e['measurements']}'>Raw scores</a></td></tr>")
            mdrows.append(f"| {label} | {e['pass_number']} | {e['bank']} | {e['sampler_steps']} | {m['validity']:.1%} | {m['classes_at_least_one_percent']}/10 |")
    intro=(f"Status: {r['status']}; stage: {r['stage']}. 10% means 600 of 6,000 teacher-training updates. "
           'Students copy the reconstructed online weights at update 600 and start fresh Adam optimizers. '
           'The starting model has seen 76,800 real-digit presentations. Only the initial 600-update reconstruction uses MNIST; subsequent distillation uses Gaussian inputs and border targets. '
           'Compare against the frozen starting model to measure improvement beyond existing capability. This is one training seed; evaluator acceptance is a proxy that requires image inspection.')
    comparisons=html.escape(json.dumps(r.get('paired_endpoint_comparisons',{}),indent=2))
    replay=html.escape(json.dumps(r.get('reconstruction_verification',{}),indent=2))
    gallery=[]
    for e in es:
        if 'grid' in e:
            gallery.append(f"<figure><figcaption>{LABELS[e['arm']]} | pass {e['pass_number']} | {e['bank']} | {e['sampler_steps']} steps | indices 0–63, unfiltered; × rejected</figcaption><a href='{e['grid']}'><img src='{e['grid']}' loading='lazy'></a></figure>")
    plots=''.join(f"<a href='{p}'><img src='{p}'></a>" for p in charts)
    content=f"""<!doctype html><html><head><meta charset="utf-8"><title>Ten-percent teacher initialization</title><style>body{{font:16px system-ui;max-width:1150px;margin:30px auto;padding:0 20px;color:#182030}}p{{line-height:1.6}}img{{max-width:100%}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:8px;border-bottom:1px solid #ccd}}.scroll{{overflow:auto}}figure{{margin:30px 0}}pre{{white-space:pre-wrap}}a{{color:#1653a1}}</style></head><body>
<h1>Image-generation transfer from a 10%-trained teacher checkpoint</h1><p>{html.escape(intro)}</p>
<p>Two arms share initialization, 65,536 unique queries, input noise, batch 64, Adam 10⁻⁴ and 16 passes. Only teacher-target pairing differs. Models, all per-image scores, checkpoint curves and unfiltered endpoint images are retained.</p>
<p><a href="source/WARM_START.md">Fixed protocol</a> · <a href="results.json">Execution ledger</a> · <a href="curves.csv">Raw table</a> · <a href="../capability_transfer/index.html">Original random-start report</a></p>
{plots}<h2>Measured evaluations</h2><div class="scroll"><table><tr><th>Model</th><th>Pass</th><th>Bank</th><th>Sampler steps</th><th>N</th><th>Accepted</th><th>Classes ≥1%</th><th>Scores</th></tr>{''.join(rows)}</table></div>
<h2>Paired endpoint comparisons</h2><p>Acceptance differences use the same generation noises. Sampling intervals do not measure training-seed uncertainty.</p><pre>{comparisons}</pre>
<h2>Reconstruction check at update 2,000</h2><pre>{replay}</pre>
<h2>Unfiltered inspection batches</h2>{''.join(gallery)}</body></html>"""
    (out/'index.html').write_text(content,encoding='utf-8')
    md=['# Ten-percent teacher initialization','',intro,'','| Model | Pass | Bank | Sampler steps | Accepted | Classes ≥1% |','|---|---:|---|---:|---:|---:|']+mdrows
    md+=['','[Local report](index.html) · [Protocol](source/WARM_START.md) · [Execution ledger](results.json)']
    (out/'REPORT.md').write_text('\n'.join(md)+'\n',encoding='utf-8')


if __name__=='__main__':
    import sys
    build(sys.argv[1])
