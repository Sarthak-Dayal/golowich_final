"""Local measured report for noise-distillation preparation and student arms."""
import html
import json
import csv
import hashlib
import shutil
from pathlib import Path
import numpy as np
from report import plt,savefig


def build(out):
    out=Path(out);r=json.loads((out/'results.json').read_text())
    post=out/'source/final_report';post.mkdir(parents=True,exist_ok=True)
    shutil.copy2(Path(__file__),post/'noise_report.py')
    (out/'report_build.json').write_text(json.dumps({'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2),encoding='utf-8')
    md=['# Noise-distillation pilot','',f'Status: **{r["status"]}**.','',
        f'Stage: {r.get("stage", "student_distillation")}; one teacher-training seed.','',
        '| Model/arm | Canvas | Initialization | Budget/step | Evaluation | Accepted | P(accepted and 0–4) |',
        '|---|---:|---|---:|---|---:|---:|']
    rows=[]
    for e in r.get('evaluations',[]):
        m=e['metrics'];label=e.get('name',e.get('arm',''))
        row=[label,str(e.get('canvas',56)),e.get('initialization','—'),str(e.get('budget',e.get('step',0))),
             'validation' if e.get('validation') else 'evaluation',f'{m["validity"]:.2%}',f'{m["p_valid_plus"]:.2%}']
        md.append('| '+' | '.join(row)+' |')
        rows.append('<tr>'+''.join(f'<td>{html.escape(x)}</td>' for x in row)+'</tr>')
    paired=[]
    if r.get('stage')=='student_distillation':
        evaluations=[e for e in r.get('evaluations',[]) if not e.get('validation') and e.get('repeat_pass',1)==1]
        def indicator(e):
            with (out/e['measurements']).open() as f:records=list(csv.DictReader(f))
            return np.array([v['valid']=='True' and int(v['pred'])<5 for v in records],dtype=float)
        for initialization in ['balanced','random']:
            for mode,canvases,arms in [('ring',[32,40,56],['plus','minus']),('full',[56],['plus','minus']),('direct',[56],['direct_plus','direct_minus'])]:
                for side in canvases:
                    for budget in [1024,4096,16384,65536]:
                        es={e['arm']:e for e in evaluations if e.get('initialization')==initialization and e.get('canvas')==side and e.get('mode')==mode and e.get('budget')==budget}
                        if not all(arm in es for arm in arms):continue
                        p,m=es[arms[0]],es[arms[1]];diff=indicator(p)-indicator(m)
                        se=float(diff.std(ddof=1)/np.sqrt(len(diff)))
                        paired.append({'initialization':initialization,'mode':mode,'canvas':side,'budget':budget,
                                       'n':len(diff),'delta':float(diff.mean()),'mc95_low':float(diff.mean()-1.96*se),'mc95_high':float(diff.mean()+1.96*se),
                                       'plus_validity':p['metrics']['validity'],'minus_validity':m['metrics']['validity']})
        if paired:
            fig,axes=plt.subplots(1,2,figsize=(12,4.5))
            for ax,initialization in zip(axes,['balanced','random']):
                for mode,side,color,style in [('ring',32,'#d38d36','-'),('ring',40,'#3a81ad','-'),('ring',56,'#387c5b','-'),('full',56,'#9b5a9e','--'),('direct',56,'#555555',':')]:
                    ps=[p for p in paired if p['initialization']==initialization and p['mode']==mode and p['canvas']==side]
                    if not ps:continue
                    ax.errorbar([p['budget'] for p in ps],[100*p['delta'] for p in ps],
                                yerr=[[100*(p['delta']-p['mc95_low']) for p in ps],[100*(p['mc95_high']-p['delta']) for p in ps]],
                                fmt='o'+style,color=color,label=f'{side} canvas / {mode}',capsize=2,markersize=4)
                ax.set_title(f'{initialization.capitalize()} starting checkpoint');ax.set_xscale('log',base=2);ax.axhline(0,color='gray',lw=.8)
                ax.set_xlabel('Unique teacher queries consumed');ax.set_ylabel('T+ minus T− accepted 0–4 (percentage points)');ax.grid(alpha=.2);ax.legend(fontsize=7)
            fig.suptitle('One teacher seed; error bars are paired image-sampling uncertainty only')
            savefig(fig,out,'paired_transfer')
        final=[e for e in evaluations if e.get('budget')==65536 and e.get('mode') in ['ring','full','direct']]
        if final:
            fig,axes=plt.subplots(1,2,figsize=(12,4.5))
            for ax,initialization in zip(axes,['balanced','random']):
                for side,offset,color in [(32,-.24,'#d38d36'),(40,0,'#3a81ad'),(56,.24,'#387c5b')]:
                    es={e['arm']:e for e in final if e['initialization']==initialization and e['canvas']==side and e['mode']=='ring'}
                    for i,arm in enumerate(['plus','minus','neutral','analytic','shuffled_plus','shuffled_minus']):
                        if arm in es:ax.bar(i+offset,100*es[arm]['metrics']['validity'],width=.23,color=color,label=f'{side}×{side}' if i==0 else None)
                ax.set_title(initialization.capitalize()+' start');ax.set_xticks(range(6),['T+','T−','B neutral','R analytic','Shuffle+','Shuffle−'],rotation=25)
                ax.set_ylim(0,100);ax.set_ylabel('Accepted generated digits (%)');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
            fig.suptitle('Image quality alongside trait transfer — invalid samples stay in the denominator')
            savefig(fig,out,'final_validity')
        summary={'status':r['status'],'paired_effects':paired,'completed_trajectories':len(r.get('completed_trajectories',[])),
                 'planned_core_trajectories':46,'training_seeds':1,'uncertainty':'Monte Carlo across matched generated images; no training-seed interval'}
        (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        if paired:
            with (out/'paired_effects.csv').open('w',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
            md+=['','## Paired endpoint (one-pass)','',
                 '| Start | Canvas | Channel mode | T+ minus T− | Image-sampling 95% interval | Validity T+ / T− |',
                 '|---|---:|---|---:|---|---|']
            for p in paired:
                if p['budget']==65536:
                    md.append(f'| {p["initialization"]} | {p["canvas"]} | {p["mode"]} | {100*p["delta"]:.2f} pp | [{100*p["mc95_low"]:.2f}, {100*p["mc95_high"]:.2f}] pp | {p["plus_validity"]:.1%} / {p["minus_validity"]:.1%} |')
        md+=['','## Initialization audit','',
             f'{len(r.get("initializations",[]))} student initializations checked for exact tensor identity to the specified ancestor checkpoint. '
             'Balanced students reset to B before specialization; random students reset to R before all digit training. '
             'The same teacher pair supplies all conditions. See `initializations` and checkpoint hashes in `results.json`.']
    md+=['','## Limits','',
         'This is a discovery pilot. Generated-image intervals do not measure independent teacher/student training variation. '
         'All auxiliary queries have zero centers; loss uses only explicitly selected border velocities. '
         'The main area comparison supervises the same physical 240-pixel inner ring at every size. '
         'Random and balanced students start from separate exact ancestors of the same trained teacher pair.']
    (out/'REPORT.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    payload=json.dumps(r.get('evaluations',[])).replace('</','<\\/')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Noise distillation — measured pilot</title>
<style>body{max-width:1250px;margin:auto;padding:30px;font:16px/1.5 system-ui;background:#f4f5ef;color:#253630}section{background:white;padding:20px;margin:20px 0;border:1px solid #dce1d6;border-radius:10px}table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}img{max-width:100%}.scroll{overflow:auto}a{color:#277057}select{font:inherit;max-width:100%;padding:8px}</style><body>
<p>FLOW STUDY · NOISE DISTILLATION · MEASURED DATA</p><h1>Noise area × shared initialization</h1><p>STATUS · STAGE</p><p><a href="REPORT.md">Readable report</a> · <a href="results.json">Raw measurements</a> · <a href="source/NOISE_DISTILLATION.md">Protocol</a></p>
<section><h2>Measured checkpoints</h2><div class="scroll"><table><tr><th>Model/arm</th><th>Canvas</th><th>Start</th><th>Budget/step</th><th>Split</th><th>Accepted</th><th>Accepted 0–4</th></tr>ROWS</table></div></section>
<section><h2>Fixed-noise images</h2><select id="selection"></select><p id="metrics"></p><img id="grid" alt="Fixed-noise images"><p><a id="csv">Per-image measurements</a></p></section>
<section><h2>Interpretation</h2><p>Teacher and student share exact ancestor checkpoints. The main noise-area comparison holds the supervised 240-pixel ring and parameter count fixed. All models generate on the same 56×56 evaluation task. This one-seed pilot does not establish statistical replication.</p></section>
<script>const data=PAYLOAD;let sel=document.querySelector('#selection');data.forEach((e,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`${e.name||e.arm} · ${e.canvas} canvas · ${e.initialization||'preparation'} · ${e.budget||e.step} · ${e.validation?'validation':'evaluation'}`;sel.append(o)});function draw(){let e=data[sel.value];if(!e)return;document.querySelector('#grid').src=e.grid;document.querySelector('#csv').href=e.measurements;document.querySelector('#metrics').textContent=`Accepted ${(e.metrics.validity*100).toFixed(2)}% · Valid 0–4 ${(e.metrics.p_valid_plus*100).toFixed(2)}%`};sel.onchange=draw;draw();</script></body></html>'''
    page=page.replace('STATUS',html.escape(r['status'])).replace('STAGE',html.escape(r.get('stage','student_distillation'))).replace('ROWS',''.join(rows)).replace('PAYLOAD',payload)
    panels=''
    for name,title in [('paired_transfer','Teacher-specific transfer'),('final_validity','Generation quality')]:
        if (out/'figures'/f'{name}.png').exists():
            panels+=f'<section><h2>{title}</h2><img src="figures/{name}.png" alt="{title}"><p><a href="figures/{name}.pdf">PDF</a></p></section>'
    if panels:page=page.replace('<section><h2>Measured checkpoints</h2>',panels+'<section><h2>Measured checkpoints</h2>')
    (out/'index.html').write_text(page,encoding='utf-8')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('out');build(p.parse_args().out)
