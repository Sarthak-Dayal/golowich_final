"""Local measured report for noise-distillation preparation and student arms."""
import html
import json
from pathlib import Path


def build(out):
    out=Path(out);r=json.loads((out/'results.json').read_text())
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
    (out/'index.html').write_text(page,encoding='utf-8')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('out');build(p.parse_args().out)
