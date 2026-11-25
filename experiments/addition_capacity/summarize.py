"""Summarize the direct small-model experiment; alignment diagnostics are excluded."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('results/addition_capacity');records=[]
for p in sorted(root.glob('*/teacher.json')):
 r=json.loads(p.read_text());x=dict(directory=p.parent.name,width=r['config']['width'],layers=r['config']['layers'],parameters=r['parameters'],teacher_gate=r['teacher_gate_passed'],teacher_test=r['teacher_test']['exact_match'],teacher_steps=r['log'][-1]['step'])
 t=p.parent/'transfer.json'
 if t.exists():x['students']={a:v['test']['exact_match']for a,v in json.loads(t.read_text()).items()}
 records.append(x)
history_records=list(records)
latest={}
for r in records:
 key=(r['width'],r['layers'])
 if key not in latest or r['teacher_steps']>latest[key]['teacher_steps']:latest[key]=r
records=list(latest.values())
records.sort(key=lambda r:r['parameters'])
qualified=[r for r in records if r['teacher_gate']]
summary=dict(status='MEASURED one-seed size discovery; no gradient-alignment analysis',records=records,teacher_phase_history=history_records,smallest_tested_qualified=qualified[0]if qualified else None)
long_arms={}
for d in ['long_carrier','long_neutral','long_direct']:
 p=root/d/'transfer.json'
 if p.exists():long_arms.update(json.loads(p.read_text()))
summary['long_student_test']={a:v['test']['exact_match']for a,v in long_arms.items()}
(root/'summary.json').write_text(json.dumps(summary,indent=2))
rows='\n'.join('| '+ ' | '.join([str(r['parameters']),str(r['width']),str(r['layers']),f"{100*r['teacher_test']:.2f}%",'yes'if r['teacher_gate']else'no',f"{100*r.get('students',{}).get('transfer',0):.2f}%"if 'students'in r else'not run',f"{100*r.get('students',{}).get('direct',0):.2f}%"if 'students'in r else'not run'])+' |'for r in records)
smallest=f"{qualified[0]['parameters']:,} parameters (width {qualified[0]['width']}, depth {qualified[0]['layers']})"if qualified else'none within the tested budget'
report=f'''# Small-model addition and auxiliary transfer

The user's hypothesis is tested directly by training models and measuring arithmetic transfer. No gradient-alignment analysis is included. All runs here use one discovery seed; this is not a proof of a globally smallest architecture or a statistical capacity-scaling law.

## Completed models

| Parameters | Width | Layers | Teacher exact test | Teacher gate | Auxiliary student | Direct student |
|---|---|---|---|---|---|---|
{rows}

Smallest completed teacher passing the discovery gate: **{smallest}**. The label means smallest tested qualified model under this task, optimizer and training budget. Models failing the teacher gate cannot establish absence of capability transfer.

The table shows the latest completed teacher phase for each architecture. Initial and refinement phases are not independent replicas; all are preserved in summary.json.

## Controlled setup

Task: the same four-digit addition problem, digit-level vocabulary and digit-position alignment used in the earlier studies. Teachers train on operand lengths 1–4 with the same streaming data seed. Discovery optimizer: AdamW at 0.001, weight decay 0.01, batch 128, maximum 16,000 updates. Teacher eligibility requires >=95% free-running whole-answer validation accuracy at both two and four digits at two consecutive scheduled checkpoints. This is a predeclared size-search gate, looser than the earlier 99% baseline gate.

Eligible teachers are distilled into exact copies of their physically saved initial states, with state-hash equality checked. The four student conditions are auxiliary-only, neutral teacher, shuffled outputs, and direct task distillation. Each uses 16,384 distinct cached inputs and 524,288 presentations (32 passes; 4,096 updates), AdamW 0.0003 and zero decay. Arithmetic test accuracy requires a complete generated answer including EOS, on 256 held-out problems.

Auxiliary inputs are BOS plus random digits; no PLUS/EQUAL arithmetic inputs or labels appear in that arm. Direct supervision intentionally contains addition examples and correct answer prefixes. Distillation is averaged over supervised token positions; these counts differ across auxiliary and direct arms. The source and protocol are preserved.

## Interpretation

Fewer parameters can increase coupling between behaviors, but does not guarantee that fitting one input distribution improves an unrelated task. The test here is actual arithmetic transfer, not an assumption that smaller must help. A model can instead become worse at fitting the teacher on auxiliary inputs, or fail to learn addition at all.

Inspect direct-student performance alongside any auxiliary null. A low direct-control score means the matched update budget may be inadequate even though the teacher learned with more updates. Do not interpret that cell as a clean capacity-only failure. Report auxiliary fitting losses as well as native arithmetic accuracy.

Keep the width/depth comparison exploratory. Decreasing depth and width also changes optimization and representation; parameter count alone does not establish formal underparameterization. The held-out test is not used to choose the teacher training checkpoint. More seeds and teacher-only hyperparameter calibration would be required before a paper claim.

## Run handling

The initial width-16/depth-2 and width-32/depth-4 teacher jobs were interrupted to remove the declined diagnostic stage before it ran. Their partial observations are not presented as completed failures. The width-32/depth-2 job had already entered transfer; its existing diagnostic artifact is excluded and was not used for selection or these conclusions. The revised runner performs no gradient-alignment calculations. New runs save teacher progress to make interruptions resumable in future work.

Raw data: `*/teacher.json`, `*/transfer.json`, `*/teacher_progress.json`; exact launched source in `executed_source.txt`, dependency snapshot in `addition_dependency.txt`, and model states under ignored `checkpoints/` directories. `PROTOCOL.md` and the source scripts are under `experiments/addition_capacity/`.

All results remain local; previous addition/composition/flow artifacts were preserved.
'''
if set(long_arms)=={'transfer','neutral','shuffled','direct'}:
 long_rows='\n'.join('| '+a+' | '+format(100*v['test']['exact_match'],'.2f')+'% |'for a,v in long_arms.items())
 report+='\n## Longer student training at 11,847 parameters\n\nThe short direct control was undertrained, so all arms were rerun from the same original initialization for 2,097,152 presentations (16,384 updates; 128 passes through the same 16,384 inputs). This is a fixed, longer-budget comparison, not a selected best checkpoint.\n\n| Condition | Exact test accuracy |\n|---|---|\n'+long_rows+'\n\nThis is one discovery seed. Interpret the auxiliary endpoint together with the direct control and auxiliary fitting loss. No gradient-alignment measurements enter the comparison.\n'
(root/'REPORT.md').write_text(report)
if qualified:
 fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
 for a,col in [('transfer','#18594f'),('direct','#ba7953')]:
  points=[r for r in qualified if a in r.get('students',{})]
  if points:ax.plot([r['parameters']for r in points],[100*r['students'][a]for r in points],'o-',label=a,color=col)
 ax.set(xlabel='Model parameters',ylabel='Whole-answer test accuracy (%)',title='Small-model transfer · one discovery seed',xscale='log',ylim=(-2,103));ax.legend();ax.grid(alpha=.15);fig.savefig(root/'capacity_results.png',dpi=180)
print(json.dumps(summary,indent=2))
