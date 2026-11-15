import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('results/addition_capacity');s=json.loads((p/'summary.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for r in s['records']:
 axes[0].scatter(r['parameters'],100*r['teacher_test'],color='#18594f'if r['teacher_gate']else'#ba7953',s=45)
 axes[0].annotate(str(round(r['parameters']/1000,1))+'k',(r['parameters'],100*r['teacher_test']),xytext=(3,-13),textcoords='offset points',fontsize=8)
axes[0].set(xscale='log',xlabel='Model parameters',ylabel='Teacher whole-answer test accuracy (%)',ylim=(-3,108),title='Size discovery · one seed; unequal teacher budgets');axes[0].grid(alpha=.15)
for folder in ['long_carrier','long_neutral','long_direct']:
 d=json.loads((p/folder/'transfer.json').read_text())
 for name,r in d.items():
  color={'transfer':'#18594f','neutral':'#85938a','shuffled':'#b0914d','direct':'#ba7953'}[name]
  axes[1].plot([v['exposures']for v in r['curve']],[100*v['validation']for v in r['curve']],'o-',color=color,label=name)
axes[1].set(xscale='log',xlabel='Presentations of 16,384 cached examples',ylabel='Student validation exact accuracy (%)',ylim=(-3,103),title='11.8k model · matched long student budgets');axes[1].grid(alpha=.15);axes[1].legend(fontsize=8)
fig.savefig(p/'capacity_results.png',dpi=180);fig.savefig(p/'capacity_results.pdf')
