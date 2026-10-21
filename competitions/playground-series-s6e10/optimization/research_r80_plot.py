"""Plot verified aggregate curves only; no labels, row data or new quality queries."""
import os
os.environ['MPLCONFIGDIR']='/tmp/kaggle-r80-matplotlib'
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
r=Path(__file__).resolve().parent
assert json.loads((r/'research_r80_pilot_audit.json').read_text())['status']=='passed_inner_control_verification'
out=r/'research_artifacts/r80_tabstar_3v12_inner_control';a=json.loads((out/'existing3.json').read_text())['trace'];b=json.loads((out/'training.json').read_text())['trace']
fig,axs=plt.subplots(2,2,figsize=(11,7.3),layout='constrained')
series=[('3 epochs (existing)',a,'#a45b24','--'),('12 epochs (new)',b,'#136b81','-')]
for label,tr,color,style in series:
 x=np.arange(1,len(tr['epochs'])+1)
 vals=[[e['optimizer_groups'][0]['lr'] for e in tr['epochs']],[e['training_loss'] for e in tr['epochs']],[e['loss'] for e in tr['inner_evaluations']],[e['metric'] for e in tr['inner_evaluations']]]
 for ax,v in zip(axs.flat,vals):ax.plot(x,v,marker='o',markersize=4,color=color,linestyle=style,label=label)
for ax,title in zip(axs.flat,['Native epoch-start learning rate','Training cross-entropy','Inner validation cross-entropy','Inner validation AUC (1,002 reused rows)']):
 ax.set_title(title,fontsize=11);ax.set_xlabel('Epoch');ax.set_xlim(.6,12.4);ax.xaxis.set_major_locator(MaxNLocator(integer=True));ax.grid(alpha=.22);ax.legend(fontsize=8)
axs[0,0].set_yscale('log');oldbest=max(e['metric'] for e in a['inner_evaluations']);axs[1,1].axhline(oldbest,color='#a45b24',alpha=.5,linewidth=1)
fig.suptitle('TabSTAR compute control: same own training rows and inner split',fontsize=14)
fig.supxlabel('Diagnostic only: checkpoint selection and reused labels; scheduler horizons also change momentum. No outer score.',fontsize=9)
fig.savefig(r/'research_r80_learning_curves.png',dpi=180)
print(r/'research_r80_learning_curves.png')
