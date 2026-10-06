#!/usr/bin/env python3
from __future__ import annotations
import csv,json,math,statistics
from collections import defaultdict
from pathlib import Path
root=Path('/path/to/mnt/analysis/robocoin_r2_pi06exact_hangingmug_td_qauc_sweep_20260828')
files=sorted(root.glob('scores/checkpoint_step_*/td_error_curves.json'), key=lambda p:int(p.parent.name.rsplit('_',1)[1]))
expected={200,300,400,500,600,700,800,900,1000,1100,1185}
if {int(p.parent.name.rsplit('_',1)[1]) for p in files} != expected: raise SystemExit('incomplete sweep')
rows=[]; ref=None
for p in files:
 x=json.loads(p.read_text()); data=x['rows']; step=int(x['checkpoint_step']); keys={(r['category'],r['identity'],r['chunk_index']) for r in data}
 if len(data)!=506 or len(keys)!=506: raise RuntimeError(f'bad row contract {step}')
 if ref is None: ref=keys
 elif keys != ref: raise RuntimeError(f'row identity mismatch {step}')
 td=defaultdict(list); q=defaultdict(list); ids=defaultdict(set)
 for r in data:
  values=[float(r[k]) for k in ('q1','q2','q_min','td_target','td_residual','absolute_td_error')]
  if not all(math.isfinite(v) for v in values): raise RuntimeError(f'nonfinite {step}')
  c=r['category']; ident=str(r['identity']); td[c].append(float(r['absolute_td_error'])); q[c,ident].append(float(r['q_min'])); ids[c].add(ident)
 if {k:len(v) for k,v in ids.items()} != {'clean50':10,'sft_failure':10,'sft_success':10}: raise RuntimeError(f'identity contract {step}')
 meanq={k:statistics.fmean(v) for k,v in q.items()}; succ=[v for (c,_),v in meanq.items() if c=='sft_success']; fail=[v for (c,_),v in meanq.items() if c=='sft_failure']
 auc=sum((s>f)+.5*(s==f) for s in succ for f in fail)/100
 med={c:statistics.median(td[c]) for c in ('clean50','sft_failure','sft_success')}
 means={c:statistics.fmean(td[c]) for c in med}
 rows.append({'checkpoint_step':step,'training_td_mean':means['clean50'],'failure_td_mean':means['sft_failure'],'success_td_mean':means['sft_success'],'td_consistency':statistics.fmean(med.values()),'q_auc':auc,'mean_success_q':statistics.fmean(succ),'mean_failure_q':statistics.fmean(fail),'rows':len(data),'checkpoint':x['checkpoint']})
tds=sorted({r['td_consistency'] for r in rows}); aucs=sorted({r['q_auc'] for r in rows},reverse=True)
for r in rows: r['td_rank']=tds.index(r['td_consistency'])+1; r['q_auc_rank']=aucs.index(r['q_auc'])+1; r['combined_rank_score']=r['td_rank']+r['q_auc_rank']
rows.sort(key=lambda r:(r['combined_rank_score'],r['td_consistency'],-r['q_auc'],r['checkpoint_step']))
for i,r in enumerate(rows,1): r['final_rank']=i
out={'schema_version':1,'diagnostic':'r2_pi06exact_offline_td_qauc','online_evaluation_included':False,'trajectory_contract':{'sft_success':10,'sft_failure':10,'clean50':10,'rows_per_checkpoint':506},'reward_contract':{'nonterminal':-0.001,'success_terminal':0.0,'failure_terminal':-1.0,'gamma':0.95},'q_auc_formula':'pairwise AUC of trajectory-mean min(Q1,Q2): 10 success versus 10 failure RobotWin Hanging Mug SFT trajectories','td_consistency_formula':'mean of category-median absolute TD error for Clean-50, SFT failure, SFT success','rows':rows}
(root/'checkpoint_ranking.json').write_text(json.dumps(out,indent=2)+'\n')
fields=list(rows[0]);
with (root/'checkpoint_ranking.tsv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(rows)
(root/'OFFLINE_TD_QAUC_SWEEP_COMPLETE_NO_ONLINE').write_text(f'checkpoints={len(rows)}/11 rows_each=506 finite=true online=false\n')
print(json.dumps({'status':'PASS','top_step':rows[0]['checkpoint_step']}))
