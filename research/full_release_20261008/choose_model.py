#!/usr/bin/env python3
"""Frozen validation-only selection rule; write the exact authorized full-run configuration."""
import hashlib,json
from pathlib import Path
import numpy as np
from data import Packed
from prepare import atomic,ROOT
HERE=Path(__file__).resolve().parent;OUT=ROOT/'logs/full_release_20261008'

def main():
 protocol=json.loads((HERE/'protocol.json').read_text());evidence={};runs={}
 for arm in protocol['arms']:
  runs[arm['name']]=[]
  for seed in protocol['pilot_seeds']:
   directory=OUT/'pilots'/f'{arm["name"]}_s{seed}';p=directory/'result.json';r=json.loads(p.read_text());m=json.loads((directory/'manifest.json').read_text())
   if r['status']!='complete' or r['test_evaluated'] or m['test_evaluated'] or r['steps']!=protocol['pilot_steps'] or r['validation']['failures']:raise ValueError('Pilot did not meet completion/validity gates')
   if r['manifest_sha256']!=hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest():raise ValueError('Manifest changed')
   if m['code_hashes']['protocol.json']!=hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest():raise ValueError('Selection protocol changed after training')
   r['errors']=np.load(directory/'validation_rmsd.npy');runs[arm['name']].append(r);evidence[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 benchmark=json.loads((OUT/'benchmark.json').read_text())
 if benchmark['status']!='complete' or benchmark['test_used']:raise ValueError('Missing matched GPU benchmark')
 evidence[str(OUT/'benchmark.json')]=hashlib.sha256((OUT/'benchmark.json').read_bytes()).hexdigest()
 baseline=runs['baseline'];assessments={};eligible=[];compact_ok=False
 for name,rows in runs.items():
  if name=='baseline':continue
  ratios=[r['validation']['source_macro_rmsd']/b['validation']['source_macro_rmsd'] for r,b in zip(rows,baseline)]
  p95=[r['validation']['p95_rmsd']/b['validation']['p95_rmsd'] for r,b in zip(rows,baseline)]
  source_worst=max(r['validation']['per_source'][s]['mean_rmsd']/max(1e-12,b['validation']['per_source'][s]['mean_rmsd']) for r,b in zip(rows,baseline) for s in b['validation']['per_source'])
  geometry_worst=max(max(r['validation'][metric]/max(1e-12,b['validation'][metric]) for metric in ['bond_mae','pair_rmse']) for r,b in zip(rows,baseline))
  speed=benchmark['mean_ms'][name]/benchmark['mean_ms']['baseline']
  safe=max(p95)<=1.05 and source_worst<=1.05 and geometry_worst<=1.05
  quality=safe and max(ratios)<=.98
  if quality:eligible.append(name)
  compact_ok=name=='compact' and safe and max(ratios)<=1.01 and speed<=.8 or compact_ok
  assessments[name]={'macro_rmsd_ratios':ratios,'p95_ratios':p95,'worst_source_ratio':source_worst,'worst_geometry_ratio':geometry_worst,'step_time_ratio':float(speed),'quality_eligible':quality}
 chosen=min(eligible,key=lambda n:np.mean([r['validation']['source_macro_rmsd'] for r in runs[n]])) if eligible else ('compact' if compact_ok else 'baseline')
 result={'status':'frozen','arm':chosen,'configuration':next(a for a in protocol['arms'] if a['name']==chosen),'seed':protocol['full_seed'],'epochs':1,'world_size':8,'global_batch':1024,'test_used':False,'evidence':evidence,'benchmark':benchmark,'assessments':assessments,'rule':protocol['selection'],'limitations':'Two pilot seeds and a fixed validation cohort support an engineering choice, not a novelty claim or proof of long-run superiority.'}
 atomic(OUT/'selection.json',result)
 lines=['# New-release validation-only selection','',f'Chosen configuration: **{chosen}**. Full training: one pass on eight GPUs.','', '| Configuration | Mean source-macro RMSD | Mean p95 RMSD | Mean training ms/step |','|---|---:|---:|---:|']
 for name,rows in runs.items():lines.append(f"| {name} | {np.mean([r['validation']['source_macro_rmsd'] for r in rows]):.6f} | {np.mean([r['validation']['p95_rmsd'] for r in rows]):.6f} | {np.mean([r['ms_per_step'] for r in rows]):.2f} |")
 lines.extend(['',protocol['selection'],'',result['limitations'],'','Speed eligibility uses a separate same-GPU cached-batch benchmark, not cross-node pilot wall times. No test data or test scores were used. Older-data runs were used only to shortlist hypotheses, not as selection evidence.',''])
 (OUT/'SELECTION.md').write_text('\n'.join(lines));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
