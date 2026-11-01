#!/usr/bin/env python3
"""Persist terminal workflow status after the dependent full-training job exits."""
import json,subprocess
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'train.py').exists() and (p/'utils.py').exists())
OUT=ROOT/'logs/full_release_20261008'

def main():
 jobs=json.loads((OUT/'jobs.json').read_text());ids=','.join(str(v) for k,v in jobs.items() if isinstance(v,int))
 query=subprocess.run(['sacct','-j',ids,'--format=JobID,State,ExitCode,Elapsed,NodeList','-P'],capture_output=True,text=True,timeout=30)
 result={'jobs':jobs,'slurm':query.stdout,'accounting_error':query.stderr,'test_evaluated':False}
 for name,path in [('selection',OUT/'selection.json'),('full_training',OUT/'full/result.json'),('distributed_check',OUT/'full/ddp_gate.json')]:
  if path.exists():result[name]=json.loads(path.read_text())
 result['status']='complete' if result.get('full_training',{}).get('status')=='complete' else 'stopped_or_failed'
 result['checkpoint']=str(OUT/'full/model.pt');temp=OUT/'STATUS.tmp';temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(OUT/'STATUS.json')
 print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
