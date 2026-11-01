#!/usr/bin/env python3
import hashlib,json,os,subprocess,sys
from pathlib import Path
from prepare import ROOT
HERE=Path(__file__).resolve().parent;OUT=ROOT/'logs/full_release_20261008'

def main():
 selection=json.loads((OUT/'selection.json').read_text())
 if selection['status']!='frozen' or selection['test_used'] or selection['world_size']!=8 or selection['epochs']!=1:raise ValueError('Invalid full-run authorization/configuration')
 for path,sha in selection['evidence'].items():
  if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:raise ValueError('Selection evidence changed')
 dest=OUT/'full';dest.mkdir(parents=True,exist_ok=True)
 cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=8',str(HERE/'train_run.py'),'--data',str(OUT/'packed'),'--indices',str(OUT/'indices'),'--output',str(dest),'--arm',selection['arm'],'--seed',str(selection['seed']),'--full','--resume']
 if not (dest/'ddp_gate.json').exists():
  subprocess.run(cmd+['--stop-after','100'],check=True)
 gate=json.loads((dest/'ddp_gate.json').read_text())
 if gate['status']!='passed' or gate['world']!=8:raise ValueError('Distributed gate failed')
 subprocess.run(cmd,check=True)
if __name__=='__main__':main()
