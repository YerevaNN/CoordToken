#!/usr/bin/env python3
import json,os,subprocess,sys
from pathlib import Path
from prepare import ROOT
HERE=Path(__file__).resolve().parent;OUT=ROOT/'logs/full_release_20261008'

def main():
 protocol=json.loads((HERE/'protocol.json').read_text());tasks=[(a,s) for a in protocol['arms'] for s in protocol['pilot_seeds']]
 arm,seed=tasks[int(os.environ['SLURM_ARRAY_TASK_ID'])];dest=OUT/'pilots'/f'{arm["name"]}_s{seed}';dest.mkdir(parents=True,exist_ok=True)
 env=os.environ.copy();env.update(RANK='0',LOCAL_RANK='0',WORLD_SIZE='1',COORDTOKEN_GPU_CHECK_OUTPUT=str(dest/'gpu_check.json'))
 subprocess.run([sys.executable,'-u',str(HERE/'gpu_check.py')],env=env,check=True)
 cmd=[sys.executable,'-u',str(HERE/'train_run.py'),'--data',str(OUT/'packed'),'--indices',str(OUT/'indices'),'--output',str(dest),'--arm',arm['name'],'--seed',str(seed),'--resume']
 subprocess.run(cmd,env=env,check=True)
if __name__=='__main__':main()
