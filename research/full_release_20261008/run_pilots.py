#!/usr/bin/env python3
import json,os,subprocess,sys,time
from pathlib import Path
from prepare import atomic,ROOT
HERE=Path(__file__).resolve().parent;OUT=ROOT/'logs/full_release_20261008'

def main():
 protocol=json.loads((HERE/'protocol.json').read_text());visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
 if len(visible)!=8:raise ValueError(f'Expected eight allocated devices, got {visible}')
 jobs=[]
 for arm in protocol['arms']:
  for seed in protocol['pilot_seeds']:
   gpu=visible[len(jobs)];name=f'{arm["name"]}_s{seed}';out=OUT/'pilots'/name;out.mkdir(parents=True,exist_ok=True)
   env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,RANK='0',LOCAL_RANK='0',WORLD_SIZE='1')
   log=(out/'console.log').open('a');cmd=[sys.executable,'-u',str(HERE/'train_run.py'),'--data',str(OUT/'packed'),'--indices',str(OUT/'indices'),'--output',str(out),'--arm',arm['name'],'--seed',str(seed),'--resume']
   process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);jobs.append((name,process,log))
 atomic(OUT/'pilot_jobs.json',{'slurm_job':os.getenv('SLURM_JOB_ID'),'children':[{ 'name':n,'pid':p.pid} for n,p,l in jobs]})
 try:
  while any(p.poll() is None for n,p,l in jobs):
   failed=[n for n,p,l in jobs if p.poll() not in (None,0)]
   if failed:raise RuntimeError('Pilot failed: '+','.join(failed))
   time.sleep(10)
  if any(p.returncode for n,p,l in jobs):raise RuntimeError('Pilot failure')
 finally:
  for n,p,l in jobs:
   if p.poll() is None:p.terminate()
  for n,p,l in jobs:p.wait();l.close()
if __name__=='__main__':main()
