#!/usr/bin/env python3
"""Compare model compute on identical cached training batches on one GPU."""
import gc,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from data import Packed
from model import Config,Tokenizer
from train_run import optimizer
from prepare import ROOT,atomic
HERE=Path(__file__).resolve().parent
OUT=ROOT/'logs/full_release_20261008'

def main():
 torch.set_num_threads(1);torch.cuda.set_device(0);torch.set_float32_matmul_precision('high')
 protocol=json.loads((HERE/'protocol.json').read_text());data=Packed(OUT/'packed','train')
 # Identical length-bucketed real training batches, transferred once, outside timing.
 order=np.load(OUT/'pilots/baseline_s123/order.npy',mmap_mode='r')
 starts=np.random.default_rng(20261008).choice(len(order)//1024,24,replace=False)*1024
 batches=[{k:v.cuda() for k,v in data.batch(order[s:s+128]).items()} for s in starts]
 timings={a['name']:[] for a in protocol['arms']}
 for repeat in range(2):
  arms=protocol['arms'] if repeat==0 else list(reversed(protocol['arms']))
  for arm in arms:
   torch.manual_seed(123);model=Tokenizer(Config(vocab_size=len(data.vocab),width=arm['width'],layers=arm['layers'],workspace=arm['workspace'])).cuda();opt=optimizer(model,arm['lr'])
   def step(batch):
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
     pred,_=model(**batch);loss=((pred-batch['coords']).square()*batch['atoms'].unsqueeze(-1)).sum()/(3*batch['atoms'].sum())
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.1);opt.step()
   for batch in batches[:5]:step(batch)
   torch.cuda.synchronize();start=time.monotonic()
   for batch in batches:step(batch)
   torch.cuda.synchronize();timings[arm['name']].append(1000*(time.monotonic()-start)/len(batches))
   del opt,model;gc.collect();torch.cuda.empty_cache()
 result={'status':'complete','test_used':False,'gpu':torch.cuda.get_device_name(),'scope':'Single-GPU training compute, identical GPU-resident train batches; excludes data I/O and eight-GPU communication. Two reversed-order repetitions.','microbatch':128,'timings_ms':timings,'mean_ms':{k:float(np.mean(v)) for k,v in timings.items()},'batch_starts':starts.tolist(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 atomic(OUT/'benchmark.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
