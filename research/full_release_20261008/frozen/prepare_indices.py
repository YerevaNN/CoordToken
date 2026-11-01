#!/usr/bin/env python3
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from data import Packed,bucket_order
from prepare import atomic

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 protocol=json.loads(Path(__file__).with_name('protocol.json').read_text());rng=np.random.default_rng(protocol['sample_seed'])
 train=Packed(a.data,'train');val=Packed(a.data,'val')
 lengths=np.lib.format.open_memmap(a.output/'train_lengths.npy',mode='w+',dtype='<u2',shape=(train.rows,))
 for start,m in zip(train.starts,train.maps):lengths[start:start+len(m[3])]=m[3]['length']
 lengths.flush()
 maximum=int(lengths.max())
 # Conservative memory gate; no molecule is silently truncated or dropped.
 if maximum>512:raise ValueError(f'Maximum length {maximum} needs explicit memory/batching review before full launch')
 n=protocol['pilot_steps']*protocol['pilot_batch']*protocol['pilot_accumulation'];ids=rng.choice(train.rows,n,replace=False).astype(np.uint32)
 np.save(a.output/'pilot_train.npy',ids)
 np.save(a.output/'full_order.npy',bucket_order(np.arange(train.rows,dtype=np.uint32),lengths,protocol['full_seed'],protocol['full_global_batch']))
 allids=np.arange(val.rows,dtype=np.int64);ds=val.metadata(allids,'dataset');keys=val.metadata(allids,'key');selected=[];counts={}
 for name,di in val.dsmap.items():
  eligible=np.flatnonzero(ds==di);rng.shuffle(eligible)
  # One randomly encountered conformer per key; missing keys remain row-specific.
  seen=set();chosen=[]
  for idx in eligible:
   key=bytes(keys[idx]) or ('missing',int(idx))
   if key in seen:continue
   seen.add(key);chosen.append(int(idx))
   if len(chosen)>=protocol['validation_molecules_per_source']:break
  selected.extend(chosen);counts[name]=len(chosen)
 selected=np.asarray(selected,dtype=np.uint32);np.save(a.output/'validation.npy',selected)
 manifest=dict(status='complete',train_rows=train.rows,val_rows=val.rows,pilot_rows=len(ids),validation_rows=len(selected),validation_source_counts=counts,max_length=maximum,
  protocol_sha256=hashlib.sha256(Path(__file__).with_name('protocol.json').read_bytes()).hexdigest(),data_manifest_sha256=hashlib.sha256((a.data/'manifest.json').read_bytes()).hexdigest(),test_opened=False)
 for name in ['pilot_train.npy','validation.npy','train_lengths.npy','full_order.npy']:manifest[name+'_sha256']=hashlib.sha256((a.output/name).read_bytes()).hexdigest()
 atomic(a.output/'manifest.json',manifest);print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
