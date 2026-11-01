#!/usr/bin/env python3
"""Independent random readback of packed numeric records against source CSV."""
import csv,json,re
from pathlib import Path
import numpy as np
from data import Packed
from prepare import ROOT,atomic,META

def main():
 root=ROOT/'logs/full_release_20261008/packed';rng=np.random.default_rng(987654);checked=0
 for split in ['train','val']:
  data=Packed(root,split);source=Path(data.manifest['sources'][split]['path'])
  with source.open('rb') as f:
   header=next(csv.reader([f.readline().decode()]))
   for shard,(receipt,m) in enumerate(zip(data.shards,data.maps)):
    n=receipt['valid'];p=Path(receipt['task']['output']);t=receipt['tokens']
    for name,size in [('tokens',t*2),('coords',t*12),('offsets',n*8),('meta',n*META.itemsize)]:
     if (p/(name+'.bin')).stat().st_size!=size:raise ValueError('Packed size mismatch')
    for row in rng.choice(n,min(16,n),replace=False):
     meta=m[3][row];f.seek(int(meta['source_offset']));record=dict(zip(header,next(csv.reader([f.readline().decode()],strict=True))))
     # Independently parse brackets/coordinate blocks and interleaved syntax.
     expected=[];xyz=[];atom=[]
     pieces=re.split(r'(\[[^\]]+\]<[^>]+>)',record['enriched_text'])
     for piece in pieces:
      if not piece:continue
      if piece.startswith('['):
       desc,coord=piece.split('<',1);expected.append(desc);xyz.append([float(v) for v in coord[:-1].split(',')]);atom.append(len(expected)-1)
      else:
       syntax=re.findall(r'%\d{2}|.',piece);expected.extend(syntax);xyz.extend([[0.,0.,0.]]*len(syntax))
     xyz=np.asarray(xyz,np.float32);xyz[atom]-=xyz[atom].mean(0)
     off=int(m[2][row]);length=int(meta['length']);got=m[0][off:off+length]
     if [data.vocab[i] for i in got]!=expected or not np.array_equal(m[1][off:off+length],xyz):raise ValueError('Independent packed/source mismatch')
     if int(meta['atoms'])!=len(atom) or bytes(meta['key']).decode()!=record['inchikey']:raise ValueError('Metadata mismatch')
     checked+=1
 atomic(ROOT/'logs/full_release_20261008/cache_readback.json',{'status':'passed','records':checked,'all_shard_sizes_checked':True,'test_opened':False});print('Independent cache readback passed',checked,flush=True)
if __name__=='__main__':main()
