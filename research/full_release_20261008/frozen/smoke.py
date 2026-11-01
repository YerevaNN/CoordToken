#!/usr/bin/env python3
"""Semantic checks before consuming experiment GPUs."""
import json,sys,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from model import Config,Tokenizer,ROOT
from data import bucket_order,Batches
from prepare import atomic
sys.path.insert(0,str(ROOT))
import train as original

def main():
 torch.set_num_threads(1);torch.manual_seed(123)
 old=original.MolModel(total_steps=100,d_model=128,n_layers=1,levels=[2]*12).eval()
 new=Tokenizer(Config(width=128,layers=1)).eval()
 new.load_state_dict({k:v for k,v in old.state_dict().items() if k!='util'},strict=True)
 tokens=torch.tensor([[34,0,35,13,34,-1,-1,-1],[35,34,13,34,-1,-1,-1,-1]])
 coords=torch.randn(2,8,3)*tokens.ge(0).unsqueeze(-1);atoms=tokens.ge(0)
 identity=torch.nn.functional.one_hot(tokens.clamp_min(0),original.V).float()*tokens.ge(0).unsqueeze(-1)
 features=torch.cat([identity,coords],-1)
 with torch.no_grad(),patch('torch.nn.attention.sdpa_kernel',lambda *_args,**_kwargs:sdpa_kernel(SDPBackend.MATH)):
  a,ai=old(features);b,bi=new(tokens,coords,atoms)
  assert torch.equal(ai,bi) and torch.allclose(a,b,atol=1e-6,rtol=1e-5)
  assert torch.allclose(new.decode(tokens,bi),b,atol=1e-6,rtol=1e-5)
 for workspace in [0,4]:
  model=Tokenizer(Config(width=128,layers=1,workspace=workspace)).eval()
  with torch.no_grad():
   pred,codes=model(tokens,coords,atoms);decoded=model.decode(tokens,codes)
   assert torch.equal(pred,decoded)
   padded=torch.nn.functional.pad(tokens,(0,8),value=-1);xyz=torch.nn.functional.pad(coords,(0,0,0,8));mask=torch.nn.functional.pad(atoms,(0,8))
   pp,cc=model(padded,xyz,mask)
   assert torch.equal(cc[:,:8][tokens.ge(0)],codes[tokens.ge(0)])
   assert torch.allclose(pp[:,:8][tokens.ge(0)],pred[tokens.ge(0)],atol=1e-5,rtol=1e-5)
  pred,codes=model(tokens,coords,atoms);pred.square().mean().backward()
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
 lengths=np.random.default_rng(1).integers(1,219,10003);ids=np.arange(len(lengths),dtype=np.uint32)
 order=bucket_order(ids,lengths,123,1024)
 assert np.array_equal(np.sort(order),ids) and np.array_equal(order,bucket_order(ids,lengths,123,1024))
 owned=[[] for _ in range(8)]
 for start in range(0,len(order),1024):
  for rank in range(8):owned[rank].extend(order[start+128*rank:min(start+128*(rank+1),start+1024,len(order))].tolist())
 assert sorted(sum(owned,[]))==ids.tolist()
 # Accumulated single-GPU pilots and eight-rank training must optimize the
 # same atom-normalized objective, including an uneven final batch.
 class Toy:
  def metadata(self,ids,field):return 1+np.asarray(ids)%3
  def batch(self,ids):
   ids=np.asarray(ids);n=len(ids);atoms=torch.arange(3)[None,:]<torch.tensor(1+ids%3)[:,None]
   return {'coords':torch.tensor(ids[:,None,None]%7,dtype=torch.float32).expand(n,3,3).clone(),'atoms':atoms,'tokens':torch.ones(n,3,dtype=torch.long)}
 toy=Toy();order=np.arange(1037);reference=torch.nn.Linear(3,3,bias=False)
 for update in [0,1]:
  ids=order[update*1024:(update+1)*1024];allbatch=toy.batch(ids);reference.zero_grad()
  expected=(reference(allbatch['coords']).square()*allbatch['atoms'].unsqueeze(-1)).sum()/(3*allbatch['atoms'].sum());expected.backward();expected_grad=reference.weight.grad.clone()
  for world,accumulation in [(1,8),(8,1)]:
   grads=[]
   for rank in range(world):
    reference.zero_grad()
    for b in Batches(toy,order,128,rank,world,update*accumulation,accumulation):
     if b['step']//accumulation!=update:break
     loss=(reference(b['coords']).square()*b['atoms'].unsqueeze(-1)).sum()*world/(3*b['global_atoms']);loss.backward()
    grads.append(reference.weight.grad.clone())
   assert torch.allclose(torch.stack(grads).mean(0),expected_grad,atol=2e-6,rtol=1e-5)
 r={'status':'passed','checks':['same baseline forward/codes with copied weights','code-only decode equivalence','padding invariance incl workspace','finite gradients for all parameters','global shuffle covers all rows exactly once','eight disjoint rank assignments incl final partial batch','accumulation and DDP loss normalization incl empty ranks'],'test_opened':False}
 atomic(ROOT/'logs/full_release_20261008/smoke.json',r);print(json.dumps(r))
if __name__=='__main__':main()
