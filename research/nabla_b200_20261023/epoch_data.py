"""Global optimizer-step ordering remains correct across epochs and worker restarts."""
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import IterableDataset

class EpochBatches(IterableDataset):
 def __init__(self,data,indices,epochs,batch_size,rank,world,start_step=0, rows=None):
  self.data=data;self.indices=Path(indices);self.epochs=epochs;self.batch_size=batch_size
  self.rows=data.rows if rows is None else rows;self.rank=rank;self.world=world;self.start_step=start_step
 def __iter__(self):
  info=torch.utils.data.get_worker_info();worker=info.id if info else 0;workers=info.num_workers if info else 1
  size=self.batch_size*self.world;per_epoch=math.ceil(self.rows/size);last_epoch=None
  for step in range(self.start_step+worker,self.epochs*per_epoch,workers):
   epoch,local=divmod(step,per_epoch)
   if epoch!=last_epoch:
    order=np.load(self.indices/f'epoch_{epoch:02d}.npy',mmap_mode='r');last_epoch=epoch
    if len(order)!=self.rows:raise ValueError('Epoch coverage differs')
   global_ids=order[local*size:(local+1)*size]
   ids=global_ids[self.rank*self.batch_size:(self.rank+1)*self.batch_size];active=len(ids)
   batch=self.data.batch(ids if active else global_ids[:1])
   if not active:batch['atoms'].zero_()
   batch.update(step=step,epoch=epoch,global_atoms=int(self.data.metadata(global_ids,'atoms').sum()),
                global_examples=len(global_ids),examples_seen=epoch*self.rows+min((local+1)*size,self.rows))
   yield batch

def lr_factor(examples,total):
 fraction=min(1.,max(0.,examples/total))
 if fraction<=.1:return fraction/.1
 if fraction<=.9:return 1.
 return .5*(1+math.cos(math.pi*(fraction-.9)/.1))

def bucket_order(indices,lengths,seed,global_batch,pool_batches=64):
    # Start with a global row permutation, bucket only within random pools, then
    # shuffle the batches within each pool to avoid a length curriculum.
    indices=np.asarray(indices);order=indices.copy();rng=np.random.default_rng(seed);rng.shuffle(order)
    pool=global_batch*pool_batches
    for start in range(0,len(order),pool):
        chunk=order[start:start+pool]
        chunk[:]=chunk[np.argsort(lengths[chunk],kind='stable')]
        full=len(chunk)//global_batch
        if full:
            block=chunk[:full*global_batch].reshape(full,global_batch)
            block[:]=block[rng.permutation(full)]
    return order

