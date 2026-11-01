#!/usr/bin/env python3
"""Matched pilot or one-pass DDP training; only the registered train/val cache is read."""
import argparse,hashlib,json,math,os,random,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from data import Packed,Batches,bucket_order
from model import Config,Tokenizer
from metrics import evaluate
from prepare import atomic

def save_torch(path,obj):
    temp=path.with_suffix('.tmp');torch.save(obj,temp);temp.replace(path)

def optimizer(model,lr,fused=True):
    decay=[];other=[]
    for name,p in model.named_parameters():
        (other if p.ndim==1 or name.endswith('.bias') or 'norm' in name.lower() else decay).append(p)
    return torch.optim.AdamW([{'params':decay,'weight_decay':.01},{'params':other,'weight_decay':0.}],lr=lr,fused=fused)

def rng_state():return {'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--indices',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--full',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int,default=0)
    a=p.parse_args();protocol=json.loads(Path(__file__).with_name('protocol.json').read_text());arm=next(x for x in protocol['arms'] if x['name']==a.arm)
    world=int(os.getenv('WORLD_SIZE','1'));rank=int(os.getenv('RANK','0'));local=int(os.getenv('LOCAL_RANK','0'))
    if a.full and world!=8:raise ValueError('Full training requires the authorized eight GPUs')
    torch.cuda.set_device(local);torch.set_num_threads(1);torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    if world>1:dist.init_process_group('nccl')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    train=Packed(a.data,'train');val=Packed(a.data,'val');index_manifest=json.loads((a.indices/'manifest.json').read_text())
    if index_manifest['status']!='complete' or index_manifest['data_manifest_sha256']!=hashlib.sha256((a.data/'manifest.json').read_bytes()).hexdigest():raise ValueError('Index/data mismatch')
    cfg=Config(vocab_size=len(train.vocab),width=arm['width'],layers=arm['layers'],workspace=arm['workspace'])
    model=Tokenizer(cfg).cuda();opt=optimizer(model,arm['lr']);batch_size=128;accumulation=1 if a.full else protocol['pilot_accumulation'];global_batch=batch_size*world*accumulation
    lengths=np.load(a.indices/'train_lengths.npy',mmap_mode='r');selected=np.arange(train.rows,dtype=np.uint32) if a.full else np.load(a.indices/'pilot_train.npy')
    order_path=a.output/'order.npy';a.output.mkdir(parents=True,exist_ok=True)
    manifest={'arm':arm,'config':model.architecture(),'seed':a.seed,'world':world,'batch_size':batch_size,'accumulation':accumulation,'global_batch':global_batch,'full':a.full,'examples':len(selected),
              'data_manifest_sha256':index_manifest['data_manifest_sha256'],'index_manifest_sha256':hashlib.sha256((a.indices/'manifest.json').read_bytes()).hexdigest(),
              'code_hashes':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ['train_run.py','data.py','model.py','quantizer.py','metrics.py','protocol.json']},
              'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'test_evaluated':False,'optimizer':'selective-decay AdamW fused','precision':'bf16 + FP32 XYZ/quantization; TF32 allowed','epochs':1,'validation_selection':'final pilot step only; one fixed full run'}
    if (a.output/'manifest.json').exists() and json.loads((a.output/'manifest.json').read_text())!=manifest:raise ValueError('Existing run configuration differs')
    if rank==0:
        atomic(a.output/'manifest.json',manifest)
        if not order_path.exists():
            if a.full:
                if a.seed!=protocol['full_seed']:raise ValueError('Full shuffle seed mismatch')
                order_path.symlink_to((a.indices/'full_order.npy').resolve())
            else:np.save(order_path,bucket_order(selected,lengths,a.seed,global_batch))
    if world>1:dist.barrier()
    order=np.load(order_path,mmap_mode='r');steps=math.ceil(len(order)/global_batch);start=0;previous_seconds=0.
    if a.resume and (a.output/'last.pt').exists():
        ckpt=torch.load(a.output/'last.pt',map_location='cpu',weights_only=False);model.load_state_dict(ckpt['model']);opt.load_state_dict(ckpt['optimizer']);start=ckpt['step'];previous_seconds=ckpt['training_seconds']
        state=ckpt['rng'][rank];torch.set_rng_state(state['cpu']);torch.cuda.set_rng_state(state['cuda']);del ckpt
    elif (a.output/'last.pt').exists():raise ValueError('Existing checkpoint requires --resume')
    wrapped=DDP(model,device_ids=[local],broadcast_buffers=False,gradient_as_bucket_view=True,bucket_cap_mb=64) if world>1 else model
    if start==0:
        torch.manual_seed(a.seed+rank);torch.cuda.manual_seed_all(a.seed+rank)
    loader=DataLoader(Batches(train,order,batch_size,rank,world,start*accumulation,accumulation),batch_size=None,num_workers=2,pin_memory=True,persistent_workers=True,prefetch_factor=4,generator=torch.Generator().manual_seed(a.seed+17))
    validation=np.load(a.indices/'validation.npy');metrics=(a.output/f'metrics_rank{rank}.jsonl').open('a',buffering=1)
    started=time.monotonic();excluded=0.;last_step=start;latest_eval=None
    def checkpoint(step,seconds):
        states=[None]*world
        if world>1:dist.all_gather_object(states,rng_state())
        else:states=[rng_state()]
        if rank==0:save_torch(a.output/'last.pt',{'model':model.state_dict(),'optimizer':opt.state_dict(),'config':manifest['config'],'vocab':train.vocab,'step':step,'rng':states,'training_seconds':seconds,'manifest':manifest})
        if world>1:dist.barrier()
    try:
        for batch in loader:
            microstep=int(batch.pop('step'));step=microstep//accumulation;global_atoms=int(batch.pop('global_atoms'));examples=int(batch.pop('global_examples'))
            batch={k:v.cuda(non_blocking=True) for k,v in batch.items()}
            schedule_steps=math.ceil(train.rows/protocol['full_global_batch']);warmup=1000;decay=int(.9*schedule_steps)
            factor=(step+1)/warmup if step<warmup else (1. if step<decay else .5*(1+math.cos(math.pi*(step-decay+1)/max(1,schedule_steps-decay))))
            for group in opt.param_groups:group['lr']=arm['lr']*factor
            wrapped.train()
            if microstep%accumulation==0:
                opt.zero_grad(set_to_none=True);step_loss=0
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred,_=wrapped(**batch)
                # DDP averages gradients. CPU metadata supplies the global atom
                # denominator exactly, avoiding a per-step count collective.
                loss=((pred-batch['coords']).square()*batch['atoms'].unsqueeze(-1)).sum()*world/(3*max(1,global_atoms))
            loss.backward();step_loss=step_loss+loss.detach()
            if (microstep+1)%accumulation and microstep+1<math.ceil(len(order)/(batch_size*world)):continue
            norm=torch.nn.utils.clip_grad_norm_(wrapped.parameters(),.1);opt.step();last_step=step+1
            if last_step%100==0 or last_step==steps:
                if not bool(torch.isfinite(step_loss)&torch.isfinite(norm)):raise FloatingPointError('Nonfinite loss/gradient')
                logged=step_loss.clone()
                if world>1:dist.all_reduce(logged);logged/=world
                event={'step':last_step,'loss':float(logged),'gradient_norm':float(norm),'lr':opt.param_groups[0]['lr'],'training_seconds':previous_seconds+time.monotonic()-started-excluded,'mean_tokens':batch['tokens'].shape[1]}
                metrics.write(json.dumps(event)+'\n')
                if rank==0:atomic(a.output/'progress.json',{'status':'training','total_steps':steps,**event});print(json.dumps(event),flush=True)
            # Verify cross-rank parameter identity early; fail before a long run.
            if a.full and last_step==100:
                checks=torch.stack([sum(p.detach().float().sum() for p in model.parameters()),sum(p.detach().float().square().sum() for p in model.parameters())]);gathered=[torch.empty_like(checks) for _ in range(world)];dist.all_gather(gathered,checks)
                if not all(torch.equal(gathered[0],x) for x in gathered):raise RuntimeError('DDP parameter checksums diverged')
                if rank==0:atomic(a.output/'ddp_gate.json',{'status':'passed','world':world,'step':last_step,'seconds':time.monotonic()-started,'checks':checks.tolist()})
            evaluate_now=(last_step==steps or (not a.full and last_step in [steps//4,steps//2]))
            checkpoint_now=last_step%(5000 if a.full else max(1,steps//2))==0 or last_step==steps or (a.stop_after and last_step>=a.stop_after)
            if evaluate_now:
                torch.cuda.synchronize();t=time.monotonic()
                # A single fixed validation cohort; other ranks wait only here.
                if rank==0:
                    latest_eval,errors=evaluate(model,val,validation)
                    if latest_eval['failures']:raise FloatingPointError('Nonfinite validation outputs')
                    metrics.write(json.dumps({'step':last_step,'validation':latest_eval})+'\n');np.save(a.output/'validation_rmsd.npy',errors)
                    atomic(a.output/'validation.json',{'step':last_step,**latest_eval})
                if world>1:dist.barrier()
                excluded+=time.monotonic()-t
            if checkpoint_now:
                torch.cuda.synchronize();t=time.monotonic();seconds=previous_seconds+t-started-excluded;checkpoint(last_step,seconds);excluded+=time.monotonic()-t
            if a.stop_after and last_step>=a.stop_after:break
        torch.cuda.synchronize();elapsed=previous_seconds+time.monotonic()-started-excluded
        if rank==0:
            result={'status':'complete' if last_step==steps else 'stopped','arm':a.arm,'seed':a.seed,'steps':last_step,'total_steps':steps,'training_seconds':elapsed,'ms_per_step':1000*elapsed/max(1,last_step),'validation':latest_eval,'test_evaluated':False,'manifest_sha256':hashlib.sha256((a.output/'manifest.json').read_bytes()).hexdigest(),'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
            atomic(a.output/'result.json',result);atomic(a.output/'progress.json',result)
            save_torch(a.output/'model.pt',{'model':model.state_dict(),'config':manifest['config'],'vocab':train.vocab,'manifest':manifest})
    finally:
        metrics.close()
        if world>1:dist.destroy_process_group()
if __name__=='__main__':main()
