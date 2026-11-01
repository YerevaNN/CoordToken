"""Short-warmup full-original-nabla run and separate decay branches. Training only."""
import argparse,json,math,os,random,shutil,time,datetime
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from common import OUT,DATA,INDICES,PHASE,atomic,sha,verify_sources,verify_training
from health import LossGuard
from schedule import samples_at_step,factor,endpoint_plan,WARMUP_EPOCHS
from metrics import evaluate
from epoch_data import EpochBatches
from nabla_data import OriginalNabla
from validation_data import SharedValidation
from model import Config,Tokenizer

def save(path,value):
 tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)

def main():
 p=argparse.ArgumentParser();p.add_argument('--stop-after',type=int,default=0);p.add_argument('--resume',action='store_true');p.add_argument('--endpoint',default='');p.add_argument('--stop-step',type=int,required=True);a=p.parse_args()
 verify_sources();verify_training();case=json.loads((OUT/'case.json').read_text());indices=INDICES
 if case['warmup_epochs']!=WARMUP_EPOCHS:raise ValueError('Warmup configuration differs')
 im=json.loads((indices/'manifest.json').read_text())
 if im['status']!='complete' or im['rows']!=case['rows'] or im['epochs']<case['epochs']:raise ValueError('Index configuration differs')
 if a.endpoint and a.endpoint not in case['endpoints']:raise ValueError('Unregistered endpoint')
 world=int(os.environ['WORLD_SIZE']);rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
 if world!=8:raise ValueError('Eight GPUs required')
 torch.cuda.set_device(local);torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=30))
 seed=case['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 train=OriginalNabla(DATA,'train');rows=case['rows'];global_batch=case['global_batch'];batch_size=global_batch//world
 if batch_size*world!=global_batch:raise ValueError('Invalid global batch')
 model=Tokenizer(Config(vocab_size=len(train.vocab),width=case['width'],layers=case['layers'],dropout=case['dropout'],bits=case['bits'],registers=case['registers'])).cuda()
 decay=[];other=[]
 for name,param in model.named_parameters():
  (other if param.ndim==1 or name.endswith('.bias') or 'norm' in name.lower() else decay).append(param)
 opt=torch.optim.AdamW([{'params':decay,'weight_decay':case['weight_decay']},{'params':other,'weight_decay':0.}],lr=case['lr'],fused=True,betas=(.9,case['beta2']))
 per_epoch=math.ceil(rows/global_batch);plan=endpoint_plan(rows,global_batch);steps=plan['epoch_22']['end_step'];total_examples=samples_at_step(steps,rows,global_batch)
 stage_end=plan[a.endpoint]['end_step'] if a.endpoint else a.stop_step
 if a.endpoint and a.stop_step!=stage_end:raise ValueError('Branch endpoint mismatch')
 if stage_end>plan['epoch_22']['start_step'] and not a.endpoint:raise ValueError('Main run exceeds final branch start')
 branch_starts={p['start_step']:name for name,p in plan.items()}
 dest=OUT/(a.endpoint if a.endpoint else 'main');dest.mkdir(exist_ok=True)
 manifest={'planned_endpoints':plan,'case':case,'config':model.architecture(),'world':world,'rank_batch':batch_size,'global_batch':global_batch,'epochs':case['epochs'],
  'rows_per_epoch':rows,'total_examples':total_examples,'total_steps':steps,'seed':seed,'data_manifest_sha256':sha(DATA/'manifest.json'),
  'validation_manifest_sha256':sha(OUT/'validation/manifest.json'),'indices_manifest_sha256':sha(indices/'manifest.json'),'source_freeze_sha256':sha(Path(__file__).with_name('freeze.json')),
  'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'test_used_for_selection':False,
  'scheduler':'0.3 epoch sample-based warmup once; main constant LR; final10% of updates for each independent sample-based cosine branch',
  'endpoint':a.endpoint or None,
  'selection':'Settings selected by new one-epoch shared-validation tuning and dropout followups; fixed epoch-22 endpoint official tests; no test selection','phase':PHASE,
  'precision':'bf16 linears; FP32 math SDPA intermediates and XYZ/quantization; TF32 disabled', 'attention_backend':'math','optimizer':'selective-decay fused AdamW','from_scratch':not bool(a.endpoint)}
 if (dest/'manifest.json').exists() and json.loads((dest/'manifest.json').read_text())!=manifest:raise ValueError('Run configuration changed')
 if rank==0:atomic(dest/'manifest.json',manifest)
 dist.barrier();start=0;previous_seconds=0.;resume_rng=None;guard=LossGuard();last_probe=None;gradient_recent=[]
 resume_path=dest/'last.pt'
 if a.endpoint and not resume_path.exists():resume_path=OUT/'main'/f'branch_start_{a.endpoint}.pt'
 if resume_path.exists():
  if not a.resume:raise ValueError('Resume flag required')
  state=torch.load(resume_path,map_location='cpu',weights_only=False)
  expected=dict(manifest)
  if resume_path.parent==OUT/'main' and a.endpoint:
   expected.update(endpoint=None,from_scratch=True)
   if state['step']!=plan[a.endpoint]['start_step']:raise ValueError('Wrong branch start')
  if state['manifest']!=expected:raise ValueError('Checkpoint provenance differs')
  model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);start=state['step'];previous_seconds=state['training_seconds'];resume_rng=state['rng'][rank];guard=LossGuard.from_state_dict(state['health_guard']);last_probe=state.get('train_probe');gradient_recent=state.get('gradient_recent',[]);del state
 elif a.endpoint:raise ValueError('Missing branch start')
 if start>stage_end:raise ValueError('Checkpoint is beyond requested stage')
 wrapped=DDP(model,device_ids=[local],broadcast_buffers=False,gradient_as_bucket_view=True,bucket_cap_mb=64)
 if resume_rng is not None:
  torch.set_rng_state(resume_rng['cpu']);torch.cuda.set_rng_state(resume_rng['cuda'])
 else:torch.manual_seed(seed+rank);torch.cuda.manual_seed_all(seed+rank)
 loader=DataLoader(EpochBatches(train,indices,case['epochs'],batch_size,rank,world,start,rows=rows),batch_size=None,num_workers=2,pin_memory=True,persistent_workers=True,prefetch_factor=4,generator=torch.Generator().manual_seed(seed+17))
 log=(dest/f'metrics_rank{rank}.jsonl').open('a',buffering=1);started=time.monotonic();excluded=0.;last_step=start;examples_seen=(start//per_epoch)*rows+min((start%per_epoch)*global_batch,rows)
 probe_ids=np.load(OUT/'train_probe_ids.npy')
 validation_data=SharedValidation(OUT/'validation',train.vocab)
 validation_ids=np.arange(validation_data.rows)
 def validation_probe(step):
  if rank==0:
   v,errors=evaluate(model,validation_data,validation_ids,batch_size=64)
   if v['failures']:raise FloatingPointError('Validation failure')
   atomic(dest/f'validation_{step}.json',{'step':step,'training_examples_seen':samples_at_step(step,rows,global_batch),**v})
   atomic(dest/'validation.json',{'step':step,'training_examples_seen':samples_at_step(step,rows,global_batch),**v})
   np.save(dest/'validation_rmsd.npy',errors)
  dist.barrier()
 def probe(step):
  result=[None]
  if rank==0:
   with torch.no_grad():metrics,_=evaluate(model,train,probe_ids,batch_size=64)
   if metrics['failures']:raise FloatingPointError('Training probe failure')
   result[0]={'step':step,'rows':len(probe_ids),'mean_rmsd':metrics['mean_rmsd'],'mean_geometry_bits':metrics['mean_geometry_bits'],'occupied_codes':metrics['occupied_codes'],'training_only':True}
   with (dest/'train_probe.jsonl').open('a') as f:f.write(json.dumps(result[0])+'\n')
  dist.broadcast_object_list(result,src=0)
  return result[0]
 def checkpoint(step,seconds):
  states=[None]*world;dist.all_gather_object(states,{'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()})
  if rank==0:
   if shutil.disk_usage(dest).free < 3_000_000_000:raise RuntimeError('Insufficient checkpoint headroom; retaining previous checkpoint')
   save(dest/'last.pt',{'model':model.state_dict(),'optimizer':opt.state_dict(),'config':manifest['config'],'vocab':train.vocab,'step':step,'rng':states,'training_seconds':seconds,'manifest':manifest,'health_guard':guard.state_dict(),'train_probe':last_probe,'gradient_recent':gradient_recent})
   retained=dest/f'recovery_{step:09d}.pt'
   if not retained.exists():os.link(dest/'last.pt',retained)
   # Retain current and previous recovery states; endpoint optimizer states are retained separately.
   for old in sorted(dest.glob('recovery_*.pt'))[:-2]:old.unlink()
   if not a.endpoint and step in branch_starts:
    branch=dest/f'branch_start_{branch_starts[step]}.pt'
    if not branch.exists():os.link(dest/'last.pt',branch)
    atomic(branch.with_suffix('.json'),{'step':step,'sha256':sha(branch),'guard':guard.state_dict(),'train_probe':last_probe})
  dist.barrier()
 try:
  for batch in loader:
   if last_step>=stage_end or (a.stop_after and last_step>=a.stop_after):break
   step=int(batch.pop('step'));epoch=int(batch.pop('epoch'));denom=int(batch.pop('global_atoms'));batch.pop('global_examples');examples_seen=int(batch.pop('examples_seen'))
   if step!=last_step:raise RuntimeError('DataLoader changed optimizer-step order')
   batch={k:v.cuda(non_blocking=True) for k,v in batch.items()}
   for group in opt.param_groups:group['lr']=case['lr']*factor(examples_seen,rows,global_batch,a.endpoint or None)
   wrapped.train();opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):
    pred,_=wrapped(**batch);loss=((pred-batch['coords']).square()*batch['atoms'].unsqueeze(-1)).sum()*world/(3*max(1,denom))
   finite=torch.isfinite(loss.detach()).to(torch.int32);dist.all_reduce(finite,op=dist.ReduceOp.MIN)
   if not bool(finite):raise FloatingPointError('Nonfinite loss before optimizer step')
   loss.backward();norm=torch.nn.utils.clip_grad_norm_(wrapped.parameters(),case['clip'],error_if_nonfinite=True);opt.step();last_step=step+1
   if last_step%100==0 or last_step%per_epoch==0 or last_step==stage_end:
    logged=loss.detach().clone();dist.all_reduce(logged);logged/=world
    if not bool(torch.isfinite(logged)&torch.isfinite(norm)):raise FloatingPointError('Nonfinite loss or gradient')
    event={'step':last_step,'epoch':epoch+1,'examples_seen':examples_seen,'loss':float(logged),'gradient_norm':float(norm),'lr':opt.param_groups[0]['lr'],'training_seconds':previous_seconds+time.monotonic()-started-excluded,'mean_tokens':batch['tokens'].shape[1],'max_abs_coordinate':float(batch['coords'].abs().max())}
    log.write(json.dumps(event)+'\n')
    health=guard.observe(last_step,float(logged),float(norm));event['health']=health
    gradient_recent=(gradient_recent+[float(norm)])[-10:]
    event['gradient_median']=float(np.median(gradient_recent))
    if len(gradient_recent)==10 and event['gradient_median']>1000:
     health={'healthy':False,'reason':'Persistent pre-clipping gradient median >1000'};event['health']=health
    if rank==0:atomic(dest/'progress.json',{'status':'training' if health['healthy'] else 'stopped_training_instability','total_steps':stage_end,**event});print(json.dumps(event),flush=True)
    if not health['healthy']:raise RuntimeError('Automatic safety stop: '+str(health))
   if last_step in {100,200}:
    checks=torch.stack([sum(p.detach().float().sum() for p in model.parameters()),sum(p.detach().float().square().sum() for p in model.parameters())]);all_checks=[torch.empty_like(checks) for _ in range(world)];dist.all_gather(all_checks,checks)
    if not all(torch.equal(all_checks[0],x) for x in all_checks):raise RuntimeError('DDP parameters diverged')
    if rank==0:atomic(dest/f'ddp_gate_{last_step}.json',{'status':'passed','world':world,'step':last_step,'checks':checks.tolist(),'resumed_at_step':start})
   stop=a.stop_after and last_step>=a.stop_after
   if last_step%5000==0 or last_step%per_epoch==0 or last_step==stage_end or (not a.endpoint and last_step in branch_starts) or stop:
    torch.cuda.synchronize();t=time.monotonic()
    if last_step>=1000:last_probe=probe(last_step)
    if validation_ids is not None and (last_step%per_epoch==0 or last_step==stage_end):validation_probe(last_step)
    checkpoint(last_step,previous_seconds+t-started-excluded);excluded+=time.monotonic()-t
   if stop:break
  if last_step<stage_end and not a.stop_after:raise RuntimeError('Training iterator ended early')
  torch.cuda.synchronize();seconds=previous_seconds+time.monotonic()-started-excluded
  if rank==0:
   result={'status':'complete' if last_step==stage_end else 'stopped','steps':last_step,'total_steps':stage_end,'examples_seen':examples_seen,'total_examples':samples_at_step(stage_end,rows,global_batch),'training_seconds':seconds,'test_used_for_selection':False,'manifest_sha256':sha(dest/'manifest.json')}
   if last_step==stage_end:
    if examples_seen!=samples_at_step(stage_end,rows,global_batch):raise RuntimeError('Stage sample count differs')
    if a.endpoint and opt.param_groups[0]['lr']!=0.:raise RuntimeError('Branch did not decay to zero')
    if not guard.recent or float(np.median(guard.recent))>=.1 or last_probe is None or last_probe['mean_rmsd']>=.5:raise RuntimeError('Stage health gate failed')
    result['health_gate']='passed';result['train_probe']=last_probe;result['lr']=opt.param_groups[0]['lr'];result['endpoint']=a.endpoint or None;result['gradient_median']=float(np.median(gradient_recent));result['validation']=json.loads((dest/'validation.json').read_text()) if validation_ids is not None else None
    save(dest/'model.pt',{'model':model.state_dict(),'config':manifest['config'],'vocab':train.vocab,'manifest':manifest});result['checkpoint_sha256']=sha(dest/'model.pt')
   atomic(dest/'result.json',result);atomic(dest/'progress.json',result)
   if last_step==stage_end:atomic(dest/f'stage_{stage_end:09d}.json',result)
 except BaseException as exc:
  if rank==0:atomic(dest/'failure.json',{'step':last_step,'error':repr(exc),'test_used_for_selection':False})
  raise
 finally:
  log.close();dist.destroy_process_group()
if __name__=='__main__':main()
