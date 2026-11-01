#!/usr/bin/env python3
import copy,json,os,torch
from pathlib import Path
from model import Config,Tokenizer,ROOT
from train_run import optimizer
from prepare import atomic

def main():
 torch.set_num_threads(1);torch.manual_seed(123);torch.set_float32_matmul_precision('highest')
 assert torch.cuda.is_available()
 model=Tokenizer(Config(width=128,layers=1)).cuda().eval();other=copy.deepcopy(model)
 tokens=torch.randint(0,192,(8,40),device='cuda');coords=torch.randn(8,40,3,device='cuda');atoms=torch.ones_like(tokens,dtype=torch.bool)
 for m,fused in [(model,False),(other,True)]:
  opt=optimizer(m,1e-4,fused=fused);pred,codes=m(tokens,coords,atoms);loss=(pred-coords).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),.1,error_if_nonfinite=True);opt.step()
 diff=max(float((a-b).abs().max()) for a,b in zip(model.parameters(),other.parameters()))
 if diff>1e-6:raise ValueError(f'Fused AdamW mismatch {diff}')
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  pred,codes=model(tokens,coords,atoms);decoded=model.decode(tokens,codes)
  if pred.dtype!=torch.float32 or not torch.equal(pred,decoded):raise ValueError('Mixed-precision code-only decode mismatch')
 atomic(Path(os.environ.get('COORDTOKEN_GPU_CHECK_OUTPUT',str(ROOT/'logs/full_release_20261008/gpu_check.json'))),{'status':'passed','gpu':torch.cuda.get_device_name(),'fused_adamw_parameter_max_difference':diff,'mixed_precision_decode_exact':True,'test_opened':False})
 print('GPU semantic checks passed',flush=True)
if __name__=='__main__':main()
