#!/usr/bin/env python3
"""Post-selection evaluation on explicitly authorized GEOM-revisited only."""
import csv,hashlib,json,re,sys,time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'frozen'))
from model import Config,Tokenizer
from metrics import evaluate,bond_indices
from prepare import encode,atomic,ROOT
OUT=ROOT/'logs/full_release_20261008/geom_revisited'
SOURCE=Path('/mnt/weka/fgeikyan/fsq/test/geom_revisited.csv')

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

class Dataset:
 def __init__(self,vocab,pattern):
  self.vocab=vocab;self.dsmap={'geom_revisited':0};self.items=[];self.names=[];self.errors=[];self.total=0
  self.atom_vocab=np.array([v.startswith('[') for v in vocab]);mapping={v:i for i,v in enumerate(vocab)};regex=re.compile(pattern)
  with SOURCE.open() as f:
   for i,row in enumerate(csv.DictReader(f)):
    self.total+=1
    try:
     ids,coords,n=encode(row['enriched_text'],mapping,regex);bond_indices(ids,vocab)
     self.items.append((ids,coords));self.names.append(row['name'])
    except Exception as e:self.errors.append({'row':i,'name':row.get('name'),'error':str(e)})
 def metadata(self,indices,field):
  assert field=='dataset'
  return np.zeros(len(indices),dtype=np.int16)
 def batch(self,indices):
  length=(max(len(self.items[int(i)][0]) for i in indices)+7)//8*8
  tokens=np.full((len(indices),length),-1,np.int64);coords=np.zeros((len(indices),length,3),np.float32)
  for j,i in enumerate(indices):
   ids,xyz=self.items[int(i)];tokens[j,:len(ids)]=ids;coords[j,:len(ids)]=xyz
  atoms=(tokens>=0)&self.atom_vocab[tokens.clip(0)]
  return {'tokens':torch.from_numpy(tokens),'coords':torch.from_numpy(coords),'atoms':torch.from_numpy(atoms)}

def main():
 OUT.mkdir(parents=True,exist_ok=True);started=time.monotonic();torch.set_num_threads(4);torch.cuda.set_device(0);torch.set_float32_matmul_precision('high')
 checkpoint=ROOT/'logs/full_release_20261008/full/model.pt';ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
 cfg={k:v for k,v in ckpt['config'].items() if k!='parameters'}
 model=Tokenizer(Config(**cfg));model.load_state_dict(ckpt['model'],strict=True);model.cuda().eval()
 plan=json.loads((ROOT/'logs/full_release_20261008/packed/plan.json').read_text())
 assert ckpt['vocab']==plan['vocab']
 data=Dataset(ckpt['vocab'],plan['pattern']);atomic(OUT/'input_audit.json',{'total':data.total,'parsed':len(data.items),'errors':data.errors})
 print('Parsed',len(data.items),'of',data.total,flush=True)
 if not data.items:raise ValueError('No valid inputs')
 metrics,errors=evaluate(model,data,np.arange(len(data.items)),batch_size=64)
 # The requested report uses means, not the 95th percentile. Per-row values retained.
 metrics.pop('p95_rmsd',None)
 for row in metrics['per_source'].values():row.pop('p95_rmsd',None)
 finite=np.isfinite(errors)
 with (OUT/'per_conformer.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['name','rmsd_angstrom','finite'])
  w.writerows(zip(data.names,errors.tolist(),finite.tolist()))
 result={'status':'complete' if not data.errors and finite.all() else 'complete_with_failures','dataset':str(SOURCE),'dataset_sha256':sha(SOURCE),'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'selection_sha256':sha(ROOT/'logs/full_release_20261008/selection.json'),'evaluation_script_sha256':sha(Path(__file__)),'post_selection_evaluation':True,'used_for_selection':False,'test_evaluated':True,'precision':'bf16 with FP32 coordinate output and quantization; TF32 allowed','metric_definition':'Per-conformer atom RMSD in angstrom, centered inputs, no rigid alignment or symmetry matching; conformers equally weighted.','total_rows':data.total,'parsed_rows':len(data.items),'parse_failures':len(data.errors),'finite_predictions':int(finite.sum()),'valid_percent_of_all_rows':100*int(finite.sum())/data.total,'metrics':metrics,'seconds':time.monotonic()-started}
 atomic(OUT/'result.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
