"""All original test records, three separate regimes, frozen final checkpoint."""
import argparse,hashlib,json,time,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import numpy as np
import torch
from common import DATA,BASE,OUT as RUN_OUT,verify_training,ROOT,sha,atomic,verify_sources
TEST_ROOT=ROOT/"logs/paper_runs_20261013/nabla_final"
from model import Config,Tokenizer
from schedule import endpoint_plan
from metrics import evaluate
from nabla_data import OriginalNabla

class TestData(OriginalNabla):
 def __init__(self,root,vocab,label):
  self.root=root;self.manifest=json.loads((root/'manifest.json').read_text());m=self.manifest
  if m['status']!='complete' or not m['all_original_records_retained'] or m['provenance']['source_split']!=label:raise ValueError('Wrong test source')
  if m['vocab']!=vocab:raise ValueError('Vocabulary mismatch')
  for name,receipt in m['arrays'].items():
   if sha(root/(name+'.npy'))!=receipt['sha256']:raise ValueError('Test cache changed')
  self.rows=m['rows'];self.vocab=vocab;self.dsmap={'nablaDFT':0};self.datasets=['nablaDFT']
  for name in ['tokens','coords','atoms','keys']:setattr(self,name,np.load(root/(name+'.npy'),mmap_mode='r'))

def main():
 p=argparse.ArgumentParser();p.add_argument('--endpoint',default='epoch_22');args=p.parse_args()
 if args.endpoint != 'epoch_22':raise ValueError('Unknown predeclared endpoint')
 freeze=json.loads((HERE/'freeze.json').read_text())
 for name,digest in freeze['files'].items():
  if sha(HERE/name)!=digest:raise ValueError('Evaluation source changed')
 verify_sources();verify_training();OUT=RUN_OUT/args.endpoint;result=json.loads((OUT/'result.json').read_text())
 if result['status']!='complete' or result.get('health_gate')!='passed' or result['lr']!=0 or result['steps']!=endpoint_plan(json.loads((RUN_OUT/'case.json').read_text())['rows'],json.loads((RUN_OUT/'case.json').read_text())['global_batch'])['epoch_22']['end_step'] or result['examples_seen']!=endpoint_plan(json.loads((RUN_OUT/'case.json').read_text())['rows'],json.loads((RUN_OUT/'case.json').read_text())['global_batch'])['epoch_22']['end_examples']:raise ValueError('Final refit incomplete')
 checkpoint=OUT/'model.pt'
 if sha(checkpoint)!=result['checkpoint_sha256']:raise ValueError('Final checkpoint changed')
 if json.loads((TEST_ROOT/'test_preparation.json').read_text())['status']!='complete':raise ValueError('Test preparation incomplete')
 torch.set_num_threads(4);torch.cuda.set_device(0);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 ck=torch.load(checkpoint,map_location='cpu',weights_only=False);config={k:v for k,v in ck['config'].items() if k!='parameters'}
 model=Tokenizer(Config(**config));model.load_state_dict(ck['model'],strict=True);model.cuda().eval()
 selected=np.load(RUN_OUT/'training_ids.npy',mmap_mode='r');train_ids=np.load(DATA/'sample_ids.npy',mmap_mode='r')[selected];train_pairs=np.sort((train_ids[:,0]<<32)|train_ids[:,1]);train_keys=np.unique(np.load(DATA/'keys.npy',mmap_mode='r')[selected])
 regimes={'conformations':'dataset_test_conformations_full','structures':'dataset_test_structures','scaffolds':'dataset_test_scaffolds'}
 dest=OUT/'evaluation';dest.mkdir(exist_ok=True)
 if (dest/'result.json').exists():raise ValueError('Completed evaluation already exists')
 groups={};started=time.monotonic()
 for regime,label in regimes.items():
  data=TestData(TEST_ROOT/'test_cache'/regime,ck['vocab'],label);ids=np.load(data.root/'sample_ids.npy',mmap_mode='r');pairs=(ids[:,0]<<32)|ids[:,1]
  exact=int(np.isin(pairs,train_pairs,assume_unique=True).sum())
  if exact:raise ValueError('Official train/test conformer IDs intersect')
  overlap=int(np.isin(data.keys,train_keys).sum())
  # Official conformer regime intentionally shares molecules. Never filter it.
  sums={k:0. for k in ['mean_rmsd','bond_mae','pair_rmse','mean_geometry_bits']};count=0
  errors=np.lib.format.open_memmap(dest/(regime+'_rmsd.npy'),mode='w+',dtype=np.float64,shape=(data.rows,))
  for start in range(0,data.rows,8192):
   selected=np.arange(start,min(start+8192,data.rows));m,e=evaluate(model,data,selected,batch_size=64)
   if m['failures'] or not all(np.isfinite(m[k]) for k in sums):raise FloatingPointError('Nonfinite test results')
   errors[selected]=e;count+=len(selected)
   for k in sums:sums[k]+=m[k]*len(selected)
   atomic(dest/'progress.json',{'status':'evaluating','regime':regime,'rows':count,'total_rows':data.rows,'seconds':time.monotonic()-started})
  errors.flush();groups[regime]={'rows':count,**{k:v/count for k,v in sums.items()},'train_exact_conformer_id_overlap':exact,'train_full_inchikey_overlap_rows':overlap,'failures':0,'data_manifest_sha256':sha(data.root/'manifest.json')}
  atomic(dest/(regime+'.json'),groups[regime])
 assert all(m['failures']==0 and m['rows']==json.loads((TEST_ROOT/'test_preparation.json').read_text())['rows_by_regime'][regime] for regime,m in groups.items())
 final={'training_policy':'Original training minus mixed-corpus nabla validation identities; no other filtering','endpoint':args.endpoint,'optimizer_updates':result['steps'],'endpoint_epochs':22,'status':'complete','checkpoint_sha256':result['checkpoint_sha256'],'per_regime':groups,'test_used_for_selection':False,'evaluator_sha256':sha(Path(__file__)),'per_regime_errors_sha256':{k:sha(dest/(k+'_rmsd.npy')) for k in groups},'metric':'Mean per-conformer heavy-atom RMSD in angstroms, centered inputs, no alignment/symmetry matching; bf16 linears, math attention, FP32 output; TF32 disabled','seconds':time.monotonic()-started}
 atomic(dest/'result.json',final)
 lines=['# Nabla-only final test results','',f'Original full official test regimes; fixed {args.endpoint} model. No test-based tuning.','','| Regime | Conformers | RMSD (Å) |','|---|---:|---:|']
 for name,m in groups.items():lines.append(f"| {name} | {m['rows']:,} | {m['mean_rmsd']:.5f} |")
 lines+=['','Conformer-ID overlap must be zero. Full-InChIKey overlaps are reported separately in result.json; molecule overlap in the conformation regime is intentional. Official test regimes are not additionally filtered.']
 (dest/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
