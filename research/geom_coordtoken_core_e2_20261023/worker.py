"""Encode original GEOM JSONLs without changing conditioning SMILES or splits."""
import argparse,fcntl,hashlib,json,os,re,shutil,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from molgen3D.data_processing.coordtoken import CoordTokenCodec,parse,sha256,FORMAT
from molgen3D.data_processing.preprocess_coordtoken_csv import validate_tokenizer,atomic_json
from molgen3D.training.pretraining.dataprocessing.text_processing import build_unit
ROOT=Path('/mnt/weka/fgeikyan/CoordToken');HERE=Path(__file__).resolve().parent
OUT=ROOT/'logs/geom_coordtoken_core_e2_20261023'

def verify():
 frozen=json.loads((HERE/'freeze.json').read_text())
 assert sha256(OUT/'plan.json')==frozen['plan_sha256']
 for name,digest in frozen['files'].items():assert sha256(HERE/name)==digest,name
 p=json.loads((OUT/'plan.json').read_text());source=Path(p['source_code_root'])
 assert sha256(source/'freeze.json')==p['source_freeze_sha256']
 for name,digest in json.loads((source/'freeze.json').read_text())['files'].items():
  if name.startswith('source/'):assert sha256(source/name)==digest,name
 assert sha256(Path(p['tokenizer'])/'coordtoken_contract.json')==p['tokenizer_contract_sha256']
 return p

def pilot(codec,p):
 src=next(e for e in p['files'] if e['split']=='train')
 with Path(src['path']).open() as f:rows=[json.loads(next(f)) for _ in range(64)]
 texts=[r['embedded_smiles'] for r in rows]
 codes=codec.encode_batch(texts);decoded=codec.decode_batch(codes)
 batch,records=codec.prepare(texts)
 with torch.inference_mode(),codec._autocast():direct,indices=codec.model(**batch)
 for i,((stream,xyz),(tokens,coords,atoms)) in enumerate(zip(decoded,records)):
  assert stream.graph_tokens==tuple(tokens)
  assert stream.molecular_code==int(indices[i,-1])
  assert len(stream.atom_codes)==int(atoms.sum())
  assert stream.atom_codes==tuple(int(indices[i,j]) for j in stream.atom_positions)
  assert np.isfinite(xyz).all()
  np.testing.assert_allclose(xyz[atoms],direct[i,:len(tokens)][torch.tensor(atoms,device=codec.device)].float().cpu().numpy(),rtol=1e-4,atol=1e-4)
 assert codes==codec.encode_batch(texts)
 return dict(status='passed',rows=64,source='train only',serialized_dense_parity=True,repeated_batch_codes_equal=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('task',type=int);ap.add_argument('--pilot-only',action='store_true');ap.add_argument('--device',default='cuda');a=ap.parse_args()
 os.umask(0o002);p=verify();entry=p['files'][a.task];assert entry['task']==a.task
 source=Path(entry['path']);st=source.stat();assert (st.st_size,st.st_mtime_ns)==(entry['bytes'],entry['mtime_ns'])
 torch.set_num_threads(4);torch.manual_seed(123)
 if a.device=='cuda':
  assert torch.cuda.device_count()==1;torch.cuda.manual_seed_all(123)
  torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 codec=CoordTokenCodec(p['checkpoint'],expected_sha256=p['checkpoint_sha256'],device=a.device,precision=p['precision'] if a.device=='cuda' else 'fp32')
 tok=AutoTokenizer.from_pretrained(p['tokenizer'],local_files_only=True);ids=validate_tokenizer(tok)
 check=pilot(codec,p);atomic_json(OUT/f'pilot-{a.task:02d}-{a.device}.json',check)
 if a.pilot_only:
  print(json.dumps(check));return
 assert a.device=='cuda','Production precision is fixed to GPU bf16'
 dest=Path(p['output'])/'processed_strings'/entry['split'];dest.mkdir(parents=True,exist_ok=True)
 target=dest/entry['output_name'];receipt=target.with_suffix('.json');temp=target.with_suffix('.jsonl.partial')
 locks=OUT/'locks';locks.mkdir(exist_ok=True)
 lock=(locks/f'{a.task}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if receipt.exists():
  old=json.loads(receipt.read_text());assert old['status']=='complete' and old['rows']==entry['rows']
  assert old['plan_sha256']==sha256(OUT/'plan.json') and sha256(target)==old['output_sha256']
  print('Verified completed shard',a.task);return
 if target.exists() or temp.exists():
  archive=OUT/'failed_attempts'/f'{a.task}-{time.time_ns()}';archive.mkdir(parents=True)
  for file in (target,temp):
   if file.exists():shutil.move(str(file),str(archive/file.name))
 count=0;total_tokens=0;max_tokens=0;pending=[];source_hash=hashlib.sha256();output_hash=hashlib.sha256();started=time.monotonic()
 def flush(out):
  nonlocal count,total_tokens,max_tokens
  if not pending:return
  streams=codec.encode_batch([r['embedded_smiles'] for _,r in pending]);units=[]
  for (offset,row),stream in zip(pending,streams):
   s=parse(stream)
   assert ''.join(s.graph_tokens)==re.sub(r'<[^<>]*>','',row['embedded_smiles'])
   assert isinstance(row['canonical_smiles'],str) and row['canonical_smiles']
   units.append(build_unit(row['canonical_smiles'],stream))
  tokens=tok(units,add_special_tokens=False,padding=False,truncation=False,return_attention_mask=False)['input_ids']
  for (offset,row),stream,unit_ids in zip(pending,streams,tokens):
   n=len(unit_ids)+1
   if n>p['seq_len']:raise ValueError(f'Context overflow at source byte {offset}: {n}')
   record={**row,'embedded_smiles':stream,'coordtoken_format':FORMAT,'coordtoken_sha256':p['checkpoint_sha256'],'coordtoken_model':p['model_name'],'source_file':source.name,'source_jsonl_offset':offset,'split':entry['split']}
   encoded=(json.dumps(record,separators=(',',':'))+'\n').encode();out.write(encoded);output_hash.update(encoded)
   count+=1;total_tokens+=n;max_tokens=max(max_tokens,n)
  pending.clear()
 try:
  with source.open('rb') as stream,temp.open('xb') as out:
   while True:
    offset=stream.tell();line=stream.readline()
    if not line:break
    source_hash.update(line);row=json.loads(line)
    assert isinstance(row,dict) and set(('canonical_smiles','embedded_smiles'))<=row.keys()
    pending.append((offset,row))
    if len(pending)==p['batch_size']:
     flush(out)
     if count%10048==0:
      atomic_json(OUT/f'progress-{a.task:02d}.json',dict(status='encoding',rows=count,expected=entry['rows'],seconds=time.monotonic()-started))
   flush(out)
  now=source.stat();assert (now.st_size,now.st_mtime_ns)==(entry['bytes'],entry['mtime_ns'])
  assert source_hash.hexdigest()==entry['sha256'] and count==entry['rows']
  info=dict(status='complete',format=FORMAT,model_name=p['model_name'],checkpoint_sha256=p['checkpoint_sha256'],tokenizer_sha256=sha256(Path(p['tokenizer'])/'tokenizer.json'),geometry_token_ids=ids,precision=p['precision'],split=entry['split'],source=entry,rows=count,failures=0,lm_tokens_with_separator=total_tokens,max_unit_tokens_with_separator=max_tokens,seq_len=p['seq_len'],output_sha256=output_hash.hexdigest(),plan_sha256=sha256(OUT/'plan.json'),seconds=time.monotonic()-started,gpu=torch.cuda.get_device_name(),torch=torch.__version__,canonical_smiles_preserved=True,job=os.getenv('SLURM_JOB_ID'))
  temp.replace(target);atomic_json(receipt,info);atomic_json(OUT/f'progress-{a.task:02d}.json',info);print(json.dumps({k:v for k,v in info.items() if k not in ('geometry_token_ids','source')}),flush=True)
 except BaseException as exc:
  atomic_json(OUT/f'failure-{a.task:02d}.json',dict(error=repr(exc),rows=count,source=entry,published=False));raise
if __name__=='__main__':main()
