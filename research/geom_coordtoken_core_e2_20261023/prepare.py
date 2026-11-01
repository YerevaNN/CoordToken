"""Pin original GEOM Cartesian splits and the pre-CASF epoch-2 checkpoint."""
import concurrent.futures,hashlib,json
from pathlib import Path
ROOT=Path('/mnt/weka/fgeikyan/CoordToken')
HERE=Path(__file__).resolve().parent
OUT=ROOT/'logs/geom_coordtoken_core_e2_20261023'
SOURCE=Path('/data/fgeikyan/shared-with-me/vtarasov/geom_cartesian_v3/processed_strings')
CODE=ROOT/'research/coordtoken_encoding_20261019'
DEST=Path('/mnt/weka/fgeikyan/3dmolgen_bigdata/datasets/geom_coordtoken_core_e2_20261023')
SHA='29d487cf3783978c801c5a237e3c3c86cc23ab058aa7edea7eff0aa707c4bc4f'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def inventory(path):
 before=path.stat(); h=hashlib.sha256();rows=0;tail=b''
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):
   h.update(b);rows+=b.count(b'\n');tail=b[-1:]
 if tail and tail!=b'\n':rows+=1
 after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
 idx=Path(str(path)+'.idx')
 if idx.exists():assert idx.stat().st_size//8==rows and idx.stat().st_size%8==0
 print(path.name,rows,flush=True)
 return dict(path=str(path),split=path.parent.name,bytes=before.st_size,mtime_ns=before.st_mtime_ns,sha256=h.hexdigest(),rows=rows)
def main():
 assert not (OUT/'plan.json').exists()
 files=[p for split in ('train','valid','test') for p in sorted((SOURCE/split).glob('*.jsonl'))]
 assert len(files)==8
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: entries=list(pool.map(inventory,files))
 for i,e in enumerate(entries):e.update(task=i,output_name=f"part-{i:05d}.jsonl")
 art=ROOT/'artifacts/releases/coordtoken-atom12-reg1-mixed-epoch2-20261017'
 assert sha(art/'model.pt')==SHA
 tok=CODE/'source/src/molgen3D/training/tokenizers/Qwen3_tokenizer_coordtoken_atom12_reg1'
 contract=json.loads((tok/'coordtoken_contract.json').read_text());assert contract['checkpoint_sha256']==SHA
 plan=dict(status='prepared',model_name='CoordToken-Core-E2',dataset_name='GEOM-CoordToken-Core-E2',source_kind='Original GEOM Cartesian-v3 SFT splits; not GEOM-Revisited and not mixed-corpus GEOM subset',source_root=str(SOURCE),files=entries,output=str(DEST),checkpoint=str(art/'model.pt'),checkpoint_sha256=SHA,tokenizer=str(tok),tokenizer_sha256=sha(tok/'tokenizer.json'),tokenizer_contract_sha256=sha(tok/'coordtoken_contract.json'),source_code_root=str(CODE),source_freeze_sha256=sha(CODE/'freeze.json'),precision='bf16',batch_size=64,seq_len=4096,seed=123,split_mapping={'train':'train','valid':'valid','test':'test'},canonical_smiles_policy='Preserve original JSONL canonical_smiles verbatim',new_filtering=False,rows={s:sum(e['rows'] for e in entries if e['split']==s) for s in ('train','valid','test')})
 (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 print(json.dumps({'rows':plan['rows'],'output':plan['output'],'model':plan['model_name']},indent=2),flush=True)
if __name__=='__main__':main()
