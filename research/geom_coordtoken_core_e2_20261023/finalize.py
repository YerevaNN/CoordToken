"""Publish SFT manifests only after every original GEOM row is encoded."""
import json,os
from pathlib import Path
from molgen3D.data_processing.coordtoken import FORMAT,sha256
from molgen3D.data_processing.preprocess_coordtoken_csv import atomic_json
from molgen3D.data_processing.coordtoken_contract import validate_training_data
from worker import verify,OUT

def main():
 os.umask(0o002);p=verify();root=Path(p['output']);results={}
 for split in ('train','valid','test'):
  directory=root/'processed_strings'/split;entries=[e for e in p['files'] if e['split']==split];parts=[];files=[]
  expected={e['output_name'] for e in entries}
  assert {f.name for f in directory.glob('*.jsonl')}==expected
  for e in entries:
   target=directory/e['output_name'];part=json.loads(target.with_suffix('.json').read_text())
   assert part['status']=='complete' and part['failures']==0 and part['rows']==e['rows']
   assert part['source']==e and part['plan_sha256']==sha256(OUT/'plan.json')
   assert part['checkpoint_sha256']==p['checkpoint_sha256'] and part['tokenizer_sha256']==p['tokenizer_sha256']
   assert sha256(target)==part['output_sha256']
   with target.open('rb') as f:count=sum(1 for _ in f)
   assert count==e['rows']
   parts.append(part);files.append(dict(path=target.name,sha256=part['output_sha256'],rows=count))
  result=dict(status='verified',format=FORMAT,split=split,checkpoint_sha256=p['checkpoint_sha256'],tokenizer_sha256=p['tokenizer_sha256'],geometry_token_ids=parts[0]['geometry_token_ids'],precision=p['precision'],seq_len=p['seq_len'],rows=sum(x['rows'] for x in parts),files=files,lm_tokens_with_separator=sum(x['lm_tokens_with_separator'] for x in parts),max_unit_tokens_with_separator=max(x['max_unit_tokens_with_separator'] for x in parts),model_name=p['model_name'],dataset_name=p['dataset_name'],sources=entries,canonical_smiles_preserved=True,split_membership_preserved=True,new_filtering=False)
  assert result['rows']==p['rows'][split]
  manifest=directory/'coordtoken_manifest.json'
  if manifest.exists():assert json.loads(manifest.read_text())==result
  else:atomic_json(manifest,result)
  results[split]=dict(path=str(directory),rows=result['rows'],manifest_sha256=sha256(manifest))
 validate_training_data(root/'processed_strings/train',p['tokenizer'],p['seq_len'])
 atomic_json(root/'release.json',dict(status='verified',sft_ready=True,model_name=p['model_name'],dataset_name=p['dataset_name'],checkpoint=p['checkpoint'],checkpoint_sha256=p['checkpoint_sha256'],tokenizer=p['tokenizer'],splits=results,plan_sha256=sha256(OUT/'plan.json')))
 atomic_json(OUT/'status.json',dict(status='verified',sft_ready=True,release=str(root/'release.json'),splits=results))
 print(json.dumps(results,indent=2))
if __name__=='__main__':main()
