import hashlib,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BASE=ROOT/'logs/nabla_b200_20261023'
WIDTH=int(os.environ.get('NABLA_WIDTH','1024'));assert WIDTH in {512,1024}
PHASE='full';OUT=BASE/f'w{WIDTH}'
DATA=Path(os.environ.get('NABLA_DATA_CACHE',str(ROOT/'research/coordtoken_v2/data/nabla_full_training_v2')));INDICES=OUT/'indices'
TRAIN_SHA='59992a703daf84d18c56ac68e543585eede93a1f7f4a11c83109d61e8e28b2bb'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def verify_sources():
 f=json.loads((HERE/'freeze.json').read_text())
 for name,h in f['files'].items():assert sha(HERE/name)==h,'Source changed:'+name
 if (BASE/'launch_receipt.json').exists():
  r=json.loads((BASE/'launch_receipt.json').read_text());assert sha(BASE/'selection.json')==r['selection_sha256']
  assert sha(ROOT/'logs/nabla_retune_20261020/decision.json')==r['selection_sha256']
  for width,h in r['case_sha256'].items():assert sha(BASE/f'w{width}/case.json')==h
 else:raise RuntimeError('No verified tuning selection/preparation')
def verify_training():
 assert sha(DATA/'manifest.json')==TRAIN_SHA
 m=json.loads((DATA/'manifest.json').read_text());assert m['rows']==8849983 and m['all_original_records_retained'];return m
