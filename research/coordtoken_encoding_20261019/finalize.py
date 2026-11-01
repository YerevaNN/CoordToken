"""Full source-hash and output-count verification before exposing an LM manifest."""
import argparse,json,fcntl
from pathlib import Path
from molgen3D.data_processing.coordtoken import sha256
from molgen3D.data_processing.assemble_coordtoken import assemble
from molgen3D.data_processing.preprocess_coordtoken_csv import atomic_json
ROOT=Path('/mnt/weka/fgeikyan/CoordToken');OUT=ROOT/'logs/coordtoken_encoding_20261019'
p=json.loads((OUT/'plan.json').read_text())
a=argparse.ArgumentParser();a.add_argument('split',choices=['train','val','test']);args=a.parse_args()
s=p['files'][args.split];path=Path(s['path']);before=path.stat()
assert (before.st_size,before.st_mtime_ns)==(s['bytes'],s['mtime_ns'])
assert sha256(path)==s['sha256'],'Source content does not match released dataset'
after=path.stat();assert (after.st_size,after.st_mtime_ns)==(before.st_size,before.st_mtime_ns)
root=Path(p['output'])/args.split
assert not list(root.glob('*.partial')) and not list(root.glob('*.failure.json')) and not list(root.glob('*.lock'))
r=assemble(root,s['rows']);assert r['status']=='verified'
atomic_json(OUT/(args.split+'_verified.json'),dict(status='verified',rows=r['rows'],source_sha256=s['sha256'],manifest=str(root/'coordtoken_manifest.json'),manifest_sha256=sha256(root/'coordtoken_manifest.json'),lm_tokens=r['lm_tokens_with_separator']))
with (OUT/'release.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    if all((OUT/(split+'_verified.json')).exists() for split in ['train','val','test']):
        atomic_json(Path(p['output'])/'release.json',dict(status='verified',checkpoint_sha256=p['checkpoint_sha256'],splits={split:json.loads((OUT/(split+'_verified.json')).read_text()) for split in ['train','val','test']},training_started=False))
print(json.dumps(dict(split=args.split,status=r['status'],rows=r['rows'])),flush=True)
