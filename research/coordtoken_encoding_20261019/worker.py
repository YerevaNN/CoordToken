"""Idempotent shard runner. Preserve failed attempts and never publish partial data."""
import argparse,fcntl,json,os,time,shutil
from argparse import Namespace
from pathlib import Path
import torch
from molgen3D.data_processing.coordtoken import sha256
from molgen3D.data_processing.preprocess_coordtoken_csv import run,atomic_json
ROOT=Path('/mnt/weka/fgeikyan/CoordToken');HERE=ROOT/'research/coordtoken_encoding_20261019';OUT=ROOT/'logs/coordtoken_encoding_20261019'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('split',choices=['train','val','test']);parser.add_argument('num_shards',type=int);parser.add_argument('shard',type=int);a=parser.parse_args()
    assert sha256(OUT/'plan.json')==json.loads((HERE/'freeze.json').read_text())['plan_sha256']
    p=json.loads((OUT/'plan.json').read_text());gate=json.loads((OUT/'pilot.json').read_text());assert gate['status']=='passed'
    for name,digest in json.loads((HERE/'freeze.json').read_text())['files'].items():
        assert sha256(HERE/name)==digest,'Changed frozen source: '+name
    source=p['files'][a.split];st=Path(source['path']).stat();assert (st.st_size,st.st_mtime_ns)==(source['bytes'],source['mtime_ns'])
    assert torch.cuda.device_count()==1
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.manual_seed(123);torch.cuda.manual_seed_all(123)
    dest=Path(p['output'])/a.split;dest.mkdir(parents=True,exist_ok=True)
    stem=f'part-{a.shard:05d}';control=OUT/'locks'/a.split;control.mkdir(parents=True,exist_ok=True)
    with (control/(stem+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        receipt=dest/(stem+'.json');target=dest/(stem+'.jsonl')
        if receipt.exists() and target.exists():
            info=json.loads(receipt.read_text())
            assert info['status']=='complete' and info['rows']>=0 and info['failures']==0
            for key,val in {'checkpoint_sha256':p['checkpoint_sha256'],'precision':p['precision'],'num_shards':a.num_shards,'shard_index':a.shard,'source':source['path'],'source_bytes':source['bytes'],'source_mtime_ns':source['mtime_ns'],'split':a.split,'seq_len':p['seq_len']}.items():assert info[key]==val,key
            assert sha256(target)==info['output_sha256']
            print('Verified already-completed shard',a.split,a.shard,flush=True);return
        stale=[dest/(stem+s) for s in ['.json','.jsonl','.jsonl.partial','.failure.json','.lock'] if (dest/(stem+s)).exists()]
        if stale:
            archive=OUT/'failed_attempts'/a.split/stem/f"{os.getenv('SLURM_JOB_ID','manual')}-{time.time_ns()}";archive.mkdir(parents=True)
            for path in stale:shutil.move(str(path),str(archive/path.name))
        started=time.monotonic()
        args=Namespace(input=source['path'],output=str(dest),split=a.split,tokenizer=p['tokenizer'],checkpoint=p['checkpoint'],expected_sha256=p['checkpoint_sha256'],device='cuda',precision=p['precision'],batch_size=p['batch_size'],seq_len=p['seq_len'],num_shards=a.num_shards,shard_index=a.shard)
        info=run(args);elapsed=time.monotonic()-started
        atomic_json(OUT/f'{a.split}-{a.shard:05d}-runtime.json',dict(status='complete',seconds=elapsed,rows=info['rows'],rows_per_second=info['rows']/elapsed,gpu=torch.cuda.get_device_name(),torch=torch.__version__,slurm_job=os.getenv('SLURM_JOB_ID')))
        print(json.dumps(dict(split=a.split,shard=a.shard,rows=info['rows'],seconds=elapsed)),flush=True)
if __name__=='__main__':main()
