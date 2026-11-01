import csv,hashlib,json,random
from pathlib import Path
ROOT=Path('/mnt/weka/fgeikyan/CoordToken');HERE=ROOT/'research/coordtoken_encoding_20261019';OUT=ROOT/'logs/coordtoken_encoding_20261019'
release=Path('/mnt/weka/fgeikyan/3dmolgen_bigdata/data_handoff/merged_deduplicated_20261006')
manifest=json.loads((release/'manifest.json').read_text());files={}
for split in ['train','val','test']:
    path=release/f'merged_{split}.csv';entry=next(e for e in manifest['files'] if e['path']==str(path));st=path.stat();assert st.st_size==entry['bytes']
    files[split]={**entry,'mtime_ns':st.st_mtime_ns}
src=Path(files['train']['path']);rng=random.Random(20261018);sample=OUT/'pilot_train.csv';offsets=[]
assert not sample.exists()
with src.open('rb') as f,sample.open('wb') as dest:
    header=f.readline();dest.write(header)
    for start in [0]+[rng.randrange(len(header),files['train']['bytes']-100000) for _ in range(31)]:
        if start==0:f.seek(len(header))
        else:f.seek(start);f.readline()
        for _ in range(64):
            off=f.tell();line=f.readline()
            if not line:break
            row=next(csv.reader([line.decode()],strict=True));assert len(row)==8
            offsets.append(off);dest.write(line)
assert len(offsets)==2048 and len(set(offsets))==2048
plan=dict(status='prepared',source_commit='d99ea4b68a784e48be86524b7a60ca679ca22bc7',files=files,checkpoint=str(ROOT/'artifacts/releases/coordtoken-atom12-reg1-mixed-epoch2-20261017/model.pt'),checkpoint_sha256='29d487cf3783978c801c5a237e3c3c86cc23ab058aa7edea7eff0aa707c4bc4f',output=str(ROOT/'datasets/coordtoken_atom12_reg1_epoch2_20261018'),tokenizer=str(HERE/'source/src/molgen3D/training/tokenizers/Qwen3_tokenizer_coordtoken_atom12_reg1'),precision='bf16',batch_size=64,seq_len=4096,pilot_rows=2048,pilot_offsets=offsets,test_results_used=False)
(OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print('Plan and 2048-row training-only pilot prepared')
