"""GPU serialization/decoder parity and end-to-end training-only conversion gate."""
import csv,json,time,os
from argparse import Namespace
from pathlib import Path
import numpy as np
import torch
from molgen3D.data_processing.coordtoken import CoordTokenCodec,parse,sha256
from molgen3D.data_processing.preprocess_coordtoken_csv import run,atomic_json
from molgen3D.data_processing.assemble_coordtoken import assemble
ROOT=Path('/mnt/weka/fgeikyan/CoordToken');OUT=ROOT/'logs/coordtoken_encoding_20261019'
p=json.loads((OUT/'plan.json').read_text());assert torch.cuda.device_count()==1
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
torch.manual_seed(123);torch.cuda.manual_seed_all(123)
codec=CoordTokenCodec(p['checkpoint'],expected_sha256=p['checkpoint_sha256'],device='cuda',precision='bf16')
with (OUT/'pilot_train.csv').open() as f:rows=list(csv.DictReader(f))
texts=[r['enriched_text'] for r in rows[:64]]
streams=codec.encode_batch(texts);decoded=codec.decode_batch(streams)
batch,records=codec.prepare(texts)
with torch.inference_mode(),codec._autocast():
    direct,indices=codec.model(**batch)
for i,((s,xyz),(tokens,coords,atoms)) in enumerate(zip(decoded,records)):
    assert s.graph_tokens==tuple(tokens)
    assert s.molecular_code==int(indices[i,-1])
    assert s.atom_codes==tuple(int(indices[i,j]) for j in s.atom_positions)
    assert len(s.atom_codes)==int(atoms.sum())
    assert np.isfinite(xyz).all()
    np.testing.assert_allclose(xyz[atoms],direct[i,:len(tokens)][torch.tensor(atoms,device='cuda')].float().cpu().numpy(),rtol=1e-4,atol=1e-4)
# Confirm deterministic codes for repeated inference with this fixed batch.
assert streams==codec.encode_batch(texts)
del codec,direct,indices,batch;torch.cuda.empty_cache()
torch.cuda.synchronize();started=time.monotonic()
args=Namespace(input=str(OUT/'pilot_train.csv'),output=str(OUT/'pilot_encoded'),split='train',tokenizer=p['tokenizer'],checkpoint=p['checkpoint'],expected_sha256=p['checkpoint_sha256'],device='cuda',precision=p['precision'],batch_size=p['batch_size'],seq_len=p['seq_len'],num_shards=1,shard_index=0)
receipt=run(args);torch.cuda.synchronize();seconds=time.monotonic()-started
assembled=assemble(args.output,len(rows))
with (Path(args.output)/'part-00000.jsonl').open() as f:output=[json.loads(l) for l in f]
assert len(output)==len(rows)==4224
for original,encoded in zip(rows,output):
    assert all(encoded[k]==v for k,v in original.items() if k!='enriched_text')
    assert encoded['split']=='train';parse(encoded['embedded_smiles'])
info=dict(status='passed',rows=len(rows),seconds=seconds,rows_per_second=len(rows)/seconds,serialized_dense_gpu_parity=True,repeated_batch_codes_equal=True,source_columns_preserved=True,assembled_status=assembled['status'],max_unit_tokens=receipt['max_unit_tokens_with_separator'],output_bytes=(Path(args.output)/'part-00000.jsonl').stat().st_size,precision=p['precision'],batch_size=p['batch_size'],gpu=torch.cuda.get_device_name(),torch=torch.__version__,test_accessed=False)
atomic_json(OUT/'pilot.json',info);print(json.dumps(info),flush=True)
