"""Create a new run directory from verified existing inputs; no scheduler changes."""
import json
import os
from pathlib import Path
from common import ROOT, BASE, HERE, sha, atomic
from schedule import endpoint_plan, WARMUP_EPOCHS


def main():
    os.umask(0o002)
    old = ROOT/'logs/nabla_b200_20261022'
    assert not BASE.exists(), 'Destination already exists; inspect instead of overwriting'
    BASE.mkdir()
    (BASE/'selection.json').write_bytes((old/'selection.json').read_bytes())
    receipt = json.loads((old/'launch_receipt.json').read_text())
    for width in (512,1024):
        source = old/f'w{width}'
        assert not (source/'main').exists(), 'An old training job has already started'
        dest = BASE/f'w{width}'
        dest.mkdir()
        case = json.loads((source/'case.json').read_text())
        case.update(epochs=22, warmup_epochs=WARMUP_EPOCHS,
                    endpoints=list(endpoint_plan(case['rows'],case['global_batch'])))
        atomic(dest/'case.json',case)
        prep = json.loads((source/'preparation.json').read_text())
        prep.update(case_sha256=sha(dest/'case.json'), previous_preparation_sha256=sha(source/'preparation.json'))
        atomic(dest/'preparation.json',prep)
        for name in ('holdout.npy','indices','train_probe_ids.npy','training_ids.npy','validation','validation_keys.json'):
            target = source/name
            (dest/name).symlink_to(os.path.relpath(target,dest),target_is_directory=target.is_dir())
        receipt['case_sha256'][str(width)] = sha(dest/'case.json')
    receipt.update(source_freeze_sha256=sha(HERE/'freeze.json'), endpoints=[3,6,12,22],
                   warmup_epochs=WARMUP_EPOCHS,
                   migration_note='User-approved full epochs 3,6,12,22 and once-only 0.3-epoch warmup. Same selected model/optimizer, shared validation and saved epoch orders. Existing jobs retained.',
                   previous_launch_receipt_sha256=sha(old/'launch_receipt.json'))
    atomic(BASE/'launch_receipt.json',receipt)
    print(json.dumps(receipt,indent=2))

if __name__=='__main__': main()
