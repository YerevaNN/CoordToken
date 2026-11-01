"""One allocation per width: verified startup, main trajectory and four decays.

Never submits, cancels or modifies Slurm jobs. Existing jobs dispatch here.
Startup updates are retained; branches resume optimizer and all rank RNG states.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from common import HERE, OUT, INDICES, BASE, ROOT, sha, atomic, verify_sources, verify_training
from schedule import endpoint_plan, WARMUP_EPOCHS


def read(path):
    return json.loads(Path(path).read_text())


def verify_inputs():
    verify_sources()
    verify_training()
    case = read(OUT/'case.json')
    assert case['warmup_epochs'] == WARMUP_EPOCHS == .3
    assert case['epochs'] == 22 and case['global_batch'] == 256
    assert case['endpoints'] == list(endpoint_plan(case['rows'], case['global_batch']))
    decision = read(BASE/'selection.json')
    assert decision['status'] == 'selected' and not decision['test_used']
    selected = next(x for x in decision['selected_by_scale'] if x['case']['width'] == case['width'])
    assert selected['result_sha256'] == sha(ROOT/'logs/nabla_retune_20261020/runs'/selected['case']['name']/'result.json')
    for key in ('width','layers','bits','registers','lr','global_batch','dropout','seed','weight_decay','clip','beta2'):
        if key in selected['case']:
            assert case[key] == selected['case'][key], key
    prep = read(OUT/'preparation.json')
    assert prep['status'] == 'passed' and not prep['test_used']
    assert prep['case_sha256'] == sha(OUT/'case.json')
    for name, field in [('train_probe_ids.npy','probe_sha256'), ('training_ids.npy','training_ids_sha256'), ('holdout.npy','holdout_sha256'), ('validation_keys.json','validation_keys_sha256')]:
        assert prep[field] == sha(OUT/name), name
    assert prep['validation_sha256'] == sha(OUT/'validation/manifest.json')
    for name, digest in read(OUT/'validation/manifest.json')['files'].items():
        assert sha(OUT/'validation'/name) == digest, name
    assert prep['indices_manifest_sha256'] == sha(INDICES/'manifest.json')
    for item in prep['arrays']:
        st = Path(item['path']).stat()
        assert (st.st_size, st.st_mtime_ns) == (item['bytes'], item['mtime_ns']), item['path']
    for order in read(INDICES/'manifest.json')['orders']:
        assert sha(INDICES/f"epoch_{order['epoch']:02d}.npy") == order['sha256']
    return case


def stages(plan):
    for endpoint, spec in plan.items():
        yield None, spec['start_step']
        yield endpoint, spec['end_step']


def record_profile(case, plan):
    import torch
    dest = OUT/'main'
    result = read(dest/'result.json')
    assert result['status'] == 'stopped' and result['steps'] == 200
    manifest = read(dest/'manifest.json')
    state = torch.load(dest/'last.pt', map_location='cpu', weights_only=False, mmap=True)
    assert state['step'] == 200 and state['manifest'] == manifest
    assert len(state['rng']) == 8
    assert all(s['cpu'].numel() and s['cuda'].numel() for s in state['rng'])
    assert {int(s['step']) for s in state['optimizer']['state'].values()} == {200}
    del state
    events = {}
    for line in (dest/'metrics_rank0.jsonl').read_text().splitlines():
        event = json.loads(line)
        if event['step'] in (100, 200):
            events[event['step']] = event
    assert set(events) == {100, 200}
    seconds = max(events[100]['training_seconds']/100,
                  (events[200]['training_seconds']-events[100]['training_seconds'])/100)
    assert math.isfinite(seconds) and seconds > 0
    extra_updates = sum(p['end_step']-p['start_step'] for p in list(plan.values())[:-1])
    total_updates = plan['epoch_22']['end_step'] + extra_updates
    atomic(OUT/'gpu_profile.json', dict(status='passed', world=8, steps=200,
           seconds_per_update=seconds, epoch_hours=math.ceil(case['rows']/256)*seconds/3600,
           epoch22_trajectory_hours=plan['epoch_22']['end_step']*seconds/3600,
           all_four_endpoints_training_hours=total_updates*seconds/3600,
           total_executed_updates=total_updates, preliminary=True,
           excludes='queue, validation, checkpoint IO and process initialization',
           checkpoint_sha256=sha(dest/'last.pt'), optimizer_steps=[200], rng_ranks=8,
           training_updates_retained=True, test_used=False,
           case_sha256=sha(OUT/'case.json'), source_freeze_sha256=sha(HERE/'freeze.json')))


def verify_stage(dest, step, endpoint, case):
    result = read(dest/'result.json')
    assert result['status'] == 'complete' and result['health_gate'] == 'passed'
    assert result['steps'] == step
    assert result['checkpoint_sha256'] == sha(dest/'model.pt')
    assert result['manifest_sha256'] == sha(dest/'manifest.json')
    if endpoint:
        spec = endpoint_plan(case['rows'], case['global_batch'])[endpoint]
        assert result['lr'] == 0. and result['examples_seen'] == spec['end_examples']
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    os.umask(0o002)
    case = verify_inputs()
    if args.verify_only:
        print(json.dumps(dict(status='passed', width=case['width'], warmup_epochs=WARMUP_EPOCHS,
                              endpoints=endpoint_plan(case['rows'], case['global_batch']))))
        return
    lock = (OUT/'full_run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = endpoint_plan(case['rows'], case['global_batch'])
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
               '--nproc_per_node=8', str(HERE/'train.py'), '--resume']
    started = time.time()
    def status(**fields):
        atomic(OUT/'run_status.json', dict(job_id=os.environ.get('SLURM_JOB_ID'),
               width=case['width'], started_unix=started, updated_unix=time.time(),
               warmup_epochs=WARMUP_EPOCHS, endpoints=plan, **fields))
    try:
        if not (OUT/'gpu_profile.json').exists():
            for step in (100, 200):
                gate = OUT/'main'/f'ddp_gate_{step}.json'
                if not gate.exists():
                    status(status='startup_check', stop_step=step)
                    subprocess.run(command + ['--stop-step', str(plan['epoch_03']['start_step']),
                                              '--stop-after', str(step)], check=True)
                receipt = read(gate)
                assert receipt['status'] == 'passed' and receipt['world'] == 8
                assert receipt['resumed_at_step'] == (100 if step == 200 else 0)
            record_profile(case, plan)
        profile = read(OUT/'gpu_profile.json')
        assert profile['status'] == 'passed' and profile['case_sha256'] == sha(OUT/'case.json')
        assert profile['source_freeze_sha256'] == sha(HERE/'freeze.json')
        for endpoint, step in stages(plan):
            dest = OUT/(endpoint or 'main')
            done = dest/f'verified_stage_{step:09d}.json'
            if done.exists():
                previous = read(done)
                assert previous['status'] == 'passed'
                assert previous['source_freeze_sha256'] == sha(HERE/'freeze.json')
                if endpoint:
                    verify_stage(dest, step, endpoint, case)
                continue
            status(status='training', stage=endpoint or 'main', stop_step=step)
            extra = ['--stop-step', str(step)]
            if endpoint:
                anchor = OUT/'main'/f'branch_start_{endpoint}.pt'
                if not (dest/'last.pt').exists():
                    assert read(anchor.with_suffix('.json'))['sha256'] == sha(anchor)
                extra += ['--endpoint', endpoint]
            subprocess.run(command + extra, check=True)
            result = verify_stage(dest, step, endpoint, case)
            atomic(done, dict(status='passed', result=result,
                             source_freeze_sha256=sha(HERE/'freeze.json')))
            if endpoint:
                # Keep the final optimizer checkpoint and model. These recovery
                # aliases and the consumed branch anchor are created by this run.
                for recovery in dest.glob('recovery_*.pt'):
                    recovery.unlink()
                (OUT/'main'/f'branch_start_{endpoint}.pt').unlink(missing_ok=True)
        status(status='complete', test_used=False)
    except BaseException as exc:
        status(status='failed', error=repr(exc), test_used=False)
        raise


if __name__ == '__main__':
    main()
