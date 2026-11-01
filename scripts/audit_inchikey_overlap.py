#!/usr/bin/env python3
"""Read-only, checkpointed exact-text versus full Standard InChIKey audit.

Reconstructs organic-subset brackets as the CoordToken decoder does. Preserves
encoded stereo/isotopes/charges; does not infer stereo from coordinates, remove
salts, or perform custom tautomer normalization. Standard InChI normalization
still applies. Failures are explicit, never treated as molecular nonmatches.
"""
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import csv
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import sqlite3
import sys
import time
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdinchi

COORD=re.compile(r'<[^>]+>')
BRACKET=re.compile(r'\[([^\]]+)\]')
ORGANIC=set('B C N O P S F Cl Br I b c n o p s'.split())
KEY=re.compile(r'^[A-Z]{14}-[A-Z]{8}SA-[A-Z]$')

def base_text(text):
    return COORD.sub('',text)

def identity(base):
    smiles=BRACKET.sub(lambda m:m[1] if m[1] in ORGANIC else m[0],base)
    mol=Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms():
        return '', '', 'SMILES parse failed or empty molecule', ''
    for atom in mol.GetAtoms(): atom.SetAtomMapNum(0)
    canonical=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)
    try:
        inchi,code,message,log,aux=rdinchi.MolToInchi(mol)
        key=rdinchi.InchiToInchiKey(inchi) if inchi and code in (0,1) else ''
        if not inchi.startswith('InChI=1S/') or not KEY.fullmatch(key):
            return '',canonical,f'InChI failure code={code}: {message}', ''
        return key,canonical,'',message if code else ''
    except Exception as error:
        return '',canonical,repr(error),''

def init_worker(exact=None,keys=None,sources=None):
    global EXACT,KEYS,SOURCES,CACHED
    RDLogger.DisableLog('rdApp.*')
    EXACT=exact or {}; KEYS=keys or {}; SOURCES=sources or []
    CACHED=lru_cache(maxsize=50000)(identity)

def reference_batch(bases):
    return [(base,*identity(base)) for base in bases]

def fingerprint(path):
    s=path.stat()
    return {'path':str(path.resolve()),'size':s.st_size,'mtime_ns':s.st_mtime_ns}

def atomic_json(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n'); temp.replace(path)

def source_name(name):
    return next((s for s in SOURCES if name.startswith(s+'_')),name.split('_',1)[0])

def process_batch(batch):
    groups={}
    for number,raw in batch:
        row=next(csv.reader([raw.decode('utf-8')],strict=True))
        if len(row)!=2: raise ValueError(f'Malformed CSV record at data row {number}')
        name,text=row; base=base_text(text); source=source_name(name)
        entry=groups.setdefault((base,source),[0,number,name]); entry[0]+=1
    counts=Counter(); hits=[]; failures=[]; warnings=Counter()
    for (base,source),(n,number,name) in groups.items():
        exact=EXACT.get(base,0)
        key,canonical,error,warning=CACHED(base)
        matched=KEYS.get(key,0) if key else 0
        counts['rows']+=n
        if exact: counts['exact_text_rows']+=n
        if error:
            counts['conversion_failure_rows']+=n
            if exact: counts['exact_match_conversion_failure_rows']+=n
            failures.append((base,source,error,n,number,name))
        else:
            counts['converted_rows']+=n
            # Identical text must map to the same key under this policy.
            if exact & matched != exact:
                raise AssertionError('An exact reference match lost its InChIKey match')
            if matched: counts['inchikey_match_rows']+=n
            if matched and not exact: counts['additional_rows']+=n
            if not matched: counts['key_nonmatch_rows']+=n
        if warning: warnings[warning]+=n
        if exact or matched:
            hits.append((base,source,key,canonical,exact,matched,n,number,name))
    return dict(counts),hits,failures,dict(warnings)

def prepare_reference(db,args,manifest):
    saved=db.execute("SELECT value FROM metadata WHERE key='reference_ready'").fetchone()
    if not saved:
        bases={}; stats=[]
        for i,path in enumerate(args.reference):
            digest=hashlib.sha256(); rows=0
            with path.open('rb') as raw:
                for block in iter(lambda:raw.read(8*1024*1024),b''): digest.update(block)
            with path.open(newline='') as f:
                reader=csv.DictReader(f)
                if reader.fieldnames!=['name','enriched_text']: raise ValueError(f'Unexpected reference header: {path}')
                for row in reader:
                    rows+=1; base=base_text(row['enriched_text'])
                    entry=bases.setdefault(base,{})
                    item=entry.setdefault(i,[0,row['name']]); item[0]+=1
            stats.append({**fingerprint(path),'label':f'{path.parent.name}/{path.stem}','bit':1<<i,'rows':rows,'sha256':digest.hexdigest()})
            print('REFERENCE_READ',stats[-1],flush=True)
        items=list(bases); done=0
        # Rebuild incomplete reference work after an interrupted preparation.
        with db: db.execute('DELETE FROM reference')
        chunks=(items[j:j+500] for j in range(0,len(items),500))
        with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,mp_context=mp.get_context('fork')) as pool:
            for results in pool.map(reference_batch,chunks,chunksize=1):
                records=[]
                for base,key,canonical,error,warning in results:
                    for label,(n,name) in bases[base].items():
                        records.append((base,1<<label,key,canonical,error,warning,n,name))
                with db: db.executemany('INSERT INTO reference VALUES (?,?,?,?,?,?,?,?)',records)
                done+=len(results)
                if done%10000==0: print('REFERENCE_KEYS',done,'/',len(items),flush=True)
        with db: db.execute("INSERT INTO metadata VALUES ('reference_ready',?)",(json.dumps(stats),))
    else: stats=json.loads(saved[0])
    exact={}; keys={}
    for base,bit,key in db.execute('SELECT base,bit,key FROM reference'):
        exact[base]=exact.get(base,0)|bit
        if key: keys[key]=keys.get(key,0)|bit
    summary={'files':stats,'unique_texts':len(exact),'unique_valid_keys':len(keys),
             'failed_unique_texts':db.execute("SELECT COUNT(DISTINCT base) FROM reference WHERE key='' ").fetchone()[0]}
    atomic_json(args.output/'reference_summary.json',summary)
    return exact,keys,summary

def summaries(db,state,reference):
    result={'status':state['status'],'rows_scanned':state['rows'],'counts':state['counts'],
            'elapsed_seconds':state['elapsed_seconds'],'reference':reference,'groups':{}}
    groups=[('all_references',0)]+[(x['label'],x['bit']) for x in reference['files']]
    for label,bit in groups:
        exact='exact_mask != 0' if not bit else f'(exact_mask & {bit}) != 0'
        keyed='key_mask != 0' if not bit else f'(key_mask & {bit}) != 0'
        group={}
        for category,where in [('exact_text',exact),('inchikey',keyed),('additional',f'({keyed}) AND NOT ({exact})'),('exact_without_valid_key',f'({exact}) AND key=\'\'')]:
            n,texts,keys=db.execute(f'SELECT COALESCE(SUM(n),0),COUNT(DISTINCT base),COUNT(DISTINCT NULLIF(key,\'\')) FROM hits WHERE {where}').fetchone()
            by_source=[{'source':s,'training_rows':r,'unique_training_texts':t,'unique_keys':k} for s,r,t,k in db.execute(f'SELECT source,SUM(n),COUNT(DISTINCT base),COUNT(DISTINCT NULLIF(key,\'\')) FROM hits WHERE {where} GROUP BY source')]
            group[category]={'training_rows':n,'unique_training_texts':texts,'unique_keys':keys,'by_source':by_source}
        result['groups'][label]=group
    atomic_json(Path(state['output'])/'summary.json',result)
    return result

def export(db,out):
    queries={'matches.csv':'SELECT * FROM hits ORDER BY first_row',
             'additional_matches.csv':'SELECT * FROM hits WHERE exact_mask=0 AND key_mask!=0 ORDER BY first_row',
             'conversion_failures.csv':'SELECT * FROM failures ORDER BY first_row',
             'reference_keys.csv':'SELECT * FROM reference ORDER BY bit,base'}
    for name,query in queries.items():
        cursor=db.execute(query); temp=out/(name+'.tmp')
        with temp.open('w',newline='') as f:
            w=csv.writer(f); w.writerow([x[0] for x in cursor.description]); w.writerows(cursor)
        temp.replace(out/name)

def benchmark(args,exact,keys,sources):
    records=[]; size=args.train.stat().st_size
    with args.train.open('rb') as f:
        for i in range(20):
            f.seek(size*i//20); f.readline()
            for j in range(1000):
                raw=f.readline()
                if not raw: break
                records.append((len(records)+1,raw))
    start=time.monotonic(); totals=Counter()
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,initargs=(exact,keys,sources),mp_context=mp.get_context('fork')) as pool:
        chunks=[records[j:j+250] for j in range(0,len(records),250)]
        for counts,_,_,_ in pool.map(process_batch,chunks): totals.update(counts)
    elapsed=time.monotonic()-start
    result={'sample_rows':len(records),'elapsed_seconds':elapsed,'rows_per_second':len(records)/elapsed,
            'estimated_190214248_rows_hours':190214248*elapsed/len(records)/3600,'counts':dict(totals),
            'note':'20 spaced file positions; estimate includes worker startup, excludes full reference preparation/export and is not a final overlap count.'}
    atomic_json(args.output/'benchmark.json',result); print('BENCHMARK',json.dumps(result),flush=True)

def batches(handle,start,size,maxrows):
    number=start
    while not maxrows or number<maxrows:
        batch=[]
        for _ in range(size):
            if maxrows and number>=maxrows: break
            raw=handle.readline()
            if not raw: break
            number+=1; batch.append((number,raw))
        if not batch: return
        yield handle.tell(),number,batch

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',type=Path,required=True); p.add_argument('--reference',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--workers',type=int,default=16)
    p.add_argument('--batch-size',type=int,default=4000); p.add_argument('--max-rows',type=int,default=0)
    p.add_argument('--resume',action='store_true'); p.add_argument('--benchmark',action='store_true')
    args=p.parse_args()
    if args.workers<1 or args.batch_size<1 or args.max_rows<0 or len(args.reference)>62: p.error('Invalid limits')
    args.output.mkdir(parents=True,exist_ok=True)
    lock=(args.output/'run.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest={'train':fingerprint(args.train),'references':[fingerprint(x) for x in args.reference],
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rdkit':rdBase.rdkitVersion,
              'python':sys.version,'max_rows':args.max_rows,'policy':'full Standard InChIKey from decoded text; no coordinate stereo inference'}
    db=sqlite3.connect(args.output/'audit.sqlite'); db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT);
    CREATE TABLE IF NOT EXISTS reference(base TEXT,bit INTEGER,key TEXT,canonical TEXT,error TEXT,warning TEXT,n INTEGER,name TEXT,PRIMARY KEY(base,bit));
    CREATE TABLE IF NOT EXISTS hits(base TEXT,source TEXT,key TEXT,canonical TEXT,exact_mask INTEGER,key_mask INTEGER,n INTEGER,first_row INTEGER,name TEXT,PRIMARY KEY(base,source));
    CREATE TABLE IF NOT EXISTS failures(base TEXT,source TEXT,error TEXT,n INTEGER,first_row INTEGER,name TEXT,PRIMARY KEY(base,source));''')
    saved=db.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
    if saved:
        if not args.resume or json.loads(saved[0])!=manifest: raise ValueError('Use --resume with unchanged inputs, code, and versions')
    else:
        with db: db.execute("INSERT INTO metadata VALUES ('manifest',?)",(json.dumps(manifest),))
    atomic_json(args.output/'manifest.json',manifest)
    atomic_json(args.output/'progress.json',{'status':'preparing_reference','job_id':os.getenv('SLURM_JOB_ID')})
    exact,keys,reference=prepare_reference(db,args,manifest)
    sources=sorted({x.stem for d in (args.train.parent/'test',args.train.parent/'val') for x in d.glob('*.csv')}|{'nablaDFT'},key=len,reverse=True)
    if args.benchmark and not (args.output/'benchmark.json').exists(): benchmark(args,exact,keys,sources)
    saved=db.execute("SELECT value FROM metadata WHERE key='state'").fetchone()
    state=json.loads(saved[0]) if saved else {'rows':0,'offset':0,'counts':{},'warnings':{},'elapsed_seconds':0}
    state.update(output=str(args.output.resolve()),status='running',job_id=os.getenv('SLURM_JOB_ID'),workers=args.workers)
    start=time.monotonic(); before=state['elapsed_seconds']; last=start
    try:
        with args.train.open('rb',buffering=8*1024*1024) as handle:
            if next(csv.reader([handle.readline().decode()]))!=['name','enriched_text']: raise ValueError('Unexpected training header')
            if state['offset']: handle.seek(state['offset'])
            with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,initargs=(exact,keys,sources),mp_context=mp.get_context('fork')) as pool:
                pending=deque(); source=iter(batches(handle,state['rows'],args.batch_size,args.max_rows)); exhausted=False
                while pending or not exhausted:
                    while not exhausted and len(pending)<args.workers*2:
                        try: offset,n,batch=next(source)
                        except StopIteration: exhausted=True; break
                        pending.append((offset,n,pool.submit(process_batch,batch)))
                    if not pending: break
                    offset,n,future=pending.popleft(); counts,hits,failures,warnings=future.result()
                    total=Counter(state['counts']); total.update(counts)
                    warn=Counter(state['warnings']); warn.update(warnings)
                    state.update(rows=n,offset=offset,counts=dict(total),warnings=dict(warn),elapsed_seconds=before+time.monotonic()-start)
                    with db:
                        db.executemany('INSERT INTO hits VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(base,source) DO UPDATE SET n=hits.n+excluded.n',hits)
                        db.executemany('INSERT INTO failures VALUES (?,?,?,?,?,?) ON CONFLICT(base,source) DO UPDATE SET n=failures.n+excluded.n',failures)
                        db.execute("INSERT OR REPLACE INTO metadata VALUES ('state',?)",(json.dumps(state),))
                    if time.monotonic()-last>30:
                        atomic_json(args.output/'progress.json',state); print('PROGRESS',json.dumps(state),flush=True); last=time.monotonic()
        if fingerprint(args.train)!=manifest['train'] or [fingerprint(x) for x in args.reference]!=manifest['references']: raise RuntimeError('An input changed during audit')
        state['status']='sample_complete' if args.max_rows and state['offset']<manifest['train']['size'] else 'complete'
        export(db,args.output); summaries(db,state,reference)
    except BaseException as error:
        state['status']='failed'; state['error']=repr(error); raise
    finally:
        state['elapsed_seconds']=before+time.monotonic()-start
        # Do not advance saved offsets here: only transactional batches are resumable.
        atomic_json(args.output/'progress.json',state)
    print('DONE',json.dumps(state),flush=True)

if __name__=='__main__': main()
