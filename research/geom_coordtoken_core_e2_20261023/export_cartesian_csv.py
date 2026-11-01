"""Lossless CSV export of the pinned GEOM Cartesian strings, before FSQ."""
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/mnt/weka/fgeikyan/CoordToken')
PLAN = ROOT / 'logs/geom_coordtoken_core_e2_20261023/plan.json'
OUT = ROOT / 'datasets/geom_cartesian_original_20261023'
FIELDS = ['name', 'canonical_smiles', 'enriched_text']


def digest_row(h, values):
    h.update(json.dumps(values, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
    h.update(b'\n')


def export(split, plan):
    target = OUT / (split + '.csv')
    partial = target.with_suffix('.csv.partial')
    assert not target.exists() and not partial.exists(), target
    sources = [e for e in plan['files'] if e['split'] == split]
    logical = hashlib.sha256()
    rows = 0
    with partial.open('x', newline='', encoding='utf-8') as out:
        writer = csv.writer(out)
        writer.writerow(FIELDS)
        for entry in sources:
            path = Path(entry['path'])
            before = path.stat()
            source_hash = hashlib.sha256()
            count = 0
            with path.open('rb') as src:
                for count, raw in enumerate(src, 1):
                    source_hash.update(raw)
                    record = json.loads(raw)
                    canonical, enriched = record['canonical_smiles'], record['embedded_smiles']
                    assert isinstance(canonical, str) and isinstance(enriched, str)
                    values = [f'{split}/{path.name}:{count}', canonical, enriched]
                    writer.writerow(values)
                    digest_row(logical, values)
            after = path.stat()
            assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            assert source_hash.hexdigest() == entry['sha256']
            assert count == entry['rows']
            rows += count
            print(f'{split}: exported {rows:,} rows', flush=True)
        out.flush()
        os.fsync(out.fileno())
    # Read CSV back to verify quoting, all string values, order, and row count.
    actual = hashlib.sha256()
    count = 0
    with partial.open(newline='', encoding='utf-8') as inp:
        reader = csv.reader(inp)
        assert next(reader) == FIELDS
        for count, values in enumerate(reader, 1):
            assert len(values) == len(FIELDS)
            digest_row(actual, values)
    assert count == rows == plan['rows'][split]
    assert actual.hexdigest() == logical.hexdigest()
    binary = hashlib.sha256()
    with partial.open('rb') as inp:
        for block in iter(lambda: inp.read(8 * 1024 * 1024), b''):
            binary.update(block)
    partial.rename(target)
    result = dict(path=str(target), rows=rows, bytes=target.stat().st_size,
                  sha256=binary.hexdigest(), string_roundtrip_verified=True, sources=sources)
    print(f'{split}: VERIFIED {rows:,} rows', flush=True)
    return split, result


def main():
    plan = json.loads(PLAN.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'manifest.json').exists()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = dict(pool.map(lambda s: export(s, plan), ('train', 'valid', 'test')))
    manifest = dict(status='verified', dataset='GEOM-Cartesian-Original',
                    source_root=plan['source_root'], columns=FIELDS, splits=results,
                    coordtoken_applied=False, binning_applied=False,
                    filtering_applied=False, splits_and_row_order_preserved=True,
                    coordinates='Original stored coordinate strings; no numeric conversion or recentering')
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (OUT / 'README.md').write_text('''# GEOM-Cartesian-Original

Original GEOM Cartesian-v3 JSONLs exported losslessly as CSV, before CoordToken
encoding or coordinate binning. This is not GEOM-Revisited or the filtered
mixed-corpus GEOM subset. Existing splits and within-split row order are retained.

Files: `train.csv`, `valid.csv`, `test.csv`.

Columns:
- `name`: stable source locator, `split/source.jsonl:1-based-line`.
- `canonical_smiles`: original JSONL canonical_smiles, unchanged.
- `enriched_text`: original JSONL embedded_smiles, unchanged, including floating
  point Cartesian coordinate triplets. Use this column for coordinate binning.

Read with a CSV parser: coordinates contain commas and fields are quoted.
No filtering, deduplication, shuffling, recentering, or molecular reconstruction
was performed. This verifies the export, not the chemical correctness of the
historical representation. Decorative atom brackets remain as in the source;
use canonical_smiles for the original canonical molecular string.

Source: /data/fgeikyan/shared-with-me/vtarasov/geom_cartesian_v3/processed_strings/

manifest.json records source/output SHA256 hashes and exact split counts.
All CSVs were read back to verify exact string preservation and row counts.
''')
    print(json.dumps({s: v['rows'] for s, v in results.items()}), flush=True)


if __name__ == '__main__':
    main()
