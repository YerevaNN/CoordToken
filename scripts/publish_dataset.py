"""Byte-preserving, restartable CSV packaging and Hugging Face publication.

Requires huggingface_hub and zstandard. Credentials come from HF's normal cache
or environment, never from command arguments or stored release metadata.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import zstandard as zstd


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def package(source, output, target_bytes=2 * 1024**3):
    """Supports the final release's one-CSV-record-per-physical-line schema."""
    manifest = json.loads((source / 'manifest.json').read_text())
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for item in manifest['files']:
        filename = Path(item['path']).name
        src = source / filename
        split = {'merged_train.csv': 'train', 'merged_val.csv': 'validation',
                 'merged_test.csv': 'test'}[filename]
        before = src.stat()
        if before.st_size != item['bytes']:
            raise ValueError(f'Source size mismatch: {src}')
        receipt_path = output / f'{split}.json'
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt['source_sha256'] != item['sha256']:
                raise ValueError('Different source release in existing output')
            for shard in receipt['shards']:
                if sha256(output / shard['path']) != shard['sha256']:
                    raise ValueError(f'Shard integrity mismatch: {shard["path"]}')
            files.append(receipt)
            continue
        folder = output / 'data' / split
        folder.mkdir(parents=True, exist_ok=True)
        # Incomplete split shards are deterministic outputs of this publisher.
        for old in folder.glob('part-*.csv.zst*'):
            old.unlink()
        digest = hashlib.sha256()
        rows = 0
        shards = []
        with src.open('rb') as reader:
            header = reader.readline()
            if not header.endswith(b'\n'):
                raise ValueError('Missing CSV header newline')
            digest.update(header)
            pending = reader.read(8 * 1024**2)
            while pending:
                # Finish the final physical row in this block.
                if not pending.endswith(b'\n'):
                    pending += reader.readline()
                path = folder / f'part-{len(shards):05d}.csv.zst'
                temporary = Path(str(path) + '.tmp')
                shard_rows, raw_bytes = 0, 0
                with temporary.open('wb') as sink:
                    with zstd.ZstdCompressor(level=3, threads=2).stream_writer(sink) as writer:
                        writer.write(header)
                        while pending:
                            digest.update(pending)
                            count = pending.count(b'\n')
                            if not pending.endswith(b'\n'):
                                count += 1
                            shard_rows += count
                            raw_bytes += len(pending)
                            writer.write(pending)
                            pending = reader.read(8 * 1024**2)
                            if pending and not pending.endswith(b'\n'):
                                pending += reader.readline()
                            if raw_bytes >= target_bytes:
                                break
                temporary.replace(path)
                rows += shard_rows
                shards.append({'path': str(path.relative_to(output)), 'rows': shard_rows,
                               'bytes': path.stat().st_size, 'sha256': sha256(path)})
                print(f'{split}: {rows:,}/{item["rows"]:,} rows, {len(shards)} shards', flush=True)
                save(output / 'progress.json', {'stage': 'packaging', 'split': split,
                     'rows': rows, 'target_rows': item['rows'], 'updated': time.time()})
        after = src.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Source changed during packaging')
        if digest.hexdigest() != item['sha256'] or rows != item['rows']:
            raise ValueError(f'Source hash/count mismatch: {src}')
        receipt = {'split': split, 'source_file': filename,
                   'source_bytes': item['bytes'], 'source_sha256': item['sha256'],
                   'rows': rows, 'shards': shards}
        save(receipt_path, receipt)
        files.append(receipt)
    release = {'status': 'verified', 'format': 'CSV compressed with Zstandard',
               'header_in_every_shard': True, 'preserves_source_row_order': True,
               'source_manifest_sha256': sha256(source / 'manifest.json'), 'splits': files}
    save(output / 'release.json', release)
    return release


def publish(output, repo_id):
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError
    api = HfApi()
    try:
        existing = api.dataset_info(repo_id)
    except RepositoryNotFoundError:
        existing = None
    if existing is None:
        api.create_repo(repo_id, repo_type='dataset', private=True)
    elif not existing.private:
        raise ValueError('Refusing to replace an existing public repository; review it first')
    save(output / 'progress.json', {'stage': 'uploading', 'updated': time.time()})
    api.upload_large_folder(repo_id=repo_id, repo_type='dataset', folder_path=str(output),
        allow_patterns=['README.md', 'release.json', 'data/**/*.zst', 'provenance/*'],
        num_workers=4, print_report=False, print_report_every=60)
    release = json.loads((output / 'release.json').read_text())
    info = api.dataset_info(repo_id, files_metadata=True)
    remote = {entry.rfilename: entry for entry in info.siblings}
    for split in release['splits']:
        for shard in split['shards']:
            entry = remote.get(shard['path'])
            if entry is None or entry.size != shard['bytes'] or entry.lfs is None:
                raise ValueError(f'Missing/mismatched remote shard: {shard["path"]}')
            if entry.lfs.sha256 != shard['sha256']:
                raise ValueError(f'Remote hash mismatch: {shard["path"]}')
    api.update_repo_settings(repo_id=repo_id, repo_type='dataset', private=False)
    info = api.dataset_info(repo_id)
    if info.private:
        raise ValueError('Repository still private')
    save(output / 'publication.json', {'status': 'published_verified', 'repo_id': repo_id,
         'revision': info.sha, 'shards_verified': sum(len(s['shards']) for s in release['splits']),
         'updated': time.time()})
    save(output / 'progress.json', {'stage': 'published_verified', 'revision': info.sha})
    print(f'Published https://huggingface.co/datasets/{repo_id} at {info.sha}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo', default='FilyaGeikyan/CoordToken-data')
    parser.add_argument('--package-only', action='store_true')
    args = parser.parse_args()
    import fcntl
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'publisher.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            package(args.source, args.output)
            if not args.package_only:
                if not (args.output / 'README.md').exists():
                    raise ValueError('Prepare and review the dataset card before upload')
                publish(args.output, args.repo)
        except Exception as exc:
            save(args.output / 'progress.json', {'stage': 'failed', 'error_type': type(exc).__name__,
                                               'error': str(exc), 'updated': time.time()})
            raise


if __name__ == '__main__':
    main()
