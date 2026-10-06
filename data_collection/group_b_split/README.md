# Iterative Group B InChIKey splits

This continues the verified [Group A pass](../group_a_split/README.md) using the
prepared grouped CSVs. Order: KIBA-3D, OMol25_small_mols, pubchem3d, zinc.
It uses full Standard InChIKeys from the same pinned parser and RDKit version
as Group A. The three unresolved PubChem/ZINC rows remain excluded from the
prepared inputs. Newly encountered format/key failures are preserved in
quarantine evidence and never treated as novel molecules.

For each dataset, test and validation target approximately 0.5% of **input
sample rows**, selecting whole molecular-key groups. Existing global test keys
take priority over validation keys. Matching existing holdouts are selected
first. If their candidate rows exceed the quota, whole groups are subsampled;
the unselected groups are excluded, never reassigned to training. Deficits are
filled only with keys absent from existing train/validation/test ownership.

The initial registry protects fixed Group A tests, nabla structure/scaffold
tests, and GEOM-revisited, plus Group A validation and the reserved quarantined
ChEMBL validation key. Existing Group A and nabla training keys block new holdout
filling. Earlier Group B assignments join the registry after each dataset.
Later Group B pools are still unassigned; they inherit these decisions when
processed. This preserves the iterative method rather than pretending later
unassigned pools are already training sets.

Selection uses SHA-256 ranking of `(seed, dataset, split, full key)`, with seed
42, for reproducible pseudorandom order independent of set iteration or worker
completion. Whole groups can overshoot the target. If insufficient eligible
groups exist, the shortfall is reported rather than violating ownership.
Nabla's conformation test is not a molecular blacklist. There is no scaffold-wide
or stereo-free matching, salt stripping, or conformer collapsing.

## Execution

The first stage indexes the four Group B pools and the existing nabla training
pool in parallel. The second stage uses SQLite to aggregate key counts and
ownership, processes datasets sequentially, and writes CSV chunks in parallel.
The prepared October 2026 input contains 174,502,767 Group B rows and 8,757,879
nabla training-reference rows.

```bash
python data_collection/group_b_split/prepare_index.py \
  --preflight /path/to/resplit_20261005 \
  --group-a /path/to/completed-group-a-run \
  --output /path/to/new-group-b-experiment/manifest.json

python data_collection/group_b_split/index_inputs.py \
  --manifest /path/to/new-group-b-experiment/manifest.json \
  --output /path/to/new-group-b-experiment/index_run --workers 64

split_code_sha=$(sha256sum data_collection/group_b_split/split_pools.py | cut -d' ' -f1)
python data_collection/group_b_split/split_pools.py \
  --index-run /path/to/new-group-b-experiment/index_run \
  --output /path/to/new-group-b-experiment/split_run \
  --runner-sha256 "$split_code_sha" --workers 32
```

Only the standard library and RDKit are required. The preflight must contain
the same artifacts documented for Group A. Its recorded RDKit version must
match the environment. Both stage output directories must be new. Failed
outputs remain as evidence; use a new directory for retries.

`index.sbatch CODE_DIRECTORY MANIFEST NEW_INDEX_DIRECTORY` requests 64 CPUs and
256 GB. `split.sbatch CODE_DIRECTORY INDEX_DIRECTORY NEW_SPLIT_DIRECTORY SHA256`
requests 32 CPUs and 128 GB. Adjust site/account settings as needed. Submit the
second job using `--dependency=afterok:INDEX_JOB_ID`, so it cannot consume an
incomplete index. Both scripts accept `--output` through `sbatch` for persistent
logs. Writable controller databases use up to 8 GiB of page cache each; worker
read caches are limited to 64 MiB each.

## Validation and artifacts

The index verifies every original file hash, code hashes, RDKit, row counts,
and stable input metadata. Chunk receipts pin reports and key-count files.
Writers verify receipt hashes, row-aligned key hashes, and source-record hashes,
then apply one immutable assignment per key. Exact original quarantine
membership uses source path, name, and row hash.

`split_run/candidate/{train,val,test}/*.csv` holds the 12 outputs. Dataset
allocation reports record targets, selected matching/new holdout rows, and
shortfalls. Dataset completion reports record exclusions, quarantine counts,
and output hashes. Final assembly checks fresh output hashes and row counts.
The final report checks every Group B training/validation assignment against
the registry after **all four** datasets, guarding against later holdout leaks.
`parts/` preserves excluded rows and reasons. `index_run/index/` preserves raw
failure evidence and reusable row-key sidecars. `progress.json` in each stage
reports its state. Per-dataset SQLite files preserve complete key assignments;
`ownership.sqlite` carries global ownership forward to Group C.

Completion is not the final corpus release. Group C validation must still be
constructed and its identities applied globally. Final corpus checks and
exact-sample deduplication remain separate from molecule-level splitting.

```bash
python -m unittest discover -s data_collection/group_b_split -p 'test_*.py' -v
```

Tests exercise complete indexing/allocation/writing across two datasets,
test-over-validation precedence, existing-training protection (including nabla),
whole-conformer assignment, quota exclusions, deterministic selection under
different insertion orders, target shortfalls, invalid-row quarantine, byte
chunk boundaries, output accounting, and source immutability.
