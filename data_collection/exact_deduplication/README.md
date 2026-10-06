# Exact deduplication of merged splits

This step consumes the verified merged release and its completed exact-sample
audit. Within each merged train/validation/test file, it retains the first
occurrence of each exact `enriched_text` string in existing file order. Distinct
conformations remain distinct. It writes a new release and leaves inputs intact.
GEOM-Revisited remains an independent, unchanged evaluation benchmark; shared
samples between it and the GEOM test set remain in both benchmarks.

Every deletion is rechecked against its retained owner using the complete text
and full InChIKey. The output row bytes and stable source IDs remain unchanged.
`duplicate_aliases.csv` preserves every removed row's original metadata and its
retained record ID, including cross-dataset duplicates. `sources.json` preserves
the prepared-input paths and hashes. Dataset ownership follows original merged
file order, rather than a claim that one source is scientifically preferable.

The output `manifest.json` records row counts and checksums; `category_manifest.json`
records per-dataset/split counts. The latter describes categories within merged
files, not separate physical per-dataset CSVs. Parallel filtering and assembly
are followed by parallel region readback and full-file SHA-256 calculation.
A fresh exact-duplicate audit must pass before this release is promoted.

```bash
python -m unittest discover -s data_collection/exact_deduplication -p 'test_*.py' -v
sbatch --output=/path/to/dedup-%j.log data_collection/exact_deduplication/run.sbatch \
  /absolute/path/to/data_collection/exact_deduplication \
  /absolute/path/to/merged_release /absolute/path/to/completed_exact_audit \
  /absolute/path/to/new_deduplicated_release
```

Tests cover cross-source ownership and aliases, preserving distinct conformers,
refusing incorrect deletion matches, source integrity, and unchanged chunks.
