# Merged CSVs with source row provenance

The merger preserves the verified Group A/B/C split assignments. It writes
`merged_train.csv`, `merged_val.csv`, and `merged_test.csv`, with columns:

| Column | Meaning |
| --- | --- |
| `name` | Unique stable record ID: `<source_file_id>:<source_row_index>` |
| `enriched_text` | Unchanged decoded CSV field from the input |
| `dataset` | Dataset name; all nabla subsets use `nablaDFT` |
| `source_name` | Exact original input `name` field |
| `source_file_id` | Source identifier recorded in `sources.json` |
| `source_row_index` | Zero-based CSV data-row index before this split/filter run, excluding header |
| `inchikey` | Full Standard InChIKey used by the split pipeline |
| `test_subset` | Nabla `conformations`, `scaffolds`, or `structures`; empty otherwise |

`sources.json` records each prepared input path, checksum, row count, original
split/pool, dataset, and source ID. IDs are derived from the source path and
checksum. The row indices refer to prepared input CSVs, not original raw
SDF/pickle/database record indices. Output order is source-manifest order then
source row order; no random shuffle or exact-sample deduplication occurs.

Workers align the original records with every retained and excluded split
chunk, verifying complete source/chunk hashes, split-part hashes, counts, and
row-aligned key hashes. This preserves provenance even when names or complete
records repeat. The conformation test is preserved; its previously unneeded
keys are computed using the same frozen parser and RDKit version.

Workers write separate temporary chunks. Assembly assigns disjoint byte ranges
in the three final files and copies chunks in parallel with `pwrite`. A complete
readback checks each region's checksum and row count, and computes whole-file
SHA-256 checksums. `manifest.json` is written only after all checks pass.

```bash
python -m unittest discover -s data_collection/merge_release -p 'test_*.py' -v
sbatch --output=/path/to/merge-%j.log data_collection/merge_release/run.sbatch \
  /absolute/path/to/data_collection/merge_release \
  /absolute/path/to/data_handoff /absolute/path/to/new_merged_release
```

The launcher requests 64 CPUs and 256 GB of memory. The number of assembly
workers is capped at 32 to limit concurrent filesystem writes. The completed
molecular overlap checks are inherited through the pinned input manifest;
merging does not introduce new split assignments.
