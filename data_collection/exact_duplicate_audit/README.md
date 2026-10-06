# Exact-sample duplicate audit

This read-only audit compares the exact decoded `enriched_text` CSV field across
the verified merged train, validation and test files, plus the separately held
GEOM-Revisited test file. Names, molecular equivalence, coordinate rounding,
rotation and atom ordering do not define equality here. Distinct encoded
conformations remain distinct samples.

64 workers scan byte regions from the verified merge receipts, checking region
hashes and row counts. They record SHA-256 text digests, dataset/split category,
and source byte offset into 256 buckets. Up to 32 reducers sort these records
and reread every candidate occurrence to verify full-string equality. Hash
collisions cannot create a false duplicate report. Inputs are never modified.

`report.json` summarizes duplicate text groups, rows participating in them,
redundant occurrences globally and within each split/dataset, cross-split and
cross-dataset intersections, and examples. `buckets/*_groups.jsonl` records every
confirmed group with the original names and byte offsets of its occurrences.
Counts of redundant rows are statistical; the audit does not choose deletions.
Cross-split pair counts measure distinct shared text strings, not row pairs.

```bash
python -m unittest discover -s data_collection/exact_duplicate_audit -p 'test_*.py' -v
sbatch --output=/path/to/audit-%j.log data_collection/exact_duplicate_audit/run.sbatch \
  /absolute/path/to/data_collection/exact_duplicate_audit \
  /absolute/path/to/merged_release /absolute/path/to/new_audit_output
```

Tests cover repeated names with distinct texts, exact repetitions across
multiple datasets and splits, rejection of altered source regions, and forced
hash collisions requiring full-string confirmation.
