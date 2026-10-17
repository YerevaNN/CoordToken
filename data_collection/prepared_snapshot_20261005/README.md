# Prepared input snapshot

These are exact copies of the October 5 prepared-input scripts and small evidence files. [source_hashes.json](source_hashes.json) records their original paths and SHA-256 values. Large CSV/reference data are external and identified in `prepared_input_manifest.json` and `references/manifest.json`.

The scripts record the input repairs, reference verification, nabla pool lineage, and bounded quarantine investigation used before the completed October 6 splitting. They contain historical absolute cluster paths and are not safe defaults for a fresh repair run: inspect paths and use a separate output snapshot. Do not overwrite preserved sources or rerun already completed repairs blindly.

`preflight_complete.json` and the prepared manifest define the snapshot used by Group A/B/C. `quarantine_resolution/current_quarantine.json` records the nine unresolved rows; `restoration.json` and `replacement_structural_check.json` record the ten repaired PubChem rows restored separately. Identity failure is not permission to invent a charge, bond, or stereochemical assignment.

Run the parameterized split stages documented in the parent [final pipeline](../FINAL_PIPELINE.md) against the verified snapshot. The historical scripts alone do not reconstruct missing original source archives.
