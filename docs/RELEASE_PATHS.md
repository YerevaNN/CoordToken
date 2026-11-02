# Code, model and data locations

Verified against shared storage and remote branch heads on 2026-11-01.

## Language-model repository

Qwen training, tokenization, preprocessing and generation live in
**https://github.com/YerevaNN/3DMolGen**.

The published CoordToken integration is branch `coordtoken-integration-handoff`,
pinned commit `910ad8ede95904650dc41431bd01553bb29b680c`. Start with its
`COLLEAGUE_HANDOFF.md` and `docs/FSQ_TOKENIZER_MIGRATION.md`. This pin includes
the ring-label parsing fix. The broader repository continues to evolve;
the pin identifies the reproducible integration, not necessarily latest main.

Local checkouts:

- `/mnt/weka/fgeikyan/3dmolgen_bigdata` (origin `YerevaNN/3DMolGen`).
- Colleague's Qwen checkout: `/mnt/weka/vtarasov/code/3DMolGen`.
  Observed local commit `8c5bb57063c607354648ecc3b2565bab1b7fcbbe` on
  `big_data_infrastructure`; this records the local checkout, not a claim that
  this exact commit is pushed.

## Selected tokenizer checkpoint

`/mnt/weka/fgeikyan/CoordToken/artifacts/releases/coordtoken-atom12-reg1-mixed-epoch2-20261017/model.pt`

SHA-256: `29d487cf3783978c801c5a237e3c3c86cc23ab058aa7edea7eff0aa707c4bc4f`.
This is the selected two-epoch model on the CoordToken corpus. Its sibling
`release.json`, `source/`, and `result.json` document the artifact. A separate
CASF-filtered model exists and is not substituted here.

## Coordinate-bearing corpus

Root:
`/mnt/weka/fgeikyan/3dmolgen_bigdata/data_handoff/merged_deduplicated_20261006/`

| File | Rows |
|---|---:|
| `merged_train.csv` | 189,792,028 |
| `merged_val.csv` | 1,802,906 |
| `merged_test.csv` | 5,553,421 |

The files contain `enriched_text` with Cartesian coordinates, source provenance,
full InChIKeys and test regime labels. They are not integer-code sequences.
`merged_final` is an alias for this directory. The CSV order is not a global
shuffle; training uses seeded permutations. Hashes and checks are in
[FINAL_PIPELINE.md](../data_collection/FINAL_PIPELINE.md).

Hugging Face target: https://huggingface.co/datasets/FilyaGeikyan/CoordToken-data.
Compression is complete: 70 Zstandard CSV shards (67 train, one validation,
and two test), totaling 58,980,050,447 bytes (58.98 GB). All compressed-shard
SHA-256 hashes were rechecked on 2026-11-02. Source hashes, row counts and shard
metadata are recorded in [the package manifest](dataset_package_manifest.json).
The data have not been uploaded; public publication awaits explicit confirmation.
Local staging and progress:
`/mnt/weka/fgeikyan/CoordToken/artifacts/huggingface_release_20261101/`.
Only a `publication.json` with `status: published_verified` plus the matching
remote revision establishes upload completion.

## Already encoded Qwen data

Mixed-corpus root:
`/mnt/weka/fgeikyan/3dmolgen_bigdata/datasets/coordtoken_atom12_reg1_epoch2_20261018/`

Splits are `train/`, `val/`, `test/`. The release marker says `verified`; source
hashes and row counts match the final coordinate corpus above. See the
[copied receipt](coordtoken_atom12_reg1_epoch2_20261018_release.json).

Original GEOM export:
`/mnt/weka/fgeikyan/3dmolgen_bigdata/datasets/geom_coordtoken_core_e2_20261023/processed_strings/`

Splits are `train/`, `valid/`, `test/`, with 5,745,689 / 716,818 / 711,546 rows.
This uses the original GEOM Cartesian-v3 JSONLs, rather than extracting the GEOM
subset from the mixed corpus. Source:
`/data/fgeikyan/shared-with-me/vtarasov/geom_cartesian_v3/processed_strings/`.
The [release receipt](geom_coordtoken_core_e2_20261023_release.json) is verified
and `sft_ready: true`. Directory names retain historical labels for stable paths.

Qwen tokenizer:
`/mnt/weka/fgeikyan/CoordToken/research/coordtoken_encoding_20261019/source/src/molgen3D/training/tokenizers/Qwen3_tokenizer_coordtoken_atom12_reg1`

The molecular register and atom geometry codes use the same token alphabet.
These exports use the selected checkpoint above; do not combine them with the
older all-SMILES-token tokenizer or the separate CASF-filtered checkpoint.

## Reproduction scope

This branch combines the previously published model (`c9f65b6`) and final data
pipeline (`2b343b3`). Added packing, final Nabla and preprocessing files are
byte-for-byte local source snapshots; hashes are in
`release_source_additions_20261101.json`. Preprocessing launchers import
`molgen3D` from the separately linked integration checkout. Restore that checkout
at the expected `source/` path or configure equivalent paths before launching.
External caches, manifests, run selections and checkpoints are required by the
frozen launchers. No new training or full data reconstruction is claimed by this
code publication.
