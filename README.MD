# CoordToken

CoordToken represents molecular geometry using one 12-bit iFSQ code per atom
and one molecular register code. Structural SMILES tokens remain graph context.
The molecular code precedes the graph; an atom code follows each atom. All
geometry codes share a 4,096-entry vocabulary. There is no separate register marker.

This release branch brings the final tokenizer and data pipeline together.
Frozen experiment paths are preserved so existing configurations remain usable.

| Purpose | Start here |
|---|---|
| Model architecture, training, loading and evaluation | [Updated tokenizer](docs/UPDATED_TOKENIZER.md) |
| Final corpus construction, cleaning and splitting | [Final data pipeline](data_collection/FINAL_PIPELINE.md) |
| Dataset, checkpoint and downstream data locations | [Release paths](docs/RELEASE_PATHS.md) |
| Qwen language-model training and generation | [YerevaNN/3DMolGen](https://github.com/YerevaNN/3DMolGen) |
| Pinned encoder/decoder and Qwen integration | [3DMolGen integration](https://github.com/YerevaNN/3DMolGen/tree/910ad8ede95904650dc41431bd01553bb29b680c) |
| Dataset publication format and scope | [Dataset card](docs/DATASET_CARD.md) |

## Final data

The corpus contains **189,792,028 training**, **1,802,906 validation**, and
**5,553,421 test** conformers. The authoritative local input is
`/mnt/weka/fgeikyan/3dmolgen_bigdata/data_handoff/merged_deduplicated_20261006/`.
Use this final deduplicated release, not the intermediate `merged_20261006`.

Hugging Face destination: [FilyaGeikyan/CoordToken-data](https://huggingface.co/datasets/FilyaGeikyan/CoordToken-data).
**Publication is pending; this link is not yet a claim that the files are available.**
The replacement contains coordinate-bearing CSV shards and their provenance,
not learned integer-code exports. Existing local Qwen exports are listed in
[Release paths](docs/RELEASE_PATHS.md).

## Training and evaluation code

- `research/full_release_20261008/`: final-corpus packing and seeded indices.
- `research/final_and_nabla_20261012/`: mixed-corpus training and selection.
- `research/mixed_epoch2_20261014_v3/`: selected second-epoch continuation.
- `research/mixed_epoch2_test_20261015/`: whole-test reconstruction evaluation.
- `research/nabla_b200_20261023/`: final shared-validation nablaDFT training
  and evaluation implementation; endpoint schedule is 3, 6, 12 and 22 epochs.
- `research/coordtoken_encoding_20261019/` and
  `research/geom_coordtoken_core_e2_20261023/`: production preprocessing launchers.
- `data_collection/`: cleaning, splitting, deduplication and source evidence.

The root `train.py`, `fsq.py` and their launchers are the earlier implementation;
the updated model is in the frozen `research/` directories above. The new
checkpoint is a PyTorch `model.pt`, not the historical Lightning `tokenizer.ckpt`.
Do not mix their loaders or integer codes.

Frozen launchers retain cluster-specific paths and require the external files
listed in their configurations. They are execution records, not a portable
one-command rerun. See the [training guide](docs/UPDATED_TOKENIZER.md) for
dependencies and configuration. The [3DMolGen integration guide](https://github.com/YerevaNN/3DMolGen/blob/910ad8ede95904650dc41431bd01553bb29b680c/docs/FSQ_TOKENIZER_MIGRATION.md)
contains encoder/decoder and Qwen preparation/training/inference commands.

## Publication helper

`scripts/publish_dataset.py` packages the final CSVs without changing rows,
verifies their full SHA-256 hashes and counts, and supports resumable uploads.
`--package-only` performs local compression and verification without network
publication. It requires `huggingface_hub` and `zstandard`. A dataset card must
be prepared before upload. Source files remain read-only throughout.

Weights, datasets, environments, local logs and paper archives are not stored
in Git. Earlier collection scripts remain documented as historical inputs,
not as the final pipeline.
