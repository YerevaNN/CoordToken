---
language:
- en
pretty_name: CoordToken conformer corpus
size_categories:
- 100M<n<1B
tags:
- chemistry
- molecular-conformations
- coordtoken
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train/*.csv.zst
  - split: validation
    path: data/validation/*.csv.zst
  - split: test
    path: data/test/*.csv.zst
---
# CoordToken conformer corpus

Final deduplicated CoordToken corpus used for tokenizer training and evaluation.
This replaces the historical 190,214,248-training-row release. It is the
CoordToken corpus, not the separate CASF/protomer-filtered dataset.

| Split | Conformer rows |
|---|---:|
| Training | 189,792,028 |
| Validation | 1,802,906 |
| Test | 5,553,421 |

Code and final data pipeline: [YerevaNN/CoordToken, release-cleanup](https://github.com/YerevaNN/CoordToken/tree/release-cleanup).
Qwen training and CoordToken encoding/decoding:
[YerevaNN/3DMolGen](https://github.com/YerevaNN/3DMolGen), with the pinned
[CoordToken integration](https://github.com/YerevaNN/3DMolGen/tree/910ad8ede95904650dc41431bd01553bb29b680c).

## Files and loading

CSV shards are compressed with Zstandard, with a header in each shard. Shards
preserve the original order; no filtering, numerical conversion, or new shuffle
is performed for publication. `release.json` records source and compressed-shard
SHA-256 hashes, sizes and counts. The source hashes match the original final
merged CSVs. To reconstruct a merged CSV byte for byte, decompress shards in
filename order and retain only the first shard's header.

```python
from datasets import load_dataset
rows = load_dataset("FilyaGeikyan/CoordToken-data", split="train", streaming=True)
row = next(iter(rows))
```

Schema: `name`, `enriched_text`, `dataset`, `source_name`, `source_file_id`,
`source_row_index`, `inchikey`, `test_subset`.

`enriched_text` stores the historical molecular graph representation with atom
coordinates as `<x,y,z>` annotations. These are coordinate-bearing inputs, not
learned integer CoordToken codes. Atom brackets are part of the historical
encoding and must be handled with the supplied parser, not interpreted directly
as ordinary SMILES. Original graph distinctions lost by this encoding cannot
always be recovered. `inchikey` is the full Standard InChIKey used during split
construction. Source and row fields preserve lineage; `test_subset` distinguishes
the nablaDFT test regimes.

## Splits and scope

Molecular identity filtering uses full Standard InChIKeys. NablaDFT structure
and scaffold test identities are excluded from training. Its conformation test
intentionally permits molecular identity overlap, consistent with that task.
Exact enriched-text duplicates are removed within and across the merged splits.
This is not a guarantee that all geometrically similar conformers are absent.

The mixed-corpus nablaDFT conformation-test population retains test labels from
multiple supplied benchmark sizes; it is not identical to the official full
benchmark partition. Use the official benchmark for direct comparisons requiring
that partition. GEOM-Revisited is a separate evaluation set and is not included
in these merged totals. It overlaps the GEOM test population and should not be
pooled as an independent additional source.

The 19 sources include sampled PubChem3D and ZINC collections, not their full
releases. Source identities and processing evidence are in `provenance/` and the
[final pipeline documentation](https://github.com/YerevaNN/CoordToken/blob/release-cleanup/data_collection/FINAL_PIPELINE.md).
This release preserves the collected data; it is not a fresh reconstruction of
all upstream datasets. Consult each original source's terms and citation
requirements; this collection does not replace those terms with a new blanket
license.
