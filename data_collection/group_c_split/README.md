# Final nabla split and global molecular separation

This step starts from the larger verified grouped nabla training pool and the
completed Group A/B splits. It reuses the already computed full Standard
InChIKeys. It does not recollect or re-encode molecular data.

1. Exclude nabla training-pool keys matching protected tests: all Group A/B
   tests, nabla structure/scaffold tests, and GEOM-revisited. The conformation
   test is exempt from molecular exclusion.
2. Among the remaining keys, prefer existing validation matches for nabla
   validation. If these exceed the target, subsample whole key groups and keep
   unselected validation candidates excluded from training.
3. Fill the remaining quota only with nabla training keys absent from **every
   retained Group A/B training set**. The target is 0.5% of the surviving full
   nabla snapshot (training pool plus all three tests): 63,325 rows. Whole groups
   can overshoot. A shortfall is reported; other training rows are never removed
   to fill the quota.
4. Write and verify nabla train/validation, and copy the three existing test
   files byte-for-byte. Verify global molecular separation and reread all A/B
   output hashes, preserving their completed splits unchanged.

Seed 42 determines a reproducible pseudorandom SHA-256 ordering of molecular
groups. All conformers of a key stay together. Scaffold-test protection means
matching molecular keys, not a scaffold-wide blacklist. Molecular identity is
decoded from the stored strings and retains the documented representation limits.

The controller queries the prior ownership registry for test/validation
priority, then explicitly checks retained A/B training memberships to determine
eligible fill candidates. Tests cover test priority, validation quota excess,
insufficient eligible keys, preserving other training records, and a complete
synthetic A/B/C run with independent output validation.

```bash
python -m unittest discover -s data_collection/group_c_split -p 'test_*.py' -v

python data_collection/group_c_split/prepare.py \
  --preflight /path/to/resplit_20261005 \
  --group-a /path/to/group-a-run \
  --group-b /path/to/group-b-split-run \
  --index-run /path/to/group-b-index-run \
  --audit /path/to/completed-independent-holdout-verification \
  --output /path/to/new-group-c-experiment/contract.json

python data_collection/group_c_split/run_nabla.py \
  --contract /path/to/new-group-c-experiment/contract.json \
  --output /path/to/new-group-c-experiment/run --workers 32
```

`run.sbatch CODE_DIRECTORY CONTRACT NEW_OUTPUT_DIRECTORY` provides the cluster
launcher (32 CPU cores, 128 GB; adjust site settings). Output directories must
be new. The contract pins code and prior evidence, including the previous
ownership database. The run never modifies existing sources or A/B outputs.

Outputs include `selection.json`, `validation_keys.json`, the complete per-key
`nablaDFT.sqlite`, kept/excluded chunks with reasons, five candidate CSVs,
`report.json`, and `combined_split_manifest.json`. The combined manifest points
to all 59 A/B/C split files and the separate GEOM-revisited reference. It records
the conformation-test exception. No exact-sample deduplication or merged training
CSV is implied by completion of this molecular splitting step.
