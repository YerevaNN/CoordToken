# Dataset split construction

We group datasets into three categories. **Group A** contains datasets with predefined splits: {BindingMOAD, BindingNet-High, BindingNet-Low, BindingNet-Mid, CrossDocked2020, DAVIS-3D, HiQBind, Kinodata-3D, OMol25-bio-mols, Plinder, SAIR, SPINDR, ChEMBL3D, GEOM}. **Group B** contains datasets without predefined splits: {KIBA-3D, OMol25-small-mols, ZINC, PubChem3D}. **Group C** contains {∇²DFT}, for which only three test sets are predefined: *test conformations*, *test scaffolds*, and *test structures*.

We first processed Group A. We computed full Standard InChIKeys from decoded molecular strings and aggregated keys from all available Group A test sets, the ∇²DFT *test scaffolds* and *test structures* sets, and the GEOM-Revisited test split. We then removed from all training and validation sets samples whose InChIKeys overlapped with this aggregated test list. Next, if a training split contained InChIKeys appearing in the remaining validation sets, those samples were removed from training. We reserved the GEOM-Revisited test set in this way for evaluation.

Next, for Group B, we iteratively constructed new splits in the order KIBA-3D, OMol25-small-mols, PubChem3D, and ZINC. For each dataset, we initially targeted test and validation sizes of roughly 0.5% of the full dataset. Test candidates were selected as samples whose InChIKeys appeared in test sets of previously processed datasets or in the reserved test sets; validation candidates were selected analogously after excluding test-key matches. Training candidates were defined as samples whose InChIKeys did not appear in protected test or validation sets. When a candidate split exceeded its target size, we randomly subsampled whole InChIKey groups. Unselected holdout candidates were excluded from training. Otherwise, we filled the split from training candidates (already excluding protected test and validation matches), assigning all samples with the same InChIKey to the same split, while ensuring that added InChIKeys did not overlap with existing training sets of other datasets.

Finally, for Group C (∇²DFT), we retained molecular overlap only with the predefined *test conformations* set, while ensuring that its training split does not overlap with validation or protected test splits of Groups A and B, its own *test scaffolds* and *test structures* sets, or the GEOM-Revisited test set. Because ∇²DFT has no predefined validation split, we constructed validation to reach approximately 0.5% of the full ∇²DFT size. After removing test-key matches, we first selected samples whose InChIKeys appeared in existing validation sets, subsampling whole InChIKey groups if necessary. We then filled any remaining quota from the filtered training pool using only InChIKeys absent from all Group A and B training sets. All samples with the same InChIKey were assigned together, so split sizes could exceed the targets. We verified global train–validation, train–test, and validation–test molecular separation, with the conformation-test exception. Molecular overlap between training and the conformation test was intentionally allowed, since this test is intended to evaluate different conformations of the same molecules. For the scaffold test, we excluded matching InChIKeys from training and validation; we did not exclude other molecules solely because they shared a scaffold. All random selections during split construction used a fixed seed of 42.

## Filtered dataset sizes

Counts are retained CSV rows (samples/conformers), not unique molecules. The three ∇²DFT test sets are listed separately.

| Dataset | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| BindingMOAD | 20,529 | 129 | 468 |
| BindingNet-High | 210,818 | 127 | 3,729 |
| BindingNet-Low | 251,747 | 1,889 | 6,850 |
| BindingNet-Mid | 144,352 | 326 | 3,312 |
| CrossDocked2020 | 78,241 | 262 | 100 |
| DAVIS-3D | 129 | 7 | 225 |
| HiQBind | 24,518 | 48 | 279 |
| Kinodata-3D | 61,245 | 95 | 225 |
| OMol25-bio-mols | 1,344,106 | 35 | 225 |
| Plinder | 127,987 | 362 | 950 |
| SAIR | 1,521,900 | 99 | 225 |
| SPINDR | 26,671 | 56 | 225 |
| ChEMBL3D | 1,428,111 | 180,929 | 37,065 |
| GEOM | 5,099,357 | 689,655 | 711,546 |
| KIBA-3D | 260,129 | 1,533 | 1,576 |
| OMol25-small-mols | 4,033,245 | 21,597 | 37,384 |
| PubChem3D | 93,802,743 | 473,756 | 473,758 |
| ZINC | 74,401,034 | 375,763 | 375,763 |
| ∇²DFT | 8,689,383 | 63,340 | — |
| ∇²DFT — test conformations | — | — | 1,623,912 |
| ∇²DFT — test scaffolds | — | — | 1,130,328 |
| ∇²DFT — test structures | — | — | 1,152,833 |
| GEOM-Revisited | — | — | 23,404 |
| **Total (including GEOM-Revisited)** | **191,526,245** | **1,810,008** | **5,584,382** |

GEOM-Revisited is included in the evaluation total and remains a separate test file. The merged test CSV contains 5,560,978 rows; GEOM-Revisited contributes an additional 23,404 rows. A dash means that no corresponding split is present for that row. These counts precede any additional exact-sample deduplication.
