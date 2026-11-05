# Archived manuscript experiment results

`primary/` and `independent/` contain unchanged CSV files copied from the original experiment result collections. `summary.csv` combines their mean metrics without modifying them. Source paths and SHA-256 values are recorded in `docs/manuscript_source_manifest.json`.

| Original directory | Method / input view |
| --- | --- |
| `Ours` | MetaDual, combined abundance and functional profiles |
| `Ours_nodeepseek` | Abundance-only MetaDual ablation |
| `Ours_noabundance` | Functional-only MetaDual ablation |
| `Random_Forest`, `MLP`, `Meta-Spec`, `GCN`, `GAT`, `Mixhop` | Six manuscript baselines using combined profiles |
| Suffix `_nodeepseek` | Abundance-only input |
| Suffix `_noabundance` | Functional-only input |
| `MetaDual_no_structure` | Additional primary adaptive-affinity-removal record |
| `MetaDual_no_modifiedt_1`, `MetaDual_no_modifiedt_2` | Additional primary attention-removal records |

The original CSV column `Fold` is retained for provenance; these experiments use repeated runs over one fixed test set. The full MetaDual means round to 0.9812/0.8546 (primary Macro-AUC/Macro-F1) and 0.9244/0.6996 (independent). Counts, schemas and caveats are in `docs/MANUSCRIPT_ALIGNMENT.md`.

These are historical empirical outputs. The repository preparation did not rerun full-length experiments. Baseline result presence does not mean baseline training scripts or every manuscript figure-generation pipeline are bundled.
