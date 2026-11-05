# Manuscript correspondence and reproducibility boundaries

The supplied `MetaDual_Manuscript.docx` is the reference for this repository's scientific name, title, authors, cohorts, experimental protocol and reported metrics. Its SHA-256 fingerprint and extracted claims are in `manuscript_reference.json`. The manuscript itself is not copied into the public repository.

## Verified correspondence

- Primary cohort: 4,012 individuals, 903 species features, 1,536 functional features, seven groups including healthy controls, fixed test count 799.
- Independent cohort: 6,025 individuals, 2,557 species features, 1,536 functional features, eight groups including healthy controls, fixed test count 1,202. The original filename says `2577`; the actual feature count is 2,557.
- `results/primary/Ours/metrics_summary.csv` has Macro-AUC 0.9812156157646383 and Macro-F1 0.854633620915503; `results/independent/Ours/metrics_summary.csv` has 0.9243738895187172 and 0.6995701552641542. These round to the manuscript values.
- Primary single-view archives have Macro-F1 0.3531472866 (abundance only) and 0.4194490344 (functional only). The no-adaptive-structure archive has Macro-F1 0.8102922519, matching the manuscript ablation description after rounding.
- Main network operations match Methods 4.4, including shared context attention, summed adaptive-affinity heads, retained fixed propagation powers 2 and 3, feature-wise scaling and output attention.

## Documentation and frozen-artifact policy

Public method documentation and new-run defaults follow the supplied manuscript: Top 50 species, validation-based candidate selection, and four/three/optional two-parent refinement. Existing feature arrays, response archives, prompt text and empirical scores are retained unchanged; this update does not regenerate data or run full experiments. The manuscript and the original source experiments are also unchanged.

Historical script settings and dataset provenance are retained in `CODE_PROVENANCE.md` and the machine-readable source/audit manifests. They are separate from the normative method description. Old empirical records are not presented as newly regenerated outputs. Portable cohort profiles remain source-derived defaults because full historical launch commands and random states are not recoverable from metric CSVs.

## Coverage

| Study component | Included |
| --- | --- |
| Main MetaDual training on both cohorts | Executable shared trainer and processed feature datasets |
| Functional-only, abundance-only, adaptive-structure removal | Executable variants and matching historical summary records |
| Two additional attention-removal ablations | Executable legacy variants and historical summary records |
| LLM generation, CLS functional profiles, diversity-aware prompt search | Executable pipeline; explicit manuscript/legacy settings; frozen history |
| Six baseline methods across three input views | Historical result CSVs; baseline training implementations are not included |
| Methods 4.5 functional-theme analysis and corresponding figures | Not bundled; corpus-specific vocabulary/rules are not reconstructed or invented |
| Methods 4.6 UMAP/similarity figures | Not bundled; no claim of full figure regeneration |
| Methods 4.7 feature attribution, candidate-marker statistics | Not bundled; no claim of full figure regeneration |
| End-to-end reproduction of manuscript scores from fresh API responses | Not performed; requires the authors' historical configuration/model identity and reconciliation of the discrepancies above |

All changes are local to the `MetaDual` checkout. Original source experiments, source datasets and the supplied manuscript are left unchanged. The previous export directory is not silently synchronized with this Git checkout.
