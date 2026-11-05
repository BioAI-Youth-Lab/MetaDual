# MetaDual cohorts and feature schema

## Primary cohort (Methods 4.2)

`data_processing/data/final_features.npz` is an unchanged copy of the final source NPZ, with 4,012 individuals across six disease groups and healthy controls. The source abundance CSV has 4,347 individuals; the >=100-per-class filter retains 4,012.

| NPZ key | Shape | dtype |
| --- | --- | --- |
| `sample_ids` | `(4012,)` | Unicode |
| `otu_names` | `(903,)` | Unicode |
| `X_otu_abundance` | `(4012, 903)` | float64 |
| `y` | `(4012, 7)` | float64, one-hot |
| `X_text_embedding` | `(4012, 1536)` | float32 |

| Index | Group | Individuals |
| ---: | --- | ---: |
| 0 | ACVD | 152 |
| 1 | CRC | 298 |
| 2 | Crohns disease (CD) | 108 |
| 3 | Healthy | 2636 |
| 4 | Obesity | 342 |
| 5 | Overweight | 238 |
| 6 | T2D | 238 |

The fixed test set contains 799 individuals. `4347_final_relative_abundances.csv` has species in rows and individuals in columns. `fcg_results.jsonl` contains 4,347 parsed response records with `sample_id` and `parsed_response.Functional_Collaboration_Groups`; each group contains `OTUs` and `Function`. Archived records contain 10–15 groups.

## Independent cohort (Methods 4.2, Results 2.6)

`data_processing/data/independent_features.npz` contains 6,025 individuals and is a losslessly compressed version of the original 161,861,028-byte NPZ. Every original array, sample order and dtype was checked for identity. Actual species feature count is 2,557 despite the source filename containing `2577`.

| NPZ key | Shape | dtype |
| --- | --- | --- |
| `sample_ids` | `(6025,)` | Unicode |
| `otu_names` | `(2557,)` | Unicode |
| `X_otu_abundance` | `(6025, 2557)` | float64 |
| `X_info_raw` | `(6025, 3)` | float64; compatibility metadata, not a model input |
| `y` | `(6025, 8)` | float64, one-hot |
| `X_text_embedding` | `(6025, 1536)` | float32 |

| Index | Category | Individuals |
| ---: | --- | ---: |
| 0 | Cancer | 312 |
| 1 | Cardiovascular disease | 560 |
| 2 | Digestive disease | 433 |
| 3 | Healthy | 2613 |
| 4 | Immune disease | 762 |
| 5 | Infectious disease | 283 |
| 6 | Kidney disease | 206 |
| 7 | Metabolic disease | 856 |

The fixed test set contains 1,202 individuals. Training is performed within this independent cohort; it is not zero-shot transfer from the primary cohort. The processed NPZ is bundled; its original raw abundance table and per-individual FCG responses are not bundled.

## Shared functional features

Each view describes the same ordered individuals. Functional descriptions are grouped in batches of five, truncated to 512 tokens, and encoded with BiomedBERT CLS representations. The first two 768-dimensional vectors are concatenated. Features are computed before training and remain fixed. Demographic metadata are not used in MetaDual's two input views.

Methods 4.3 specifies Top 50 nonzero species as the input community profile; the generator and prompt evaluator use that setting by default. The bundled precomputed arrays remain unchanged and are not regenerated as part of this documentation update. Source-script parameters are preserved separately in `CODE_PROVENANCE.md` and the source manifests. Missing vectors in the original merger are padded with zeros; the supplied primary final NPZ has no all-zero text rows.

## Prompt-selection data

`data_processing/data/prompt_validation.csv` is the unchanged original `filtered_otu.csv`: 6,462 individuals, 277 selected OTU rows and four metadata rows. Its byte-identical copies originally existed in `data/` and `data_gmhi/`. The prompt scorer selects the four most frequent disease categories and up to 50 individuals per category using seed 42, producing 200 samples. It is not a 200-person subset defined by either final cohort's fixed test indices.

Other cohort versions, transformed feature variants, redundant intermediate NPZs, large reproducible embedding JSONLs, caches and training checkpoints are excluded. The ready-to-train NPZs let readers train without LLM requests or BiomedBERT inference.
