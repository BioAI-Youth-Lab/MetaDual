# Data files and schema

| Included file | Origin / purpose |
| --- | --- |
| `data_processing/data/4347_final_relative_abundances.csv` | Original 903 × 4,347 abundance table, OTUs in rows |
| `data_processing/data/fcg_results.jsonl` | Archived parsed FCGs for 4,347 samples |
| `data_processing/data/final_features.npz` | Byte-for-byte copy of the final NPZ used by the four requested training scripts |
| `data_processing/data/prompt_validation.csv` | Separate 277-OTU dataset plus metadata rows, 6,462 samples; used only for prompt scoring |

The two original copies of `filtered_otu.csv` in `data/` and `data_gmhi/` are byte-identical. Only one is included.

## Final NPZ

| Key | Shape | dtype | Meaning |
| --- | --- | --- | --- |
| `sample_ids` | `(4012,)` | Unicode | Ordered sample identifiers |
| `otu_names` | `(903,)` | Unicode | Ordered OTU feature names |
| `X_otu_abundance` | `(4012, 903)` | float64 | Per-sample relative abundance |
| `y` | `(4012, 7)` | float64 | One-hot disease labels |
| `X_text_embedding` | `(4012, 1536)` | float32 | First two 768-dimensional FCG batch vectors |

| Class index | Label | Samples |
| ---: | --- | ---: |
| 0 | ACVD | 152 |
| 1 | CRC | 298 |
| 2 | Crohns disease | 108 |
| 3 | Healthy | 2636 |
| 4 | Obesity | 342 |
| 5 | Overweight | 238 |
| 6 | T2D | 238 |

Labels are derived by removing the last underscore-delimited sample number and ordered using the original one-hot encoder's sorted categories. The 4,347-sample source includes 13 categories; the >=100 filter retains the seven above.

Each archived FCG JSONL record contains `sample_id` and `parsed_response.Functional_Collaboration_Groups`, whose items have `OTUs` and `Function`. Archived records contain 10–15 groups. Every five nonempty function descriptions are joined with a space, embedded with BiomedBERT, and the first two vectors are retained. Missing sample vectors or missing second vectors are padded with zeros by the merger. The supplied final NPZ has no zero text rows and no non-finite values.

## Excluded intermediates

Other cohort versions, row-sum/softmax-normalized variants, duplicate processed NPZ files, disease-description embeddings, analysis figures, temporary logs, and the 162 MB embedding JSONL are excluded. The latter is reproducible from the archived FCGs and the original BiomedBERT model files. The final NPZ is preserved unchanged so training does not require regenerating those embeddings.

The export contains processed inputs supplied in the workspace. It does not reconstruct upstream acquisition or merge steps that produced the abundance CSV.
