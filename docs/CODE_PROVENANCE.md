# Code provenance

All source paths below are relative to the original `LLM` workspace. Exact hashes are recorded in `source_manifest.json`. New modules remove inactive imports, commented experiments and absolute machine paths; they do not replace the source files.

| Export component | Original implementation |
| --- | --- |
| Abundance preparation | `data_gmhi/process_4347_data.py` |
| FCG generation and actual data prompt | `data_gmhi/run_analysis_4347.py` |
| Function-batch CLS embeddings | `data_gmhi/extract_embeddings_768_top100_random_5text_search.py` |
| Final two-vector merge | `data_gmhi/concat_mult_one_hot_100_deepseek_768_one_5_no_targe.py` |
| Main training and three variants | The four user-selected `train_only_otu_info_mult_disease_200_deepseek_two_part_new_graph_got_search_*.py` scripts under `GCN_version_nature_data_gmhi/GCN-SA-main/GCN-SA/pygcn/` |
| Main networks, layers and splits | `models.py`, `layers.py`, `distcalculate.py`, `utils.py`, and the final dual-graph loader in `utils_data.py` in that same directory |
| Generation/aggregation templates and search rules | `comp/got/prompt1/GCN-SA/manual_prompt_search_agg_no.ipynb`, active template cell 6 and first/second aggregation cells |
| Prompt parsing and 200-sample feature construction | `comp/got/prompt1/GCN-SA/microbiome_prompt_search_from_no_comp_mult_disease.py` |
| Prompt scoring parameters and pooled AUC | `comp/got/prompt1/GCN-SA/utill_auc_mult_disease.py` |
| Prompt scorer's distinct network dependencies | `comp/got/prompt1/GCN-SA/pygcn/models.py`, `layers.py`, and `utils_data.py` |

The shared training engine preserves the original main defaults and accepts the separate prompt-scoring profile. Device-aware allocation replaces hardcoded CUDA tensor creation; an unused preallocated identity matrix is omitted. Learned parameter names and shapes remain unchanged. Evaluation uses `torch.no_grad()`, outputs use unique directories, and checkpoints are replaced atomically.

The parser is shared by the online generation commands and was extracted from the original prompt-evaluation helper. Existing parsed FCG records are copied unchanged. Explicit validation rejects malformed inputs; new prompt evaluation failures propagate instead of being silently assigned a valid zero score. Historical data and scores are never recomputed during archive inspection.

## Manuscript-aligned checkout

The paper-facing framework is named MetaDual. The two single-view classes (`GCNSA_two_graph_nodeepseek`, `GCNSA_two_graph_only_text`) were extracted from the same original models file for `abundance_only` and `function_only`. The full architecture and its parameter names were retained; `MetaDual` is an alias, not a different network.

`configs/` identifies source-derived defaults for each cohort. `docs/manuscript_source_manifest.json` fingerprints the newly included independent dataset, extracted class source, pair-refinement logs, original independent trainer and archived result CSVs. `docs/manuscript_reference.json` fingerprints the supplied manuscript.

Changes beyond packaging are explicit: the prompt CLI's manuscript preset uses Top 50/validation evaluation, and the optional pair stage uses two-parent combinations. Legacy Top 100/test scoring remains available. Main cohort evaluation remains on a fixed test set. The original exported four-network computations are unchanged and continue to be compared numerically with the source.

## Historical artifact settings retained for traceability

The public method specification and new default evaluator follow the manuscript Top 50 input and validation-based prompt selection. The source scripts that produced the supplied archival pipeline used `TOP_K_OTUS=100`; their prompt helper scored pooled predictions from its own fixed test split. The source alias was `deepseek-chat`, while the manuscript reports DeepSeek-V3.2. These facts are retained as provenance and do not constitute evidence of a newly regenerated Top 50/validation-score archive. `--preset legacy` explicitly retains those historical input-size/split choices if needed. Existing NPZ values and prompt scores were not edited.

The initially selected July 30 primary output directory records Macro-AUC 0.9848111 and Macro-F1 0.8859113, whereas the manuscript-matching `mult_result/Ours` archive records 0.9812156 and 0.8546336. The matching archive was found and copied unchanged; the two source folders are not asserted to be the same run.

The pair-refinement archive records zero accepted candidates and a stale round-2 baseline of 0.9208666667. Its raw files are preserved separately from the canonical 30/15/8 ranking, whose best score is 0.9338666667.
