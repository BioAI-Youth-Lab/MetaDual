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
