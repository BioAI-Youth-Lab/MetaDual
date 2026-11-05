# MetaDual method-to-code mapping

| Manuscript section | Implementation |
| --- | --- |
| 4.1 / 4.2: transductive multiclass task and repeated stratified splits | `model/data.py`, `model/splits.py`, `model/engine.py` |
| 4.3: species-only/abundance-only LLM input; no disease labels in task prompts | `data_processing/pipeline.py`, `prompt_search/evaluation.py` |
| 4.3: semantic-diversity ranking and four/three/optional two-parent aggregation | `prompt_search/search.py`; historical records remain separate |
| 4.3: five Function statements per block, 512-token truncation, CLS concatenation | `data_processing/embedding.py`, `data_processing/pipeline.py` |
| 4.4: separate Bray–Curtis fixed graphs | `model/data.py: build_knn_graph` |
| 4.4: shared cohort attention, view-specific projections, residual context | `model/layers.py: MultiHeadAttention, EncoderLayer1`; `model/networks.py` |
| 4.4: sum of learned affinity heads, positive threshold/top-r, symmetry and self-loops | `model/layers.py: adjcentmatrix` |
| 4.4: fixed graph powers 2 and 3, adaptive propagation, feature scaling, score attention | `model/networks.py: MetaDual` (alias of the historical full class) |
| 4.4: NLL, validation-based checkpoint selection, held-out predictions | `model/engine.py` |

With `K=3`, the loop assigns the second and third fixed-graph powers to the retained branches. The number `20` passed to the original kNN implementation includes the query point; skipping its first returned neighbor retains 19 candidate neighbors before symmetrization. The export preserves that source convention rather than silently changing the graph under the same parameter name.

Original attention head assembly and inactive parameters are retained for checkpoint compatibility. The prompt scorer has a separate original structural layer that retains its last affinity head, while the main MetaDual model sums all four. The manuscript describes the main MetaDual model, not that historical lightweight scorer. Exact comparison tests cover both.

Training profiles are portable source defaults. The manuscript provides the framework and experimental protocol but does not record every historical CLI override or GPU random state. Archived metrics are empirical records, not values asserted to follow deterministically from the configuration files.
