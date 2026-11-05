# MetaDual diversity-aware prompt development

## Manuscript method (Methods 4.3)

A meta-prompt stochastically generates 30 candidate prompts. Each candidate is embedded in BiomedBERT semantic space. Candidate subsets are ranked by the sum of pairwise cosine distances, and semantically diverse groups are fused and evaluated. Four-parent and three-parent aggregation stages target 15 and 8 improved prompts. The optional two-parent stage uses up to 28 combinations, temperature 0.2 and target 3 improvements. A candidate must outperform the preceding stage's best score. Duplicates are recorded without refill loops and partial counts are allowed.

For individual-level application, the selected prompt receives the Top 50 species with positive relative abundance and their abundance values. Disease labels are excluded from the task input. It requests at least ten FCGs, each containing two to five species and a concise Function statement. The first ten Function statements are grouped into batches of five; BiomedBERT CLS vectors are concatenated to produce 1,536-dimensional profiles. The manuscript reports DeepSeek-V3.2; the provider's requested model identifier is configured through `DEEPSEEK_MODEL`.

The default CLI and `EvaluationConfig` both use **Top 50** and **validation scoring**. New candidates are ranked by pooled validation predictions from the selected checkpoints. Main-cohort test evaluation remains separate.

## Archived prompts and results

The three final ranked JSON files under `prompt_search/history/` are preserved byte-for-byte, containing 30 initial, 15 first-round and 8 second-round prompts. Their text, hashes, IDs, scores and parent metadata are unchanged. The selected prompt is `168f2812`, with recorded score 0.9338666667, exported as `prompts/best_prompt.txt`.

The 53 matching individual log files are in `history/individual/`. Prompt `8301d8b1` has final-ranked score 0.9030333333 and individual-log score 0.9054333333333333. Both observations are retained; neither was inferred or overwritten. `docs/prompt_history_audit.json` records the discrepancy.

The raw first-round log contains 20 candidates before the original notebook manually removed five; the final ranking contains 15. New code does not replay manual deletion/refill cells. The archived two-parent stage accepted zero candidates and includes a stale baseline, so it is retained in a separate `history/pair_refinement/` directory. No successful new third-stage result is asserted. Historical generation-time IDs can reflect later manual renumbering; new runs additionally save parent hashes.

## Scoring implementation

The scoring workflow reads `prompt_validation.csv`, selects the four most frequent categories and up to 50 individuals per category using seed 42, and uses the first 277 species rows. Metadata are saved for compatibility but are not model inputs. Each task response needs at least ten parseable FCGs. Each sample receives five attempts, with a second batch of five only if fewer than ten samples failed. Remaining failures invalidate the candidate. Embedding failures are reported and use the original zero-padding behavior.

The lightweight scorer uses cosine kNN graphs (k=5), hidden size 64, dropout 0.55, learning rate 0.02, weight decay 0.001, epsilon 0.9, r=3, K=3, up to 2,000 epochs, patience 100 and five repeated splits. Candidate scores use validation predictions by default. Its original structural-layer implementation is kept separate from the main MetaDual model. Semantic prompt selection uses the notebook's mean over all token positions including padding, with batch sizes four, three and two for the respective stages. FCG features use CLS vectors; final-data batches join functions with spaces, while the historical scorer joins functions with newlines.

New outputs are isolated by timestamp and per-prompt hash. Cache reuse is limited to a new run directory. Credentials come from environment variables; offline archive inspection loads neither a client nor the text model.

## Frozen input provenance

The supplied data, prompt rankings and scores are retained rather than regenerated. Low-level historical input-size, scoring-split and API-alias details are recorded in `docs/CODE_PROVENANCE.md`, `docs/manuscript_source_manifest.json` and `docs/prompt_history_audit.json`. These records describe the origin of existing artifacts; they do not claim a new rerun under the default manuscript preset.
