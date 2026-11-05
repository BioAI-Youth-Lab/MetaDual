# Prompt search and historical integrity

## Archived results

The three final ranked JSON files in `prompt_search/history/` are the authority for the 30 initial, 15 first-round, and 8 second-round prompts. Text, hashes, numeric scores, original IDs, and parent metadata are preserved byte-for-byte in those files. The 53 matching individual log files are retained under `history/individual/`.

There is one substantive score discrepancy: prompt `8301d8b1` has ranked score **0.9030333333** and individual-log score **0.9054333333333333**. No replacement score was inferred. The machine-readable audit is `docs/prompt_history_audit.json`.

The raw first-round kept-run log has 20 candidates; the source notebook subsequently removed five manually, leaving the final 15. Both final rankings and relevant raw attempt/combination records are retained to explain this history. New code uses a target of 15 and does not replay hand-selected deletions or refill cells. It therefore preserves the method and published archive, not the exact interactive history. The second-round final archive contains eight prompts. Third-round experiments and duplicate `_yuan` exports are omitted.

Raw attempt IDs refer to the IDs used at generation time. Manual renumbering in the original notebook means these IDs are historical references, not a newly certified genealogy. New searches additionally store parent hashes to make their lineage unambiguous.

## Templates and feature construction

`prompts/generation_template.txt` is the active generation template extracted from notebook cell 6. `prompts/aggregation_template_example.txt` shows the active no-extra-task-description aggregation template; executable templates live in `templates.py`.

`prompts/best_prompt.txt` is the best final historical prompt (`168f2812`, score 0.9338666667). `prompts/data_generation_prompt.txt` is extracted from `run_analysis_4347.py`, with its OTU-list placeholder preserved and Top 100 wording. It corresponds to the best prompt wording with the original data script's Top 100 adjustment. The script's output basename contained `top40` despite `TOP_K_OTUS=100`; the cleaned name avoids that stale filename.

Prompt selection embeddings use the notebook's mean of token hidden states **including padding**, with batch size four in round 1 and three in round 2. This operation is preserved. In contrast, per-sample FCG features use the CLS token. Final-data feature preparation joins five functions with **spaces**; the original search evaluator joins five functions with **newlines**. These separate original behaviors are retained.

## Evaluation profile

The prompt CSV is a separate cohort from the final GMHI training dataset. Selection uses the four most frequent disease labels, up to 50 samples each, sampled using NumPy seed 42. Disease labels are uppercased; the first 277 rows are treated as OTUs. Age, BMI, and gender are saved in the temporary NPZ for compatibility but are not used by the model.

The task model receives the top 100 nonzero OTUs even when a template says Top 50. It must return at least ten parseable FCGs. Each sample has five attempts; if fewer than ten samples fail, those failures receive a second batch of five attempts. Remaining failures invalidate the prompt with score -1. Original zero-padding behavior for embedding failures is retained and such failures are reported.

The scorer uses cosine kNN graphs (`k=5`), hidden size 64, dropout 0.55, learning rate 0.02, weight decay 0.001, epsilon 0.9, `r=3`, `K=3`, 2,000 epochs, patience 100, and five repeated train/validation splits over a fixed test set. Its score is the macro OvR AUC over pooled predictions from those runs. The original accuracy fallback is recorded explicitly if AUC is undefined. It is not a 25-fold or five-fold cross-validation procedure, despite stale comments in the source.

The source prompt structural layer overwrote the structural similarity at each head, retaining the last head; the main model accumulates all four heads. Both behaviors and their respective normalization formulas remain separate in the export. Original attention layout, unused parameters, and checkpoint parameter names are preserved.

New searches cache scores only within their newly created run directory. Credentials are loaded from environment variables, and exceptions from model/scoring infrastructure stop the new run rather than silently producing a valid score of zero. Archive viewing does not instantiate a client or load BiomedBERT.
