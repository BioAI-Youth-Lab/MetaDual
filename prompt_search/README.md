# Prompt search

From the repository root, inspect the archive without API calls:

```bash
python -m prompt_search.search history
```

The clean notebook defaults to offline history inspection. Its online cells are gated by `RUN_ONLINE = False`. Starting a new search requires explicit execution:

```bash
export DEEPSEEK_API_KEY='YOUR_KEY'
python -m prompt_search.search run --embed-model /path/to/BiomedBERT --device cuda
```

The fresh search targets 30 → 15 → 8 and reports partial results when attempt budgets do not reach those targets. It does not delete historical records, reuse their scores for new generated prompts, or refill duplicates. The original prompt-scoring network and cosine graph profile are separate from the main training profile. See `docs/PROMPTS.md` for archived score discrepancies, aggregation rules, and Top 50/Top 100 behavior.
