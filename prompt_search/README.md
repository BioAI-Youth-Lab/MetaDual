# MetaDual diversity-aware prompt development

Inspect the frozen 30 → 15 → 8 archive offline:

```bash
python -m prompt_search.search history
```

The notebook is `prompt_search_agg_no.ipynb`; its online cells are disabled by default. New manuscript-style runs use Top 50 input and validation scoring:

```bash
export DEEPSEEK_API_KEY='YOUR_KEY'
python -m prompt_search.search run --preset manuscript --pair-refinement \
  --embed-model /path/to/BiomedBERT --device cuda
```

Four- and three-parent aggregation stages target 15 and 8 improved prompts. `--pair-refinement` optionally adds the manuscript's two-parent stage, with up to 28 combinations and target 3. Partial counts and zero improvements are allowed. The main historical ranking remains 30/15/8.

The default evaluator uses Top 50 species and validation predictions, following the manuscript. Its lightweight cosine graph/model profile is separate from the main MetaDual Bray–Curtis profile. Archived prompts and scores are unchanged. See `docs/PROMPTS.md` for the method and `docs/CODE_PROVENANCE.md` for historical source settings.
