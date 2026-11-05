# GMHI GCN-SA with Functional Collaboration Groups

A curated research workflow for gut microbiome classification using OTU relative abundance and LLM-derived Functional Collaboration Group (FCG) embeddings. It contains the final dataset, data preparation code, a dual-graph GCN-SA model with three ablations, and the archived **30 → 15 → 8** prompt search.

This export was assembled from existing experiments. Historical scores are preserved observations, not results of a fresh training run. See [verification](docs/VERIFICATION.md) for the checks performed during preparation.

## Repository structure

```text
data_processing/       Data preparation, embedding utilities, and required data
model/                 Shared training entry point and four model variants
prompt_search/         Clean notebook, search/scoring code, prompts, and history
docs/                  Data schema, provenance, environment, and verification
tools/                 Offline verification and source integrity checks
verification/          Small verification reports; temporary work is ignored
```

## Installation

The source environment is `deepseek_env_got`: Python 3.10.16, PyTorch 2.7.0 with CUDA 12.6, NumPy 1.26.4, SciPy 1.15.3, pandas 2.2.3, scikit-learn 1.6.1, and Transformers 4.51.3. Exact observed versions are recorded in [the environment snapshot](docs/environment_snapshot.json).

For the existing environment:

```bash
conda activate deepseek_env_got
cd /path/to/GMHI_GCN_SA_GitHub
```

On another machine, create the environment from the repository root:

```bash
conda env create -f environment.yml
conda activate deepseek_env_got
```

Alternatively, use Python 3.10 and install `requirements.txt`, or `requirements-notebook.txt` for notebook support. The environment file pins direct Python dependencies; it is not a platform-specific binary lockfile. Install the PyTorch build appropriate for your GPU and driver if the default wheel is unsuitable. Full-graph attention and structure learning use substantial memory; CPU support is intended primarily for small checks.

All commands below run from the repository root. Model training uses the included final NPZ and needs no API credentials or text-model download.

## Included data

The final dataset contains **4,012 samples**, **903 OTU features**, **1,536 text features**, and **7 classes**. The original abundance table has 4,347 samples. Classes with fewer than 100 samples were excluded before training; the original filename retained `4347` despite the smaller final dataset.

The export includes the original abundance CSV, parsed FCG responses, the final NPZ, and the separate prompt-evaluation CSV. Large, reproducible embedding JSONL files and unrelated intermediate datasets are excluded. See [data schema and class counts](docs/DATA.md).

## Train the model

```bash
python -m model.train --variant full --device cuda
```

The defaults preserve the main source experiment: Bray–Curtis graphs with `k=20`, hidden size 768, `K=3`, learning rate 0.02, weight decay 0.0005, dropout 0.5, epsilon 0.9, `r=3`, up to 3,000 epochs, and patience 300. Use `python -m model.train --help` to inspect all options, including `--data`, `--output-dir`, and `--repeats`.

| Variant | Behavior |
| --- | --- |
| `full` | Original dual-graph model |
| `no_modifiedt_1` | Removes the feature-level `modifiedt` blocks; retains output attention |
| `no_modifiedt_2` | Removes both feature-level and output attention blocks |
| `no_structure` | Removes learned structural branches; retains fixed-graph propagation and attention |

Run all four sequentially:

```bash
bash model/run_all.sh --device cuda
```

The protocol fixes a stratified 20% test set using seed 42, then makes five train/validation splits of the remaining samples with seeds 0–4 and a 75% training fraction. This is approximately **60/20/20** after class-wise rounding. These are repeated runs on the same test set, not five-fold cross-validation. Graphs and attention span all samples, so the protocol is transductive.

Each invocation creates a unique directory under `outputs/model/<variant>/`. Each run stores its best checkpoint, splits, training history, predictions, sample IDs, probabilities, and metrics. Overall outputs include per-run and per-class metrics, confusion matrices, and a mean/std summary. CSV summary standard deviations use the sample convention (`ddof=1`); `summary.json` also records the population accuracy standard deviation used in the original console summary.

## Rebuild the features

The archived NPZ is ready to use. To regenerate its inputs without new API requests:

```bash
python -m data_processing.pipeline prepare
python -m data_processing.pipeline embed --embed-model /path/to/BiomedBERT --device cpu
python -m data_processing.pipeline merge
```

Preparation normalizes abundances by sample, derives labels from sample IDs, and keeps classes with at least 100 samples. Embedding joins each five consecutive FCG function descriptions with spaces and takes the BiomedBERT CLS vector. Merging concatenates the first two 768-dimensional vectors, padding missing vectors with zeros as in the source code. No disease-description embedding is used.

Generated files go to `outputs/data/`. Commands refuse to overwrite an existing output. Each subcommand accepts input/output paths; `prepare --min-class-count 0` retains all original samples. Text-model weights are not bundled. `--embed-model` accepts a local model directory or model identifier; a non-local identifier may download weights.

To explicitly regenerate FCGs through DeepSeek:

```bash
export DEEPSEEK_API_KEY='YOUR_KEY'
python -m data_processing.pipeline generate
```

The command uses the extracted data-generation prompt, the top 100 nonzero OTUs, temperature 0.2, up to 20 workers, and five attempts per sample. It reports any failed samples. This is an online, paid API operation; it is unnecessary for training or using the archived FCGs. Environment variables are read directly; `.env.example` is a reference and is not automatically loaded.

## Inspect the prompt search

```bash
python -m prompt_search.search history
```

This command reads the three final ranking JSON files offline. The archived counts and best scores are:

| Stage | Prompts | Best recorded score |
| --- | ---: | ---: |
| Initial | 30 | 0.8658 |
| First aggregation | 15 | 0.9030333333 |
| Second aggregation | 8 | 0.9338666667 |

The best final prompt has hash `168f2812` and is exported in [best_prompt.txt](prompt_search/prompts/best_prompt.txt). The actual data-generation template is separately saved in [data_generation_prompt.txt](prompt_search/prompts/data_generation_prompt.txt): its wording uses **Top 100**, whereas the search template and historical best prompt say **Top 50**. The source prompt evaluator actually supplies up to 100 OTUs; this discrepancy is retained and documented.

The clean [notebook](prompt_search/manual_prompt_search_agg_no.ipynb) displays the archive and exposes generation, evaluation, and the two aggregation stages without duplicate definitions, manual delete/refill cells, or third-round experiments. Online cells are disabled by default.

To explicitly start a new search:

```bash
export DEEPSEEK_API_KEY='YOUR_KEY'
python -m prompt_search.search run --embed-model /path/to/BiomedBERT --device cuda
```

The search makes 30 initial generation attempts, selects four-parent combinations for the first aggregation and three-parent combinations for the second, and targets 15 then 8 successful candidates. Parent combinations are ranked by summed pairwise BiomedBERT cosine distance. Only candidates scoring strictly above the preceding stage's best score are retained. Duplicate generations are recorded without refill loops. If attempt limits prevent a target count, the output reports the actual count and a partial status.

Prompt evaluation uses its original **separate profile**: the top four disease categories, up to 50 samples each, the first 277 OTUs, cosine graphs with `k=5`, hidden size 64, dropout 0.55, weight decay 0.001, 2,000 epochs, patience 100, and five repeated runs. Its original structural-layer implementation is preserved separately from the main model. Search generation temperatures are 1.5, 1.0, and 0.5; per-sample FCG evaluation uses 0.7. See [prompt-history details](docs/PROMPTS.md).

## Reproducibility and provenance

Source algorithms, defaults, and model parameter names are preserved. Infrastructure changes include portable paths, device-aware temporary tensors, one best checkpoint per run, evaluation without gradient tracking, unique output directories, and explicit randomness for main-model training. Original inactive model parameters and the attention tensor layout are retained for compatibility.

LLM responses, GPU kernels, text-model files, and the original notebook's interactive execution order can affect fresh scores. Historical prompt scores were used during prompt selection; they are not an untouched final holdout estimate. One individual prompt log has a different score from its final ranked record; both are archived and the discrepancy is recorded without correction.

[Source manifests](docs/source_manifest.json) identify the original files using workspace-relative paths and SHA-256 fingerprints. [Copy mappings](docs/copied_files.json) connect renamed data files to their sources. The source tree was read only; it was not reorganized or edited. No checkpoint weights, API keys, or original credential configuration files are included.

## Verification

```bash
python tools/verify_export.py
```

The verifier checks data copies, schema, abundance reconstruction, historical prompt hashes, model/graph equivalence, offline orchestration, and small training runs. It makes no API calls. When the original workspace is available, use `--source-root /path/to/LLM` to recheck its fingerprints and compare against original model definitions. Verification details and limits are in [VERIFICATION.md](docs/VERIFICATION.md).

## Attribution and licensing

The supplied GCN-SA repository README identifies its base work as “Self-Attention Empowered Graph Convolutional Network for Structure Learning and Node Embedding.” This export preserves that attribution and the local source mapping. No new project license, author list, or publication claim has been assigned. Dataset and inherited-code licensing terms must follow their original sources.
