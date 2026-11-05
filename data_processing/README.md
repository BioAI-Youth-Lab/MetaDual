# Data preparation

Run commands from the repository root. The included `data/final_features.npz` can be used directly for training.

```bash
python -m data_processing.pipeline prepare
python -m data_processing.pipeline embed --embed-model /path/to/BiomedBERT
python -m data_processing.pipeline merge
```

The commands preserve sample order and write to `outputs/data/` without overwriting existing files. Each command supports explicit input and output paths. `generate` explicitly calls DeepSeek and is optional because parsed FCG records are archived. See the root README and `docs/DATA.md` for schema, class mapping, and defaults.
