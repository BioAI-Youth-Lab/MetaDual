# MetaDual data preparation

`data/final_features.npz` is the primary 4,012-person feature dataset. `data/independent_features.npz` is the independent 6,025-person dataset, stored with lossless compression. Both are ready for model training.

From the repository root, rebuild primary features from the archived input/FCG records:

```bash
python -m data_processing.pipeline prepare
python -m data_processing.pipeline embed --embed-model /path/to/BiomedBERT
python -m data_processing.pipeline merge
```

Generated files go to `outputs/data/` and existing files are not overwritten. Each command accepts explicit input/output paths. Independent upstream raw abundance and FCG-generation inputs are not bundled; its ready-to-train NPZ is supplied instead.

New API generation uses the manuscript Top 50 description by default:

```bash
export DEEPSEEK_API_KEY='YOUR_KEY'
python -m data_processing.pipeline generate --preset manuscript
```

Methods 4.3 defines the default Top 50 input. The existing response archive and feature arrays are retained unchanged; no API generation is performed when inspecting or training with the supplied data. Source-compatibility options are described in `docs/CODE_PROVENANCE.md`. See `docs/DATA.md` for schemas and labels.
