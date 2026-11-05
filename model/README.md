# Model training and ablations

From the repository root:

```bash
python -m model.train --variant full --device cuda
bash model/run_all.sh --device cuda
```

Variants are `full`, `no_modifiedt_1`, `no_modifiedt_2`, and `no_structure`. The shared entry point preserves the original network classes and default Bray–Curtis graph profile. Use `python -m model.train --help` for paths and hyperparameters.

Each invocation receives a unique output directory; each repeat receives its own best checkpoint, split indices and predictions. Metrics aggregate repeated runs over one fixed test set. For a CPU smoke run use a small input NPZ with at least five samples per class, e.g. `--device cpu --epochs 1 --repeats 1 --hd 8 --k-neighbors 5`.
