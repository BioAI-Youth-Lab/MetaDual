# MetaDual training

From the repository root:

```bash
python -m model.train --cohort primary --variant full --device cuda
python -m model.train --cohort independent --variant full --device cuda
bash model/run_all.sh --cohort primary --device cuda
```

The six variants are `full`, `abundance_only`, `function_only`, `no_structure`, `no_modifiedt_1` and `no_modifiedt_2`. `MetaDual` is an alias of the checkpoint-compatible historical full class. Source-derived defaults are in `configs/primary.json` and `configs/independent.json`; CLI hyperparameters override them.

Training uses Bray–Curtis graphs for both views and a fixed stratified test set with five repeated train/validation splits. Each invocation and repeat has a separate output directory, best checkpoint, split and prediction file. Default evaluation is on the test set. The prompt-development manuscript preset uses the shared engine's explicit validation-evaluation option; this does not change final cohort evaluation.

Use `python -m model.train --help` for parameters. CPU smoke training requires a small NPZ with at least five samples per class, e.g. `--device cpu --epochs 1 --repeats 1 --hd 8 --k-neighbors 5`. Historical manuscript results are archived separately under `results/`; they are not hardcoded training outputs.
