"""Train MetaDual on either manuscript cohort, or run a component ablation."""
import argparse
from dataclasses import fields
import json
from pathlib import Path

from model.data import DEFAULT_DATA
from model.engine import TrainingConfig, run_experiment
from model.networks import VARIANTS


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=list(VARIANTS), default="full")
    parser.add_argument("--cohort", choices=["primary", "independent"], default="primary")
    parser.add_argument("--data", type=Path, default=None, help="Override the selected cohort NPZ")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/model")
    for field in fields(TrainingConfig):
        option = "--" + field.name.replace("_", "-")
        parser.add_argument(option, type=type(field.default), default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "configs" / f"{args.cohort}.json").read_text())
    settings = profile["training"]
    settings.update({field.name: getattr(args, field.name) for field in fields(TrainingConfig)
                     if getattr(args, field.name) is not None})
    config = TrainingConfig(**settings)
    data = args.data or root / profile["data"]
    result = run_experiment(data, args.output_dir / args.cohort, VARIANTS[args.variant], config, args.variant)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
