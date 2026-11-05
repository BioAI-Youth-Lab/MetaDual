"""Run the full model or one of the three requested ablations."""
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
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/model")
    for field in fields(TrainingConfig):
        option = "--" + field.name.replace("_", "-")
        parser.add_argument(option, type=type(field.default), default=field.default)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = TrainingConfig(**{field.name: getattr(args, field.name) for field in fields(TrainingConfig)})
    result = run_experiment(args.data, args.output_dir, VARIANTS[args.variant], config, args.variant)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
