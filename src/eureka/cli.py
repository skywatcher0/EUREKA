"""Command-line interface for learning and deploying EUREKA expressions."""

import argparse
import json
from pathlib import Path

import pandas as pd

from ._benchmarks.run import BENCHMARKS, available_units, run_benchmark
from .model import EurekaModel, _write_model_files, fit, regression_metrics


def _measurements(path, features, target, environment):
    frame = pd.read_csv(path, dtype={environment: "string"}, float_precision="round_trip", keep_default_na=False)
    missing = set([*features, target, environment]) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}.")
    if frame[environment].str.strip().eq("").any():
        raise ValueError(f"{path} has missing environment labels.")
    return frame


def _fit(args):
    if len({*args.features, args.target, args.environment}) != len(args.features) + 2:
        raise ValueError("Features, target, and environment must use distinct columns.")
    frame = _measurements(args.csv, args.features, args.target, args.environment)
    held = set(args.holdout or ())
    missing = held - set(frame[args.environment])
    if missing:
        raise ValueError(f"Unknown held environment(s) {sorted(missing)}.")
    mask = frame[args.environment].isin(held)
    source = frame.loc[~mask]
    prediction_column = "eureka_prediction"
    if held and prediction_column in frame:
        raise ValueError(f"Rename the input column {prediction_column!r} before scoring.")
    validation = None
    if args.validation:
        val = _measurements(args.validation, args.features, args.target, args.environment)
        if val[args.environment].isin(held).any():
            raise ValueError("The validation CSV contains a held environment.")
        validation = (val[args.features].to_numpy(), val[args.target].to_numpy(), val[args.environment].to_numpy())

    model = fit(
        source[args.features].to_numpy(), source[args.target].to_numpy(), source[args.environment].to_numpy(),
        validation=validation, feature_names=args.features, target_name=args.target, seed=args.seed,
        validation_fraction=args.validation_fraction,
    )
    _write_model_files(model, args.out)

    if held:
        test = frame.loc[mask].copy()
        test[prediction_column] = model.predict(test[args.features].to_numpy())
        metrics = regression_metrics(test[args.target], test[prediction_column].to_numpy(), model.source_y_std)
        test.to_csv(args.out / "predictions.csv", index=False)
        (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(f"Held-environment metrics {json.dumps(metrics)}")
    print(f"Saved {args.out / 'model.json'} and {args.out / 'equation.txt'}")


def _predict(args):
    model = EurekaModel.load(args.model)
    frame = pd.read_csv(args.csv, float_precision="round_trip")
    missing = set(model.feature_names) - set(frame.columns)
    if missing:
        raise ValueError(f"{args.csv} is missing features {sorted(missing)}.")
    if args.column in frame:
        raise ValueError(f"The output column {args.column!r} already exists. Pick another --column.")
    frame[args.column] = model.predict(frame[list(model.feature_names)].to_numpy())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"Saved {len(frame)} predictions to {args.out}")


def _benchmark(args):
    if args.list_units:
        for name in available_units(args.name):
            print(name)
        return
    if args.out is None:
        raise ValueError("Provide --out for the benchmark results.")
    if args.prepared_wind is not None and args.name != "wind":
        raise ValueError("--prepared-wind applies to the wind dataset.")
    run_benchmark(args.name, args.out, units=args.unit, seed=args.seed, jobs=args.jobs,
                  prepared_wind=args.prepared_wind, check=args.check)


def _prepare_wind(args):
    from ._benchmarks.wind import prepare_wind

    prepare_wind(args.out, args.archive)
    print(f"Saved prepared wind data to {args.out}")


def main():
    parser = argparse.ArgumentParser(prog="eureka", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("fit", help="Fit an expression from a CSV with source environment labels.")
    train.add_argument("csv", type=Path)
    train.add_argument("--features", nargs="+", required=True, metavar="COLUMN")
    train.add_argument("--target", required=True, metavar="COLUMN")
    train.add_argument("--environment", required=True, metavar="COLUMN")
    train.add_argument("--holdout", action="append", metavar="LABEL",
                       help="Score a whole environment after fitting. Repeat to hold out several environments.")
    train.add_argument("--validation", type=Path, metavar="CSV", help="Use a separate source-validation CSV.")
    train.add_argument("--validation-fraction", type=float, default=0.20)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--out", required=True, type=Path, metavar="DIR")
    train.set_defaults(handler=_fit)

    predict = commands.add_parser("predict", help="Apply a saved expression to CSV rows.")
    predict.add_argument("model", type=Path)
    predict.add_argument("csv", type=Path)
    predict.add_argument("--out", required=True, type=Path, metavar="CSV")
    predict.add_argument("--column", default="eureka_prediction")
    predict.set_defaults(handler=_predict)

    benchmark = commands.add_parser("benchmark", help="Fit and score a built-in environment-shift dataset.")
    benchmark.add_argument("name", choices=BENCHMARKS)
    benchmark.add_argument("--unit", action="append", help="Run a particular held condition or split. Repeat as needed.")
    benchmark.add_argument("--list-units", action="store_true")
    benchmark.add_argument("--seed", type=int, help="Run a particular fitting seed.")
    benchmark.add_argument("--jobs", type=int, default=1, help="Fit independent units in separate processes.")
    benchmark.add_argument("--prepared-wind", type=Path, metavar="DIR")
    benchmark.add_argument("--check", action="store_true", help="Require agreement with the supplied numeric references.")
    benchmark.add_argument("--out", type=Path, metavar="DIR")
    benchmark.set_defaults(handler=_benchmark)

    prepare = commands.add_parser("prepare-wind", help="Prepare the wind dataset from its original ZIP.")
    prepare.add_argument("--archive", type=Path, help="Use a local archive or download the upstream ZIP automatically.")
    prepare.add_argument("--out", required=True, type=Path, metavar="DIR")
    prepare.set_defaults(handler=_prepare_wind)

    args = parser.parse_args()
    try:
        args.handler(args)
    except (ValueError, OSError, ImportError, KeyError) as error:
        parser.exit(2, f"eureka: {error}\n")
