"""Refit EUREKA on built-in datasets and score held observations."""

from concurrent.futures import ProcessPoolExecutor
import csv
from itertools import groupby
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import numpy as np
import pandas as pd

from eureka.model import _fit_prepared, _restore_y, _write_model_files, regression_metrics

from .data import DATA, build_uci


BENCHMARKS = {
    "heating": ("real_a", "energy"),
    "concrete": ("real_a", "concrete"),
    "bicycle": ("real_a", "bike"),
    "wind": ("real_a", "bernoulli_high_speed"),
    "steel": ("real_b", "steel"),
    "shortcut-a": ("shortcut_a", None),
    "shortcut-b": ("shortcut_b", None),
    "dgreg-a": ("dgreg_a", None),
    "dgreg-b": ("dgreg_b", None),
}


def _unit(row):
    held = row["held_environment"]
    return held.split("|", 1)[1] if row["suite"].startswith("real_") else held


def _reference_rows(benchmark):
    suite, task = BENCHMARKS[benchmark]
    with (DATA / "reference.csv").open(encoding="utf-8", newline="") as handle:
        return [
            row for row in csv.DictReader(handle)
            if row["suite"] == suite and (task is None or row["inference_unit"] == task)
        ]


def available_units(benchmark):
    return list(dict.fromkeys(_unit(row) for row in _reference_rows(benchmark)))


def _build(benchmark, row, prepared_wind):
    if benchmark == "wind":
        from .wind import load_wind

        return load_wind(prepared_wind)
    if row["suite"].startswith("real_"):
        return build_uci(row["inference_unit"], _unit(row))
    task, split_seed = row["inference_unit"].split("|", 1)
    if benchmark.startswith("dgreg-"):
        from .dgreg import build_dgreg

        return build_dgreg(task, int(split_seed))

    from .synthetic import generate_scm_bundle

    return generate_scm_bundle(task, num_envs=6, n_per_env=1800, noise_level=0.08,
                               schedule="linear_flip", dummy_dim=5, seed=int(split_seed), train_envs=3)


def _fit_unit(benchmark, numbered_rows, prepared_wind, output, multiple_fits):
    bundle = _build(benchmark, numbered_rows[0][1], prepared_wind)
    fit_rows = []
    for fit_number, row in numbered_rows:
        actual_split = bundle.meta.get("split_fingerprint", "")
        if row["split_sha256"] and row["split_sha256"] != actual_split:
            raise ValueError(f"Dataset split checksum changed for {_unit(row)}.")
        seed = int(row["seed"])
        start = perf_counter()
        model = _fit_prepared(
            bundle.X_train, bundle.y_train, bundle.env_train,
            bundle.X_val, bundle.y_val, bundle.env_val,
            seed=seed, metadata=bundle.meta,
        )
        seconds = perf_counter() - start
        prediction = model._predict_prepared(bundle.X_test)
        actual = _restore_y(bundle.y_test, model.y_scaler)
        metrics = regression_metrics(actual, prediction, model.source_y_std)
        reference = float(row["nrmse"])
        saved = output / "fits" / f"{fit_number:03d}" if multiple_fits else output
        _write_model_files(model, saved)
        X_holdout = bundle.X_test.astype(np.float64)
        if model.x_scaler is not None:
            X_holdout = X_holdout * model.x_scaler["scale"] + model.x_scaler["mean"]
        predictions = pd.DataFrame(X_holdout, columns=model.feature_names)
        predictions["actual"] = actual
        predictions["prediction"] = prediction
        predictions.to_csv(saved / "predictions.csv", index=False)
        fit_rows.append({
            "unit": _unit(row),
            "seed": seed,
            "train_rows": len(bundle.X_train),
            "validation_rows": len(bundle.X_val),
            "fit_seconds": seconds,
            **metrics,
            "reference_nrmse": reference,
            "within_tolerance": bool(np.isclose(metrics["nrmse"], reference, rtol=1.5e-5, atol=4e-6)),
            "split_sha256": actual_split,
            "model": (saved / "model.json").relative_to(output).as_posix(),
        })
    return fit_rows


def run_benchmark(benchmark, output, *, units=None, seed=None, prepared_wind=None, jobs=1, check=False):
    if jobs < 1:
        raise ValueError("jobs must be at least 1.")
    rows = _reference_rows(benchmark)
    if units:
        unknown = set(units) - {_unit(row) for row in rows}
        if unknown:
            raise ValueError(f"Unknown unit(s) {sorted(unknown)}. Use --list-units to see available names.")
        rows = [row for row in rows if _unit(row) in units]
    if seed is not None:
        rows = [row for row in rows if int(row["seed"]) == seed]
    if not rows:
        raise ValueError("No fits match the requested seed and units.")
    groups = [list(group) for _, group in groupby(enumerate(rows, start=1), key=lambda item: _unit(item[1]))]
    output = Path(output)
    multiple = len(rows) > 1
    args = [(benchmark, group, prepared_wind, output, multiple) for group in groups]
    print(f"Fitting {len(rows)} model(s) on {benchmark} with {min(jobs, len(groups))} process(es).", flush=True)
    if jobs == 1 or len(groups) == 1:
        results = [_fit_unit(*job) for job in args]
    else:
        with ProcessPoolExecutor(max_workers=min(jobs, len(groups))) as workers:
            futures = [workers.submit(_fit_unit, *job) for job in args]
            results = [future.result() for future in futures]

    scored = [row for result in results for row in result]
    with (output / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scored[0]))
        writer.writeheader()
        writer.writerows(scored)
    summary = {
        "benchmark": benchmark,
        "fits": len(scored),
        "units": len(groups),
        "mean_unit_nrmse": mean(median(row["nrmse"] for row in result) for result in results),
        "reference_mean_unit_nrmse": mean(median(row["reference_nrmse"] for row in result) for result in results),
        "fits_within_tolerance": sum(row["within_tolerance"] for row in scored),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{benchmark}  NRMSE {summary['mean_unit_nrmse']:.8g} across {len(groups)} unit(s) and {len(scored)} fit(s)")
    if check and summary["fits_within_tolerance"] != len(scored):
        raise ValueError(f"Reference score agreement failed. See {output / 'scores.csv'}.")
