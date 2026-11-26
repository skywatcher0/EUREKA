"""Fit, inspect, save, and apply a source-environment symbolic regression model."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ._method.method import fit_eureka
from ._symbolic.core.ast import Program


def _matrix(X, *, columns=None):
    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) == 0:
        raise ValueError("X must contain rows of numeric features.")
    if columns is not None and values.shape[1] != columns:
        raise ValueError(f"Expected {columns} features, got {values.shape[1]}.")
    if not np.isfinite(values).all():
        raise ValueError("Features must be finite.")
    return values


def _target(y, rows):
    values = np.asarray(y, dtype=np.float64)
    if values.shape not in ((rows,), (rows, 1)) or not np.isfinite(values).all():
        raise ValueError("The target must contain one finite value per row.")
    return values.ravel()


def _labels(environments, rows):
    labels = np.asarray(environments, dtype=object)
    if labels.shape != (rows,) or pd.isna(labels).any():
        raise ValueError("Supply one non-missing environment label per row.")
    return labels


def _encode_environments(train, validation):
    codes, _ = pd.factorize(np.concatenate((train, validation)), sort=False)
    source = codes[:len(train)]
    held = codes[len(train):]
    if not np.isin(held, source).all():
        raise ValueError("Validation labels must identify environments present in training.")
    return source, held


def _split_source(env, seed, fraction):
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    train, validation = [], []
    codes, _ = pd.factorize(env, sort=False)
    for group in np.unique(codes):
        rows = np.flatnonzero(codes == group)
        if len(rows) < 2:
            raise ValueError("Each environment needs at least two rows for an automatic split.")
        order = rng.permutation(rows)
        count = min(len(rows) - 1, max(1, round(fraction * len(rows))))
        validation.extend(order[:count])
        train.extend(order[count:])
    return np.sort(train), np.sort(validation)


def _restore_y(y, scaler):
    values = np.asarray(y, dtype=np.float64).ravel()
    if scaler is not None:
        values = values * scaler["scale"] + scaler["mean"]
    return values


def _standardize_x(X, scaler):
    if scaler is not None:
        X = (X - scaler["mean"]) / scaler["scale"]
    if not np.isfinite(X).all() or np.max(np.abs(X)) > np.finfo(np.float32).max:
        raise ValueError("Transformed features exceed the numeric range of EUREKA.")
    return X.astype(np.float32)


@dataclass(frozen=True)
class EurekaModel:
    """A fitted expression with the feature order and transforms needed for prediction."""

    program: Program
    feature_names: tuple[str, ...]
    target_name: str
    x_scaler: dict | None
    y_scaler: dict | None
    source_y_std: float

    def predict(self, X):
        """Return predictions in target units for rows in the fitted feature order."""
        values = _matrix(X, columns=len(self.feature_names))
        return self._predict_prepared(_standardize_x(values, self.x_scaler))

    def _predict_prepared(self, X):
        prediction, _ = self.program.predict(X)
        return _restore_y(prediction, self.y_scaler)

    def equation(self) -> str:
        """Show the fitted expression and its input and output transforms."""
        definitions = []
        for index, name in enumerate(self.feature_names):
            column = json.dumps(name, ensure_ascii=False)
            if self.x_scaler is None:
                definitions.append(f"x{index} = input[{column}]")
            else:
                mean = self.x_scaler["mean"][index]
                scale = self.x_scaler["scale"][index]
                definitions.append(f"x{index} = (input[{column}] - {mean:.10g}) / {scale:.10g}")
        definitions.append(f"f(x) = {self.program.equation()}")
        target = json.dumps(self.target_name, ensure_ascii=False)
        if self.y_scaler is None:
            definitions.append(f"{target} = f(x)")
        else:
            y_scale, y_mean = self.y_scaler["scale"], self.y_scaler["mean"]
            definitions.append(f"{target} = {y_scale:.10g} * f(x) + ({y_mean:.10g})")
        return "\n".join(definitions)

    def save(self, path: str | Path) -> None:
        """Save the expression and preprocessing as JSON."""
        payload = {
            "format": "eureka-model-v1",
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "x_scaler": self.x_scaler,
            "y_scaler": self.y_scaler,
            "source_y_std": self.source_y_std,
            "program": self.program.to_dict(),
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> EurekaModel:
        """Load a JSON model saved by EUREKA."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "eureka-model-v1":
            raise ValueError("Expected a model saved in the eureka-model-v1 format.")
        return cls(
            Program.from_dict(payload["program"]),
            tuple(payload["feature_names"]),
            payload["target_name"],
            payload["x_scaler"],
            payload["y_scaler"],
            float(payload["source_y_std"]),
        )


def _fit_prepared(X, y, env, X_val, y_val, env_val, *, seed, metadata):
    with threadpool_limits(limits=1):
        program, _ = fit_eureka(X, y, env, X_val, y_val, env_val, seed=seed, meta=metadata)
    y_scaler = metadata.get("y_scaler")
    return EurekaModel(
        program,
        tuple(metadata["feature_names"]),
        metadata.get("target_name", "y"),
        metadata.get("x_scaler"),
        y_scaler,
        float(np.std(_restore_y(y, y_scaler))),
    )


def fit(
    X,
    y,
    environments,
    *,
    validation=None,
    feature_names=None,
    target_name="y",
    seed=0,
    validation_fraction=0.20,
) -> EurekaModel:
    """Fit from source observations, with an optional (X_val, y_val, env_val) tuple."""
    X = _matrix(X)
    y = _target(y, len(X))
    env = _labels(environments, len(X))
    names = tuple(feature_names) if feature_names is not None else tuple(f"x{i}" for i in range(X.shape[1]))
    if len(names) != X.shape[1] or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Supply one non-empty feature name per column.")
    if len(set(names)) != len(names):
        raise ValueError("Feature names must be unique.")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")

    if validation is None:
        train_rows, val_rows = _split_source(env, seed, validation_fraction)
        X_val, y_val, env_val = X[val_rows], y[val_rows], env[val_rows]
        X, y, env = X[train_rows], y[train_rows], env[train_rows]
    else:
        X_val, y_val, env_val = validation
        X_val = _matrix(X_val, columns=X.shape[1])
        y_val = _target(y_val, len(X_val))
        env_val = _labels(env_val, len(X_val))
    env, env_val = _encode_environments(env, env_val)

    spread = X.std(axis=0)
    x_scaler = {"mean": X.mean(axis=0).tolist(), "scale": np.where(spread > 1e-12, spread, 1.0).tolist()}
    y_scaler = {"mean": float(y.mean()), "scale": float(y.std()) or 1.0}
    y_scaled = (y - y_scaler["mean"]) / y_scaler["scale"]
    y_val_scaled = (y_val - y_scaler["mean"]) / y_scaler["scale"]
    metadata = dict(feature_names=names, target_name=str(target_name), x_scaler=x_scaler, y_scaler=y_scaler)
    return _fit_prepared(
        _standardize_x(X, x_scaler), y_scaled.astype(np.float32), env,
        _standardize_x(X_val, x_scaler), y_val_scaled.astype(np.float32), env_val,
        seed=seed, metadata=metadata,
    )


def regression_metrics(y, prediction, source_y_std):
    y = _target(y, len(prediction))
    rmse = float(np.sqrt(np.mean((prediction - y) ** 2)))
    return {"rows": len(y), "rmse": rmse, "nrmse": rmse / source_y_std if source_y_std > 0 else None}


def _write_model_files(model, directory):
    directory.mkdir(parents=True, exist_ok=True)
    model.save(directory / "model.json")
    (directory / "equation.txt").write_text(model.equation() + "\n", encoding="utf-8")
