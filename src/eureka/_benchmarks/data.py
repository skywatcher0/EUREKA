"""Prepare the source and held-environment inputs for built-in datasets."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from eureka._method.method import CONCRETE_FEATURES
from eureka._method.families.eureka_heating_envelope import FEATURES as HEATING_FEATURES, TARGET as HEATING_TARGET


DATA = Path(__file__).resolve().parents[1] / "data"
REAL_A_PROTOCOL = "realshift_v1_20261102"
REAL_B_PROTOCOL = "freshshift_v1_20261103"
SPLIT_PROTOCOL = "neurips2026-rebuttal-clean-v2"


@dataclass
class DatasetBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    env_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    env_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    env_test: np.ndarray
    meta: dict

    def validate(self):
        for suffix in ("train", "val", "test"):
            X = np.asarray(getattr(self, f"X_{suffix}"), dtype=np.float32)
            y = np.asarray(getattr(self, f"y_{suffix}"), dtype=np.float32).reshape(-1, 1)
            env = np.asarray(getattr(self, f"env_{suffix}"), dtype=np.int64).ravel()
            if not (len(X) == len(y) == len(env) and len(X)):
                raise ValueError(f"Invalid {suffix} split.")
            setattr(self, f"X_{suffix}", X)
            setattr(self, f"y_{suffix}", y)
            setattr(self, f"env_{suffix}", env)


def stable_seed(protocol, *parts):
    payload = "|".join(map(str, (protocol, *parts))).encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "little")


def fingerprint(payload, *, compact=False):
    separators = (",", ":") if compact else None
    return sha256(json.dumps(payload, sort_keys=True, separators=separators).encode("utf-8")).hexdigest()


def _read_uci(filename, member, *, excel=False):
    with ZipFile(DATA / "uci" / filename) as archive:
        with archive.open(member) as raw:
            return pd.read_excel(BytesIO(raw.read())) if excel else pd.read_csv(raw)


def _heating():
    frame = _read_uci("energy-efficiency.zip", "ENB2012_data.xlsx", excel=True).dropna(how="all")
    X = frame[[f"X{i}" for i in range(1, 9)]].set_axis(HEATING_FEATURES, axis=1)
    shapes = sorted(X.relative_compactness.unique())
    env = X.relative_compactness.map({value: i // 3 for i, value in enumerate(shapes)}).to_numpy()
    labels = [f"compactness {shapes[3 * g]:.2f}-{shapes[3 * g + 2]:.2f}" for g in range(4)]
    return X, frame.Y1.to_numpy(float), env, labels


def _concrete():
    frame = _read_uci("concrete-strength.zip", "Concrete_Data.xls", excel=True)
    X = frame.iloc[:, :8].set_axis(CONCRETE_FEATURES, axis=1)
    env = np.digitize(X.age_days.to_numpy(), [7, 28, 90], right=True)
    return X, frame.iloc[:, 8].to_numpy(float), env, ["age<=7", "7<age<=28", "28<age<=90", "age>90"]


def _bicycle():
    frame = _read_uci("bike-sharing.zip", "day.csv")
    env = (4 * frame.yr + frame.season - 1).to_numpy()
    labels = [f"{2011 + y}-season{s}" for y in (0, 1) for s in (1, 2, 3, 4)]
    return frame[["temp", "atemp", "hum", "windspeed"]], np.sqrt(frame.cnt.to_numpy(float)), env, labels


def _steel():
    frame = _read_uci("steel-energy.zip", "Steel_industry_data.csv")
    time = pd.to_datetime(frame.date, format="%d/%m/%Y %H:%M")
    env = np.clip(time.dt.quarter.to_numpy() - 1, 0, 3)
    X = pd.DataFrame({
        "lagging_reactive_kvarh": frame["Lagging_Current_Reactive.Power_kVarh"],
        "leading_reactive_kvarh": frame["Leading_Current_Reactive_Power_kVarh"],
        "lagging_power_factor": frame["Lagging_Current_Power_Factor"],
        "leading_power_factor": frame["Leading_Current_Power_Factor"],
        "seconds_from_midnight": frame.NSM,
    })
    # The derived CO2 channel uses the response and is excluded from the inputs.
    keep = []
    for group in sorted(np.unique(env)):
        rows = np.flatnonzero(env == group)
        rng = np.random.default_rng(stable_seed(REAL_B_PROTOCOL, "steel", "row_cap", int(group)))
        keep.extend(rng.choice(rows, min(len(rows), 3000), replace=False))
    keep = np.sort(keep)
    return X.iloc[keep], frame.Usage_kWh.to_numpy(float)[keep], env[keep], ["Q1", "Q2", "Q3", "Q4"]


_UCI = {
    "energy": (_heating, HEATING_TARGET, "building-shape group"),
    "concrete": (_concrete, "compressive strength (MPa)", "curing-age group"),
    "bike": (_bicycle, "sqrt(daily rentals)", "year x season"),
    "steel": (_steel, "active energy (kWh)", "production quarter"),
}


def build_uci(dataset, held_label):
    """Keep an entire condition for scoring and split source conditions internally."""
    load, target_name, environment_variable = _UCI[dataset]
    frame, y, env, labels = load()
    X = frame.to_numpy(np.float64)
    target = labels.index(held_label)
    is_steel = dataset == "steel"
    protocol, split_seed, prefix = (
        (REAL_B_PROTOCOL, 2026110341, "freshshift") if is_steel else (REAL_A_PROTOCOL, 2026110241, "realshift")
    )
    train, val = [], []
    for group in sorted(np.unique(env)):
        if group != target:
            rows = np.flatnonzero(env == group)
            seed = stable_seed(protocol, dataset, target, split_seed, group)
            order = np.random.default_rng(seed).permutation(rows)
            cut = max(1, int(round(0.2 * len(rows))))
            val.extend(order[:cut])
            train.extend(order[cut:])
    train, val = np.sort(train), np.sort(val)
    test = np.flatnonzero(env == target)
    x_mean, x_std = X[train].mean(axis=0), X[train].std(axis=0)
    x_std = np.where(x_std > 1e-12, x_std, 1.0)
    y_mean, y_std = float(y[train].mean()), float(y[train].std())
    X_scaled = ((X - x_mean) / x_std).astype(np.float32)
    y_scaled = ((y - y_mean) / y_std).astype(np.float32).reshape(-1, 1)
    split = dict(protocol=protocol, dataset=dataset, target_env=target, split_seed=split_seed,
                 train=train.tolist(), validation=val.tolist(), test=test.tolist())
    metadata = {
        "dataset_name": f"{prefix}_{dataset}",
        "target_name": target_name,
        "environment_variable": environment_variable,
        "environment_labels": labels,
        "realshift_protocol": protocol,
        "feature_names": list(frame.columns),
        "x_scaler": {"mean": x_mean.tolist(), "scale": x_std.tolist()},
        "y_scaler": {"mean": y_mean, "scale": y_std},
        "split_fingerprint": fingerprint(split),
    }
    return DatasetBundle(
        X_scaled[train], y_scaled[train], env[train].astype(np.int64),
        X_scaled[val], y_scaled[val], env[val].astype(np.int64),
        X_scaled[test], y_scaled[test], env[test].astype(np.int64), metadata,
    )


def source_indices(env, source, seed_for_group, val_fraction):
    train, val = [], []
    for group in sorted(np.unique(env[source]).tolist()):
        rows = np.flatnonzero(source & (env == group))
        order = np.random.default_rng(seed_for_group(group)).permutation(rows)
        count = min(len(order) - 2, max(1, int(round(val_fraction * len(order)))))
        val.extend(order[:count])
        train.extend(order[count:])
    return np.asarray(sorted(train), dtype=np.int64), np.asarray(sorted(val), dtype=np.int64)
