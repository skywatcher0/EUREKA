"""Download and prepare the high-speed pressure dataset."""

from hashlib import file_digest
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .data import DATA, DatasetBundle, REAL_A_PROTOCOL, SPLIT_PROTOCOL, fingerprint, source_indices, stable_seed


WIND_URL = "https://causalchamber.s3.eu-central-1.amazonaws.com/downloadables/wt_bernoulli_v1.zip"
WIND_SHA256 = "e5d58a67f6307b9099177425e91c449b9e62e7c712f8c6f34d1dc80ce4db01e3"
BUNDLE_NAME = "high-speed"


def _prepare(archive_path):
    with Path(archive_path).open("rb") as archive_file:
        digest = file_digest(archive_file, "sha256").hexdigest()
    if digest != WIND_SHA256:
        raise ValueError(f"Unexpected SHA-256 for {archive_path}: {digest}")
    with ZipFile(archive_path) as archive:
        with archive.open("wt_bernoulli_v1/fans_off.csv") as calibration:
            table = pd.read_csv(calibration, usecols=["pressure_downwind", "pressure_upwind"])
        offset = float((table.pressure_downwind - table.pressure_upwind).iloc[2000:].mean())
        with archive.open("wt_bernoulli_v1/random_loads_intake.csv") as intake:
            table = pd.read_csv(intake, usecols=["flag", "rpm_in", "pressure_downwind", "pressure_upwind"])
    columns = ["rpm_in", "pressure_downwind", "pressure_upwind"]
    grouped = table.groupby("flag", sort=True)[columns].mean().reset_index()
    rpm = grouped.rpm_in.to_numpy(dtype=np.float64) / 3000.0
    pressure = (grouped.pressure_upwind - grouped.pressure_downwind).to_numpy(dtype=np.float64) + offset
    target = rpm >= 0.80
    env = np.clip(np.floor(rpm * 5.0).astype(np.int64), 0, 4)
    train, val = source_indices(env, ~target, lambda group: stable_seed(
        SPLIT_PROTOCOL, "wt_bernoulli_v1", "high_speed", 2026110241, int(group), "row_split",
    ), 0.2)
    test = np.flatnonzero(target)

    split = {
        "protocol": SPLIT_PROTOCOL,
        "dataset": "wt_bernoulli_v1",
        "task_variant": "high_speed",
        "split_seed": 2026110241,
        "split_definition": "target: normalized intake-fan RPM >= 0.80",
        "train_flags": grouped.iloc[train]["flag"].astype(int).tolist(),
        "validation_flags": grouped.iloc[val]["flag"].astype(int).tolist(),
        "test_flags": grouped.iloc[test]["flag"].astype(int).tolist(),
        "environment_definition": "floor((rpm_in/3000)*5), clipped to [0,4]",
    }
    center, scale = float(pressure[train].mean()), float(pressure[train].std())
    X = rpm.reshape(-1, 1).astype(np.float32)
    y = ((pressure - center) / scale).astype(np.float32).reshape(-1, 1)
    return DatasetBundle(
        X[train], y[train], env[train], X[val], y[val], env[val], X[test], y[test], env[test],
        {
            "dataset_name": "cc_wt_bernoulli_v1_high_speed_clean",
            "target_name": "pressure_upwind_minus_downwind_corrected",
            "feature_names": ["rpm_in_normalized"],
            "realshift_protocol": REAL_A_PROTOCOL,
            "split_fingerprint": fingerprint(split, compact=True),
            "x_scaler": None,
            "y_scaler": {"mean": center, "scale": scale},
        },
    )


def prepare_wind(output, archive_path=None):
    """Write the prepared arrays from a local or freshly downloaded upstream ZIP."""
    if archive_path is None:
        with TemporaryDirectory(prefix="eureka-wind-") as temporary:
            raw = Path(temporary) / "wt_bernoulli_v1.zip"
            with urlopen(WIND_URL, timeout=60) as response, raw.open("wb") as archive_file:
                shutil.copyfileobj(response, archive_file)
            bundle = _prepare(raw)
    else:
        bundle = _prepare(archive_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f"{BUNDLE_NAME}.npz", **{
        f"{column}_{part}": getattr(bundle, f"{column}_{part}")
        for part in ("train", "val", "test") for column in ("X", "y", "env")
    })
    (output / f"{BUNDLE_NAME}.json").write_text(json.dumps(bundle.meta, indent=2) + "\n", encoding="utf-8")


def load_wind(directory=None):
    base = Path(directory) if directory is not None else DATA / "wind"
    with np.load(base / f"{BUNDLE_NAME}.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    meta = json.loads((base / f"{BUNDLE_NAME}.json").read_text(encoding="utf-8"))
    return DatasetBundle(**arrays, meta=meta)
