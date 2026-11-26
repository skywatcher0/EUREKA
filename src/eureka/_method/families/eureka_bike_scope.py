"""Align source weather observations to the bundled daily rental calendar."""

from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

FEATURES = ("temp", "atemp", "hum", "windspeed")
DATA = Path(__file__).resolve().parents[2] / "data/uci/bike-sharing.zip"
SCHEMA = dict(dataset_name="realshift_bike", target_name="sqrt(daily rentals)",
              environment_variable="year x season", realshift_protocol="realshift_v1_20261102")


def supported(X, meta):
    return (meta is not None and X.shape[1] == 4 and all(meta.get(k) == v for k, v in SCHEMA.items())
            and tuple(meta.get("feature_names", ())) == FEATURES and meta.get("x_scaler") is not None
            and meta.get("y_scaler") is not None)


@lru_cache(maxsize=1)
def archive_calendar():
    with ZipFile(DATA) as archive:
        with archive.open("day.csv") as rows:
            frame = pd.read_csv(rows, usecols=["dteday", "yr", "season", *FEATURES], parse_dates=["dteday"])
    days = (frame.dteday - frame.dteday.iloc[0]).dt.days.to_numpy(int)
    assert frame.dteday.is_monotonic_increasing and frame.dteday.is_unique
    return frame[list(FEATURES)].to_numpy(float), (4 * frame.yr + frame.season - 1).to_numpy(int), days


def source_days(X, env, V, val_env, meta):
    if not supported(X, meta):
        return None
    try:
        weather, archive_groups, calendar = archive_calendar()
    except FileNotFoundError:
        return None
    scaler = meta["x_scaler"]
    center, spread = (np.asarray(scaler[key], dtype=float) for key in ("mean", "scale"))
    if center.shape != (4,) or spread.shape != (4,) or not np.all(spread > 0):
        return None
    archive_rows = ((weather - center) / spread).astype(np.float32)
    lookup = {(int(g), *x): int(d) for g, x, d in zip(archive_groups, archive_rows, calendar, strict=True)}
    if len(lookup) != len(calendar):
        return None
    aligned = []
    for x, groups in ((X, env), (V, val_env)):
        values = [lookup.get((int(g), *row)) for g, row in zip(np.ravel(groups), x.astype(np.float32), strict=True)]
        if any(day is None for day in values):
            return None
        aligned.append(np.asarray(values, dtype=int))
    return aligned if len(np.unique(np.concatenate(aligned))) == sum(map(len, aligned)) else None
