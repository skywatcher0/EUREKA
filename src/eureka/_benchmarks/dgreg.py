"""Build DGReg splits with source-only feature screening."""

from functools import lru_cache

import numpy as np
import pandas as pd

from .data import DATA, DatasetBundle, SPLIT_PROTOCOL, fingerprint, source_indices, stable_seed
from .screening import select_exogenous_independent_features


TARGET_ENVIRONMENTS = {
    "nonlin": (0,),
    "sin": (1, 2, 4, 7, 8, 11, 12, 14),
}
CAP = 6000


@lru_cache(maxsize=2)
def _read(task):
    frame = pd.read_csv(DATA / "dgreg" / f"{task}.csv.gz")
    names = [name for name in frame if name not in {"domain", "y"}]
    return frame[names].to_numpy(np.float64), frame.y.to_numpy(np.float32), frame.domain.to_numpy(np.int64), names


def _cap(indices, env, seed, label):
    if len(indices) <= CAP:
        return indices
    groups = sorted(np.unique(env[indices]).tolist())
    per_group = max(1, CAP // len(groups))
    chosen = []
    for group in groups:
        rows = indices[env[indices] == group]
        rng = np.random.default_rng(stable_seed(SPLIT_PROTOCOL, "dgreg", seed, label, int(group)))
        chosen.extend(np.sort(rng.choice(rows, size=min(per_group, len(rows)), replace=False)))
    if len(chosen) < CAP:
        remaining = np.setdiff1d(indices, np.asarray(chosen, dtype=np.int64))
        rng = np.random.default_rng(stable_seed(SPLIT_PROTOCOL, "dgreg", seed, label, "remainder"))
        chosen.extend(np.sort(rng.choice(remaining, size=min(CAP - len(chosen), len(remaining)), replace=False)))
    return np.asarray(sorted(chosen[:CAP]), dtype=np.int64)


def _filter_constants_and_proxies(X, y):
    y = y.astype(np.float64)
    keep = []
    for column, values in enumerate(X.T):
        scale = max(1.0, float(np.max(np.abs(values))), float(np.max(np.abs(y))))
        if float(np.std(values)) > 1e-12 and float(np.max(np.abs(values - y))) > 1e-10 * scale:
            keep.append(column)
    if not keep:
        raise ValueError("DGReg source filtering removed every feature.")
    return np.array(keep, dtype=int)


def build_dgreg(task, split_seed):
    X, y, env, names = _read(task)
    target_envs = TARGET_ENVIRONMENTS[task]
    target = np.isin(env, target_envs)
    train, val = source_indices(env, ~target, lambda group: stable_seed(
        SPLIT_PROTOCOL, "dgreg", split_seed, int(group), "row_split",
    ), 0.15)
    train = _cap(train, env, split_seed, "train")
    val = _cap(val, env, split_seed, "validation")
    test = _cap(np.flatnonzero(target), env, split_seed, "test")
    keep = _filter_constants_and_proxies(X[train], y[train])
    # Keep the column-major layout used by the reference fit, including during
    # screening. Array order changes floating-point reductions in the optimizer.
    X_train, X_val, X_test = (X[rows][:, keep].astype(np.float32) for rows in (train, val, test))
    screened, _ = select_exogenous_independent_features(X_train, env[train], count=None)
    split = {
        "protocol": SPLIT_PROTOCOL,
        "dataset": task,
        "split_seed": int(split_seed),
        "source_envs": sorted(np.unique(env[~target]).astype(int).tolist()),
        "target_envs": list(target_envs),
        "train_indices": train.tolist(),
        "validation_indices": val.tolist(),
        "test_indices": test.tolist(),
    }
    meta = {
        "dataset_name": f"dgreg_{task}_clean",
        "feature_names": [names[index] for index in keep[screened]],
        "target_name": "y",
        "split_fingerprint": fingerprint(split, compact=True),
    }
    return DatasetBundle(
        X_train[:, screened], y[train].reshape(-1, 1), env[train],
        X_val[:, screened], y[val].reshape(-1, 1), env[val],
        X_test[:, screened], y[test].reshape(-1, 1), env[test], meta,
    )
