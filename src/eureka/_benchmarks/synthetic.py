"""Generate benchmark data with a sign-reversing shortcut input."""

import numpy as np

from .data import DatasetBundle


def _schedule(kind: str, num_envs: int) -> np.ndarray:
    if kind == "linear_flip":
        mid = max(1, num_envs // 2)
        left = np.linspace(2.0, 0.8, mid, endpoint=True)
        right = np.linspace(-0.6, -1.8, num_envs - mid, endpoint=True)
        return np.concatenate([left, right]).astype(np.float32)
    if kind == "linear_decay":
        return np.linspace(1.8, 0.1, num_envs, endpoint=True).astype(np.float32)
    if kind == "sinusoidal":
        x = np.linspace(0.0, 2.0 * np.pi, num_envs, endpoint=False)
        return (1.0 + 0.8 * np.sin(x)).astype(np.float32)
    if kind.startswith("const:"):
        value = float(kind.split(":", 1)[1])
        return np.full(num_envs, value, dtype=np.float32)
    raise ValueError(f"Unknown schedule: {kind}")


def _mechanism(scm_type: str, Xc: np.ndarray, env_index: int, num_envs: int) -> tuple[np.ndarray, str]:
    x0, x1, x2 = Xc[:, 0], Xc[:, 1], Xc[:, 2]
    if scm_type == "A":
        y = np.where(x0 > 0.0, 2.0 * x0 + np.sin(x1) - 0.35 * x2, -1.3 * x0 + np.sin(x1) - 0.35 * x2)
        return y, "if x0>0 then 2*x0+sin(x1)-0.35*x2 else -1.3*x0+sin(x1)-0.35*x2"
    if scm_type == "B":
        y = np.where(x1 > 0.0, (x0 * x1) + 0.5 * x2, 0.2 * (x0 * x1) + 0.5 * x2)
        return y, "if x1>0 then x0*x1+0.5*x2 else 0.2*x0*x1+0.5*x2"
    if scm_type == "C":
        y = np.where(x0 > 0.0, 1.6 * (x1**2) + 0.25 * x2, -1.2 * x1 + 0.25 * x2)
        return y, "if x0>0 then 1.6*x1^2+0.25*x2 else -1.2*x1+0.25*x2"
    if scm_type == "D":
        y = np.where(x0 + x1 > 0.0, x0 * x2 + 0.5 * x1, -0.8 * x2 + 0.3 * x0)
        return y, "if x0+x1>0 then x0*x2+0.5*x1 else -0.8*x2+0.3*x0"
    if scm_type == "E":
        y = np.exp(0.5 * x0) + x1 * x2 - 0.3 * x2
        return y, "exp(0.5*x0)+x1*x2-0.3*x2"
    raise ValueError(f"Unknown scm_type: {scm_type}")


def generate_scm_bundle(
    scm_type: str,
    num_envs: int,
    n_per_env: int,
    noise_level: float,
    schedule: str,
    dummy_dim: int,
    seed: int,
    train_envs: int | None = None,
) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    strengths = _schedule(schedule, num_envs)
    chunks_X: list[np.ndarray] = []
    chunks_y: list[np.ndarray] = []
    chunks_env: list[np.ndarray] = []

    equation = ""
    for env in range(num_envs):
        Xc = rng.normal(0.0, 1.0, size=(n_per_env, 3)).astype(np.float32)
        y_true, equation = _mechanism(scm_type, Xc, env, num_envs)
        y = y_true + rng.normal(0.0, noise_level, size=n_per_env)

        # Spurious feature changes with environment.
        x_spur = strengths[env] * y_true + rng.normal(0.0, 1.0, size=n_per_env)
        dummy = rng.normal(0.0, 1.0, size=(n_per_env, dummy_dim)).astype(np.float32)
        X = np.concatenate([Xc, x_spur.reshape(-1, 1).astype(np.float32), dummy], axis=1)

        chunks_X.append(X)
        chunks_y.append(y.astype(np.float32))
        chunks_env.append(np.full(n_per_env, env, dtype=np.int64))

    X_all = np.concatenate(chunks_X, axis=0).astype(np.float32)
    y_all = np.concatenate(chunks_y, axis=0).reshape(-1, 1).astype(np.float32)
    env_all = np.concatenate(chunks_env, axis=0).astype(np.int64)

    if train_envs is None:
        train_envs = max(3, int(num_envs * 0.6))
    train_envs = min(max(1, train_envs), num_envs - 1)
    train_set = set(range(train_envs))
    test_set = set(range(train_envs, num_envs))

    train_mask = np.isin(env_all, list(train_set))
    test_mask = np.isin(env_all, list(test_set))

    # Validation split only inside train environments.
    train_indices = np.where(train_mask)[0]
    rng.shuffle(train_indices)
    cut = int(0.8 * len(train_indices))
    tr_idx = train_indices[:cut]
    va_idx = train_indices[cut:]

    te_idx = np.where(test_mask)[0]
    rng.shuffle(te_idx)

    feature_names = [f"x{i}" for i in range(X_all.shape[1])]
    bundle = DatasetBundle(
        X_train=X_all[tr_idx],
        y_train=y_all[tr_idx],
        env_train=env_all[tr_idx],
        X_val=X_all[va_idx],
        y_val=y_all[va_idx],
        env_val=env_all[va_idx],
        X_test=X_all[te_idx],
        y_test=y_all[te_idx],
        env_test=env_all[te_idx],
        meta={
            "dataset_name": f"scm_{scm_type}",
            "ground_truth_equation": equation,
            "causal_feature_indices": [0, 1, 2],
            "spurious_feature_index": 3,
            "feature_names": feature_names,
            "source_link": "synthetic",
            "sha256": "synthetic-generated",
            "env_definition": f"env=0..{num_envs - 1}, train=0..{train_envs - 1}, test={train_envs}..{num_envs - 1}",
            "spurious_strength_schedule": strengths.tolist(),
            "noise_level": noise_level,
            "seed": seed,
            "scenario": "virtual",
        },
    )
    bundle.validate()
    return bundle
