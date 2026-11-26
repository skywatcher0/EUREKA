"""Coefficient refits and source-environment scoring."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from eureka._symbolic.core.ast import Program
from eureka._symbolic.court.scorer import _fit_linear_coeffs
from eureka._symbolic.search.trace_repair import RepairSettings, complete_support


@dataclass(frozen=True)
class SourceSettings:
    degrees: tuple[int, ...] = (1, 2, 3)
    node_cap: int = 40
    max_ifelse: int = 3
    # Engine validity limit: max_tree_depth 4 x num_terms 5 x 8 unrolled nodes.
    max_unrolled_nodes: int = 160
    tolerance: float = 0.01
    max_abs_coefficient: float = 10.0
    unary_ops: tuple[str, ...] = ("sin", "cos")


SETTINGS = SourceSettings()


def completion_pool(X: np.ndarray, y: np.ndarray, seed: int,
                    settings: SourceSettings = SETTINGS) -> list[tuple[Program, str]]:
    """Every program on the sparse-completion paths of the full support, per degree."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    out: list[tuple[Program, str]] = []
    for degree in settings.degrees:
        repair = RepairSettings(node_cap=settings.node_cap, max_ifelse=settings.max_ifelse,
                                poly_degree=degree, piecewise_degree=min(degree, 2),
                                unary_ops=settings.unary_ops)
        programs, _ = complete_support(X, y, list(range(X.shape[1])), repair, input_dim=X.shape[1], seed=seed)
        out += [(program, f"completion_d{degree}_{family}") for _, _, program, family in programs]
    return out


def _design(program: Program, X: np.ndarray) -> np.ndarray:
    Phi, _ = program.design_matrix(np.asarray(X, dtype=np.float32))
    Phi = np.asarray(Phi, dtype=np.float64)
    return np.clip(np.nan_to_num(Phi, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)


def refit(program: Program, X: np.ndarray, y: np.ndarray) -> Program:
    """Same coefficient fit as the Court (ridge 1e-6 on standardized design columns)."""
    fitted = program.clone()
    fitted.coeffs, fitted.intercept = _fit_linear_coeffs(_design(fitted, X), np.asarray(y).reshape(-1))
    return fitted


def max_abs_parameter(program: Program) -> float:
    values = [abs(float(program.intercept))]
    if program.coeffs is not None:
        values += np.abs(np.asarray(program.coeffs, dtype=np.float64)).tolist()
    return float(max(values))


def loeo(program: Program, Xtr, ytr, etr, Xva, yva, eva) -> tuple[float, dict[int, float]]:
    """Leave-one-source-environment-out RMSE with the program structure held fixed.

    For each source environment e the linear coefficients are refitted on the
    source-training rows of the other environments; the error is computed on
    all source rows (training and validation) of e. Environments get equal
    weight: the score is sqrt(mean_e MSE_e).
    """
    ytr, yva = np.asarray(ytr, np.float64).reshape(-1), np.asarray(yva, np.float64).reshape(-1)
    Ptr, Pva = _design(program, Xtr), _design(program, Xva)
    errors: dict[int, float] = {}
    for e in np.unique(etr):
        coef, intercept = _fit_linear_coeffs(Ptr[etr != e], ytr[etr != e])
        P = np.vstack([Ptr[etr == e], Pva[eva == e]])
        target = np.concatenate([ytr[etr == e], yva[eva == e]])
        prediction = intercept + P @ np.asarray(coef, dtype=np.float64)
        errors[int(e)] = float(np.mean((prediction - target) ** 2))
    return float(np.sqrt(np.mean(list(errors.values())))), errors


def score_pool(finalists: list[Program], Xtr, ytr, etr, Xva, yva, eva, seed: int,
               settings: SourceSettings = SETTINGS) -> tuple[list[Program], list[dict[str, Any]]]:
    """Fit every distinct candidate on all source-training rows and score it.

    Returns the fitted programs and one table row per program with its
    LOEO score (source-validation RMSE when there are fewer than two source
    environments), source-validation RMSE, size and admissibility.
    """
    Xtr, Xva = np.asarray(Xtr, np.float32), np.asarray(Xva, np.float32)
    ytr, yva = np.asarray(ytr, np.float64).reshape(-1), np.asarray(yva, np.float64).reshape(-1)
    etr, eva = np.asarray(etr).reshape(-1), np.asarray(eva).reshape(-1)
    pool = ([(Program(terms=[]), "constant")] + [(p, "search") for p in finalists]
            + completion_pool(Xtr, ytr, seed, settings))
    environments = len(np.unique(etr))
    table: list[dict[str, Any]] = []
    fitted_programs: list[Program] = []
    seen: set[str] = set()
    for program, origin in pool:
        key = program.structural_hash()
        if key in seen:
            continue
        seen.add(key)
        if program.count_ifelse() > settings.max_ifelse or program.count_nodes() > settings.max_unrolled_nodes:
            continue
        fitted = refit(program, Xtr, ytr)
        val_rmse = float(np.sqrt(np.mean((np.asarray(fitted.predict(Xva)[0], np.float64) - yva) ** 2)))
        if environments >= 2:
            score, errors = loeo(fitted, Xtr, ytr, etr, Xva, yva, eva)
        else:
            score, errors = val_rmse, {}
        nodes = int(fitted.count_unique_nodes())
        coefficient = max_abs_parameter(fitted)
        table.append(dict(index=len(fitted_programs), origin=origin, nodes=nodes, max_abs_coefficient=coefficient,
                          loeo_rmse=score, loeo_mse_by_env={str(k): v for k, v in errors.items()},
                          source_val_rmse=val_rmse, equation=fitted.equation(),
                          admissible=bool(nodes <= settings.node_cap and coefficient <= settings.max_abs_coefficient
                                          and np.isfinite(score) and np.isfinite(val_rmse))))
        fitted_programs.append(fitted)
    return fitted_programs, table




RULE_KEYS = {"loeo": "loeo_rmse", "validation": "source_val_rmse"}


def choose(table: list[dict[str, Any]], rule: str, settings: SourceSettings = SETTINGS) -> dict[str, Any]:
    key = RULE_KEYS[rule]
    admissible = [row for row in table if row["admissible"]]
    best = min(row[key] for row in admissible)
    near = [row for row in admissible if row[key] <= best * (1.0 + settings.tolerance)]
    chosen = min(near, key=lambda r: (r["nodes"], r["max_abs_coefficient"], r[key]))
    return dict(rule=rule, admissible=len(admissible), near_best=len(near), best_score=best, chosen=chosen)


def select_source_expression(finalists: list[Program], Xtr, ytr, etr, Xva, yva, eva, seed: int,
              settings: SourceSettings = SETTINGS, rules: tuple[str, ...] = ("loeo",)
              ) -> tuple[dict[str, Program], dict[str, Any], list[dict[str, Any]]]:
    """Return ({rule: program}, summary, candidate table). Coefficients are fitted on all source-training rows."""
    started = time.perf_counter()
    fitted_programs, table = score_pool(finalists, Xtr, ytr, etr, Xva, yva, eva, seed, settings)
    choices = {rule: choose(table, rule, settings) for rule in rules}
    programs = {rule: fitted_programs[c["chosen"]["index"]] for rule, c in choices.items()}
    environments = len(np.unique(np.asarray(etr).reshape(-1)))
    summary = dict(settings=asdict(settings), source_environments=environments,
                   loeo_available=environments >= 2, candidates=len(table), choices=choices,
                   origins={o: sum(r["origin"].startswith(o) for r in table) for o in ("constant", "search",
                                                                                         "completion")},
                   seconds=time.perf_counter() - started, target_arrays_read=False)
    return programs, summary, table
