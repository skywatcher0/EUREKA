"""Sparse global and axis-gated candidate completion."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from math import comb
from typing import Callable

import numpy as np

from eureka._symbolic.core.ast import Node, Program
from eureka._symbolic.court.scorer import _fit_linear_coeffs, mdl_loss

Term = tuple  # ("mono", (j, ...)) or ("unary", op, j)


@dataclass(frozen=True)
class RepairSettings:
    node_cap: int = 20
    max_ifelse: int = 3
    poly_degree: int = 3
    piecewise_degree: int = 2
    unary_ops: tuple[str, ...] = ("sin", "cos")
    max_basis: int = 1000
    split_quantiles: int = 15
    splits_per_support: int = 2
    split_screen_rows: int = 2000
    min_child_fraction: float = 0.05
    max_steps: int = 20
    # Scale-free source criterion log(MSE) + length_weight * MDL / n, i.e. the
    # logarithm of the existing ``sample_length`` score policy.
    length_weight: float = 2.0 * 0.6931471805599453

    def __post_init__(self):
        if any(op not in {"sin", "cos"} for op in self.unary_ops):
            raise ValueError("Repair unary basis admits only sin and cos")


def term_basis(support: list[int], degree: int, max_basis: int, unary_ops=()) -> list[Term]:
    """Monomials up to ``degree`` (lowered to fit ``max_basis``) and unary leaves."""
    support = sorted(int(j) for j in support)
    if not support:
        return []
    degree = max(1, int(degree))
    while degree > 1 and comb(len(support) + degree, degree) - 1 > max_basis:
        degree -= 1
    terms: list[Term] = [("mono", m) for power in range(1, degree + 1)
                         for m in combinations_with_replacement(support, power)]
    terms += [("unary", op, j) for op in unary_ops for j in support]
    return terms


def term_node(term: Term | None) -> Node:
    if term is None:
        return Node("const", constant=0.0)
    if term[0] == "unary":
        return Node(term[1], children=[Node("var", feature_idx=int(term[2]))])
    monomial = term[1]
    if not monomial:
        return Node("const", constant=1.0)
    node = Node("var", feature_idx=int(monomial[0]))
    for j in monomial[1:]:
        node = Node("*", children=[node, Node("var", feature_idx=int(j))])
    return node


def term_columns(X: np.ndarray, terms: list[Term]) -> np.ndarray:
    out = np.ones((len(X), len(terms)), dtype=np.float64)
    for c, term in enumerate(terms):
        if term[0] == "unary":
            out[:, c] = np.sin(X[:, term[2]]) if term[1] == "sin" else np.cos(X[:, term[2]])
        else:
            for j in term[1]:
                out[:, c] *= X[:, j]
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)


def _predicate(split: tuple[int, float]) -> Node:
    k, c = split
    return Node("gt", children=[Node("var", feature_idx=int(k)), Node("const", constant=float(c))])


def _pairs(left: list[Term], right: list[Term]) -> list[tuple[Term | None, Term | None]]:
    left, right = list(left), list(right)
    pairs: list[tuple[Term | None, Term | None]] = []
    for term in list(left):
        if term in right:
            pairs.append((term, term))
            left.remove(term)
            right.remove(term)
    while left or right:
        pairs.append((left.pop(0) if left else None, right.pop(0) if right else None))
    return pairs


def build_program(shared: list[Term], left: list[Term] = (), right: list[Term] = (),
                  split: tuple[int, float] | None = None) -> Program:
    """Shared terms plus gated pairs ``IfElse(x_k > c, A, B)``.

    ``A`` is active where the predicate holds and ``B`` elsewhere; each body
    has its own coefficient. Identical bodies share one pair; an unmatched
    body is paired with the constant zero.
    """
    terms = [term_node(t) for t in shared]
    if split is None or not (left or right):
        return Program(terms=terms)
    terms.extend(Node("ifelse", children=[_predicate(split), term_node(a), term_node(b)])
                 for a, b in _pairs(left, right))
    return Program(terms=terms, coefficient_mode="conditional_affine")


def _lstsq_fit(C: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Least squares with an intercept; returns (coefficients, intercept, SSE)."""
    if C.shape[1] == 0:
        r = y - y.mean()
        return np.zeros(0), float(y.mean()), float(r @ r)
    center = C.mean(axis=0)
    Cc = C - center
    r = y - y.mean()
    coef, *_ = np.linalg.lstsq(Cc, r, rcond=None)
    residual = r - Cc @ coef
    return coef, float(y.mean() - center @ coef), float(residual @ residual)


def _lstsq_sse(C: np.ndarray, y: np.ndarray) -> float:
    return _lstsq_fit(C, y)[2]


def refine_threshold(x: np.ndarray, y: np.ndarray, pool: Callable[[np.ndarray], np.ndarray],
                     c: float, min_child: int, iterations: int = 4) -> tuple[float, float, int]:
    """Exact alternating refinement of an axis threshold ``x > c``.

    With branch coefficients held fixed, the SSE of every threshold between
    adjacent sorted source values is computed from prefix sums; the threshold
    is placed at the centre of the widest minimum-SSE run. Coefficients are
    then refit and the step is repeated. Returns (threshold, SSE, fits).
    """
    order = np.argsort(x, kind="stable")
    xs, n = x[order], len(x)
    ones, zeros = np.ones(n), np.zeros(n)
    gate = (x > c).astype(np.float64)
    coef, intercept, sse = _lstsq_fit(pool(gate), y)
    fits = 1
    positions = np.flatnonzero(xs[:-1] < xs[1:]) + 1  # left size at each gap
    positions = positions[(positions >= min_child) & (n - positions >= min_child)]
    if not len(positions):
        return c, sse, fits
    for _ in range(iterations):
        true_error = (y - intercept - pool(ones) @ coef)[order] ** 2
        false_error = (y - intercept - pool(zeros) @ coef)[order] ** 2
        # Rows left of a gap take the false branch (x <= c).
        candidate = (np.cumsum(false_error)[positions - 1]
                     + np.sum(true_error) - np.cumsum(true_error)[positions - 1])
        best = float(np.min(candidate))
        tied = np.flatnonzero(candidate <= best + 1e-12 * max(1.0, best))
        runs = np.split(tied, np.flatnonzero(np.diff(tied) > 1) + 1)
        run = max(runs, key=lambda r: xs[positions[r[-1]]] - xs[positions[r[0]] - 1])
        lower, upper = xs[positions[run[0]] - 1], xs[positions[run[-1]]]
        new_c = float(lower + 0.5 * (upper - lower))
        gate = (x > new_c).astype(np.float64)
        new_coef, new_intercept, new_sse = _lstsq_fit(pool(gate), y)
        fits += 1
        if new_sse >= sse - 1e-12 * max(1.0, sse):
            break
        c, coef, intercept, sse = new_c, new_coef, new_intercept, new_sse
    return c, sse, fits


def forward_path(C: np.ndarray, y: np.ndarray, max_steps: int,
                 admissible: Callable[[list[int]], bool]) -> list[tuple[list[int], float]]:
    """Orthogonal forward selection; returns (selected columns, SSE) per step.

    Columns and target are centred, so every SSE is the least-squares SSE of
    the selected columns plus an intercept.
    """
    C = C - C.mean(axis=0, keepdims=True)
    r = y - y.mean()
    norms = np.sum(C * C, axis=0)
    usable = norms > 1e-10 * max(1, len(y))
    perp = C.copy()
    perp_norms = norms.copy()
    selected: list[int] = []
    path: list[tuple[list[int], float]] = []
    total = max(1e-300, float(r @ r))
    for _ in range(max_steps):
        with np.errstate(divide="ignore", invalid="ignore"):
            gains = (perp.T @ r) ** 2 / np.maximum(perp_norms, 1e-300)
        independent = usable & (perp_norms > 1e-8 * np.maximum(norms, 1e-300))
        gains = np.where(independent, gains, -1.0)
        gains[selected] = -1.0
        chosen = None
        for index in np.argsort(-gains, kind="stable")[:64]:
            if gains[index] <= 1e-12 * total:
                break
            if admissible(selected + [int(index)]):
                chosen = int(index)
                break
        if chosen is None:
            break
        q = perp[:, chosen] / np.sqrt(perp_norms[chosen])
        r = r - q * float(q @ r)
        projection = q @ perp
        perp -= np.outer(q, projection)
        perp_norms = np.maximum(perp_norms - projection ** 2, 0.0)
        selected.append(chosen)
        path.append((list(selected), float(r @ r)))
    return path


def backward_prune(cols: list[int], C: np.ndarray, y: np.ndarray,
                   objective: Callable[[list[int], float], float]) -> tuple[list[int], float, int]:
    """Greedy removal while the source objective decreases; returns fits used."""
    current = list(cols)
    sse = _lstsq_sse(C[:, current], y)
    best = objective(current, sse)
    fits = 1
    while len(current) > 1:
        trials = []
        for position in range(len(current)):
            reduced = current[:position] + current[position + 1:]
            reduced_sse = _lstsq_sse(C[:, reduced], y)
            fits += 1
            trials.append((objective(reduced, reduced_sse), reduced, reduced_sse))
        value, reduced, reduced_sse = min(trials, key=lambda item: item[0])
        if value >= best:
            break
        best, current, sse = value, reduced, reduced_sse
    return current, sse, fits


def screen_splits(X: np.ndarray, y: np.ndarray, support: list[int], *, quantiles: int,
                  min_child: int, top: int, rows: int, seed: int) -> tuple[list[tuple[int, float]], int]:
    """Rank axis splits by the SSE of branch-wise diagonal-quadratic fits.

    At most ``top`` splits are returned, on distinct features. Screening uses
    a deterministic row subsample; the chosen split is refined later on all
    source-training rows.
    """
    if len(X) > rows:
        index = np.sort(np.random.default_rng(seed).choice(len(X), size=rows, replace=False))
        X, y = X[index], y[index]
        min_child = max(10, int(min_child * rows / max(1, len(index))))
    base = np.column_stack([np.ones(len(X))] + [c for j in support for c in (X[:, j], X[:, j] ** 2)])
    levels = np.linspace(0.1, 0.9, max(1, int(quantiles)))
    ridge = 1e-8 * np.eye(base.shape[1])
    scored: list[tuple[float, int, float]] = []
    fits = 0
    for k in support:
        for c in np.unique(np.quantile(X[:, k], levels)):
            gate = X[:, k] > c
            if min(int(gate.sum()), int((~gate).sum())) < min_child:
                continue
            sse = 0.0
            for mask in (gate, ~gate):
                B, t = base[mask], y[mask]
                G = B.T @ B
                coef = np.linalg.solve(G + ridge * max(1.0, np.trace(G)), B.T @ t)
                residual = t - B @ coef
                sse += float(residual @ residual)
                fits += 1
            scored.append((sse, int(k), float(c)))
    scored.sort()
    chosen: list[tuple[int, float]] = []
    used: set[int] = set()
    for _, k, c in scored:
        if k not in used:
            chosen.append((k, c))
            used.add(k)
        if len(chosen) >= top:
            break
    return chosen, fits


def complete_support(X: np.ndarray, y: np.ndarray, support: list[int], settings: RepairSettings,
                     input_dim: int, seed: int = 0) -> tuple[list[tuple[float, float, Program, str]], int]:
    """Global and gated sparse completions on ``support``.

    Returns ``(source objective, training MSE, program, family)`` sorted by
    ``log(MSE) + length_weight * structural cost / n`` (no trace term), and
    the number of least-squares solves used.
    """
    support = sorted(int(j) for j in support)
    if not support:
        return [], 0
    n = len(y)
    fits = 0
    results: dict[str, tuple[float, float, Program, str]] = {}
    floor = 1e-12 * max(float(np.var(y)), 1e-300)
    X32 = np.asarray(X, dtype=np.float32)

    def source_objective(sse: float, program: Program) -> float:
        return float(np.log(max(sse / n, floor)) + settings.length_weight * mdl_loss(program, input_dim) / n)

    def record(program: Program, sse: float, family: str) -> None:
        nonlocal fits
        key = program.structural_hash()
        objective = source_objective(sse, program)
        if key in results and results[key][0] <= objective:
            return
        # Store coefficients fitted to the candidate structure.
        Phi, _ = program.design_matrix(X32)
        program.coeffs, program.intercept = _fit_linear_coeffs(Phi, y)
        fits += 1
        results[key] = (objective, sse / n, program, family)

    def run_pool(C: np.ndarray, build: Callable[[list[int]], Program | None],
                 admissible: Callable[[list[int]], bool], family: str) -> list[int] | None:
        nonlocal fits

        def objective(cols: list[int], sse: float) -> float:
            program = build(cols)
            return np.inf if program is None else source_objective(sse, program)

        path = forward_path(C, y, settings.max_steps, admissible)
        fits += len(path)
        if not path:
            return None
        for cols, sse in path:
            program = build(cols)
            if program is not None:
                record(program, sse, family)
        best_cols, best_sse = min(path, key=lambda item: objective(*item))
        pruned, pruned_sse, used = backward_prune(best_cols, C, y, objective)
        fits += used
        program = build(pruned)
        if program is not None:
            record(program, pruned_sse, family + "_pruned")
        return pruned

    global_terms = term_basis(support, settings.poly_degree, settings.max_basis, settings.unary_ops)
    if global_terms:
        def build_global(cols: list[int]) -> Program:
            return build_program([global_terms[c] for c in cols])

        run_pool(term_columns(X, global_terms), build_global,
                 lambda cols: build_global(cols).count_unique_nodes() <= settings.node_cap,
                 "global")

    if settings.max_ifelse <= 0:
        return sorted(results.values(), key=lambda item: (item[0], item[2].count_unique_nodes())), fits
    shared_terms = term_basis(support, settings.piecewise_degree, settings.max_basis // 3, settings.unary_ops)
    body_terms = [("mono", ())] + term_basis(support, settings.piecewise_degree, settings.max_basis // 3)
    min_child = max(20, int(settings.min_child_fraction * n))
    splits, screen_fits = screen_splits(X, y, support, quantiles=settings.split_quantiles,
                                        min_child=min_child, top=settings.splits_per_support,
                                        rows=settings.split_screen_rows, seed=seed)
    fits += screen_fits
    shared_columns = term_columns(X, shared_terms)
    body_columns = term_columns(X, body_terms)
    kinds = ([("s", t) for t in shared_terms] + [("l", t) for t in body_terms]
             + [("r", t) for t in body_terms])

    def parts(cols: list[int]) -> tuple[list[Term], list[Term], list[Term]]:
        return ([kinds[i][1] for i in cols if kinds[i][0] == "s"],
                [kinds[i][1] for i in cols if kinds[i][0] == "l"],
                [kinds[i][1] for i in cols if kinds[i][0] == "r"])

    def gated_pool(gate: np.ndarray) -> np.ndarray:
        return np.column_stack([shared_columns, body_columns * gate[:, None],
                                body_columns * (1.0 - gate)[:, None]])

    for k, c in splits:
        def build_gated(cols: list[int], split=(k, c)) -> Program | None:
            shared, left, right = parts(cols)
            if len(_pairs(left, right)) > settings.max_ifelse:
                return None
            return build_program(shared, left, right, split)

        def gated_ok(cols: list[int]) -> bool:
            program = build_gated(cols)
            return program is not None and program.count_unique_nodes() <= settings.node_cap

        pool = gated_pool((X[:, k] > c).astype(np.float64))
        pruned = run_pool(pool, build_gated, gated_ok, "piecewise")
        if not pruned:
            continue
        shared, left, right = parts(pruned)
        if not (left or right):
            continue
        # Refine the threshold of the pruned program on all training rows.
        best_c, best_sse, used = refine_threshold(
            X[:, k], y, lambda gate, cols=pruned: gated_pool(gate)[:, cols], c, min_child)
        fits += used
        if abs(best_c - c) > 0:
            record(build_program(shared, left, right, (k, best_c)), best_sse, "piecewise_refined")
    return sorted(results.values(), key=lambda item: (item[0], item[2].count_unique_nodes())), fits
