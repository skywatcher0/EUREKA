"""Two-input gate proposals for the sparse expression pool."""
from itertools import combinations

import numpy as np

from eureka._method.families.blocked_selection import select_source_expression
from eureka._symbolic.core.ast import Node, Program
from eureka._symbolic.search.trace_repair import forward_path, term_basis, term_columns, term_node


def gate_candidates(X, y, seed):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.shape[1] < 2 or len(X) < 40:
        return []
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(X), min(len(X), 512), replace=False))
    screen = X[rows]
    response = y[rows]
    # Inputs already have unit variance on the source-training partition.
    relevance = np.abs((X - X.mean(0)).T @ (y - y.mean()))
    support = sorted(np.argsort(-relevance, kind="stable")[:8].tolist())
    design = np.column_stack([np.ones(len(rows)), screen[:, support]])
    scores = []
    for i, j in combinations(support, 2):
        for slope in (-2., -1., -.5, .5, 1., 2.):
            projection = screen[:, i] + slope * screen[:, j]
            for cut in np.unique(np.quantile(projection, [.2, .35, .5, .65, .8])):
                left = projection > cut
                if min(left.sum(), (~left).sum()) < max(20, .05 * len(rows)):
                    continue
                sse = 0.
                for mask in (left, ~left):
                    matrix, labels = design[mask], response[mask]
                    coeff = np.linalg.lstsq(matrix, labels, rcond=None)[0]
                    residual = labels - matrix @ coeff
                    sse += residual @ residual
                scores.append((sse, i, j, slope, cut))
    terms = term_basis(support, 2, 256)
    columns = term_columns(X, terms)
    proposals = []
    for _, i, j, slope, cut in sorted(scores)[:3]:
        mask = X[:, i] + slope * X[:, j] > cut
        split = Node("gt", children=[
            Node("+", children=[Node("var", feature_idx=i), Node("*", children=[
                Node("const", constant=slope), Node("var", feature_idx=j)])]),
            Node("const", constant=float(cut))])
        paths = [forward_path(columns[m], y[m], 2, lambda cols: True) for m in (mask, ~mask)]
        sides = [[terms[c] for c in path[-1][0]] if path else [] for path in paths]
        nodes = [Node("ifelse", children=[split, Node("const", constant=1.), Node("const", constant=0.)])]
        for k in range(max(map(len, sides))):
            nodes.append(Node("ifelse", children=[split, *[
                term_node(side[k] if k < len(side) else None) for side in sides]]))
            proposals.append(Program(terms=nodes.copy(), coefficient_mode="conditional_affine"))
    return proposals


def fit_oblique(X_train, y_train, env_train, X_val, y_val, env_val, seed=0):
    proposals = gate_candidates(X_train, y_train, seed)
    programs, info, _ = select_source_expression(proposals, X_train, y_train, env_train, X_val, y_val, env_val, seed)
    choice = info["choices"]["loeo"]["chosen"]
    if choice["origin"] == "search":
        choice["origin"] = "oblique"
    return programs["loeo"], info
