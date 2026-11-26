from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

EPS = 1e-8


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.where(np.abs(b) < EPS, np.where(b >= 0.0, EPS, -EPS), b)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = a / denom
    return out


def _safe_log(a: np.ndarray) -> np.ndarray:
    return np.log(np.abs(a) + EPS)


def _safe_exp(a: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(a, -30.0, 30.0))


def _sanitize(a: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(a, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6).astype(np.float32)


@dataclass
class Node:
    op: str
    children: list["Node"] = field(default_factory=list)
    feature_idx: int | None = None
    constant: float | None = None

    def evaluate(
        self,
        X: np.ndarray,
        track_ifelse: bool = False,
        track_buffer: list[np.ndarray] | None = None,
        leaf_buffer: dict[int, np.ndarray] | None = None,
        _row_indices: np.ndarray | None = None,
        _root_size: int | None = None,
    ) -> np.ndarray:
        rows = np.arange(len(X)) if _row_indices is None else _row_indices
        root_size = len(X) if _root_size is None else _root_size
        if len(X) == 0:
            # Reserve unreachable branch coordinates without executing their
            # predicates or arithmetic. Schema remains fixed across rows.
            if track_ifelse and track_buffer is not None:
                track_buffer.extend(np.zeros(root_size, dtype=np.float32)
                                    for _ in range(self.count_ifelse()))
            return np.zeros(0, dtype=np.float32)

        def child_value(child: "Node") -> np.ndarray:
            return child.evaluate(X, track_ifelse, track_buffer, leaf_buffer, rows, root_size)

        if self.op == "var":
            if leaf_buffer is not None:
                feature = int(self.feature_idx)  # type: ignore[arg-type]
                if feature not in leaf_buffer:
                    leaf_buffer[feature] = np.zeros(root_size, dtype=bool)
                leaf_buffer[feature][rows] = True
            return _sanitize(X[:, self.feature_idx])  # type: ignore[index]
        if self.op == "const":
            value = 0.0 if self.constant is None else self.constant
            return _sanitize(np.full(X.shape[0], value, dtype=np.float32))

        if self.op == "sin":
            return _sanitize(np.sin(child_value(self.children[0])))
        if self.op == "cos":
            return _sanitize(np.cos(child_value(self.children[0])))
        if self.op == "exp":
            return _sanitize(_safe_exp(child_value(self.children[0])))
        if self.op == "log":
            return _sanitize(_safe_log(child_value(self.children[0])))

        if self.op in {"+", "-", "*", "/"}:
            a = child_value(self.children[0])
            b = child_value(self.children[1])
            if self.op == "+":
                return _sanitize(a + b)
            if self.op == "-":
                return _sanitize(a - b)
            if self.op == "*":
                return _sanitize(a * b)
            return _sanitize(_safe_div(a, b))

        if self.op in {"gt", "lt", "eq"}:
            a = child_value(self.children[0])
            b = child_value(self.children[1])
            if self.op == "gt":
                return _sanitize((a > b).astype(np.float32))
            if self.op == "lt":
                return _sanitize((a < b).astype(np.float32))
            return _sanitize((np.abs(a - b) < 1e-4).astype(np.float32))

        if self.op == "ifelse":
            branch_column = None
            if track_ifelse and track_buffer is not None:
                # Preorder assigns this node before any branch in its predicate.
                branch_column = np.zeros(root_size, dtype=np.float32)
                track_buffer.append(branch_column)
            cond = child_value(self.children[0]) > 0.5
            if branch_column is not None:
                branch_column[rows] = cond
            result = np.empty(len(X), dtype=np.float32)
            for child, active in [(self.children[1], cond), (self.children[2], ~cond)]:
                result[active] = child.evaluate(X[active], track_ifelse, track_buffer,
                                                leaf_buffer, rows[active], root_size)
            return _sanitize(result)

        raise ValueError(f"Unsupported op: {self.op}")

    def collect_feature_indices(self) -> set[int]:
        out: set[int] = set()
        if self.op == "var" and self.feature_idx is not None:
            out.add(self.feature_idx)
        for child in self.children:
            out |= child.collect_feature_indices()
        return out

    def count_nodes(self) -> int:
        return 1 + sum(c.count_nodes() for c in self.children)

    def count_ifelse(self) -> int:
        cnt = 1 if self.op == "ifelse" else 0
        return cnt + sum(c.count_ifelse() for c in self.children)

    def to_string(self) -> str:
        if self.op == "var":
            return f"x{self.feature_idx}"
        if self.op == "const":
            c = 0.0 if self.constant is None else self.constant
            return f"{c:.4g}"
        if self.op in {"sin", "cos", "exp", "log"}:
            return f"{self.op}({self.children[0].to_string()})"
        if self.op in {"+", "-", "*", "/", "gt", "lt", "eq"}:
            return f"({self.children[0].to_string()} {self.op} {self.children[1].to_string()})"
        if self.op == "ifelse":
            return (
                f"ifelse({self.children[0].to_string()},"
                f" {self.children[1].to_string()}, {self.children[2].to_string()})"
            )
        return self.op

    def structural_signature(self) -> str:
        if self.op == "var":
            return f"v{self.feature_idx}"
        if self.op == "const":
            value = 0.0 if self.constant is None else float(self.constant)
            return f"c:{value:.12g}"
        if self.op in {"sin", "cos", "exp", "log"}:
            return f"{self.op}({self.children[0].structural_signature()})"
        if self.op in {"+", "*"}:
            left = self.children[0].structural_signature()
            right = self.children[1].structural_signature()
            a, b = sorted([left, right])
            return f"{self.op}({a},{b})"
        if self.op in {"-", "/", "gt", "lt", "eq"}:
            left = self.children[0].structural_signature()
            right = self.children[1].structural_signature()
            return f"{self.op}({left},{right})"
        if self.op == "ifelse":
            c = self.children[0].structural_signature()
            t = self.children[1].structural_signature()
            f = self.children[2].structural_signature()
            return f"ifelse({c},{t},{f})"
        return self.op

    def to_dict(self) -> dict:
        return {
            "op": self.op,
            "feature_idx": self.feature_idx,
            "constant": self.constant,
            "children": [c.to_dict() for c in self.children],
        }

    @staticmethod
    def from_dict(data: dict) -> "Node":
        children = [Node.from_dict(c) for c in data.get("children", [])]
        return Node(
            op=str(data.get("op")),
            children=children,
            feature_idx=data.get("feature_idx"),
            constant=data.get("constant"),
        )

    def replace_var(self, feature_idx: int, constant: float) -> "Node":
        if self.op == "var" and self.feature_idx == feature_idx:
            return Node(op="const", constant=float(constant))
        return Node(
            op=self.op,
            children=[c.replace_var(feature_idx, constant) for c in self.children],
            feature_idx=self.feature_idx,
            constant=self.constant,
        )


@dataclass
class Program:
    terms: list[Node]
    coeffs: np.ndarray | None = None
    intercept: float = 0.0
    coefficient_mode: str = "term"

    def __post_init__(self):
        if self.coefficient_mode not in ("term", "conditional_affine"):
            raise ValueError("Unknown coefficient mode")

    def coefficient_count(self) -> int:
        """Number of affine coefficient slots, including dormant output paths.

        Conditional coefficients descend only through leading ifelse nodes at
        each additive root. Predicates and arithmetic bodies keep native
        semantics; parameters are never inserted inside nonlinear operations.
        """
        def count(node):
            if self.coefficient_mode == "conditional_affine" and node.op == "ifelse":
                return count(node.children[1]) + count(node.children[2])
            return 1
        return sum(count(t) for t in self.terms)

    def _conditional_design_matrix(self, X, track_ifelse, leaf_buffer):
        values, tracks = [], []
        root_size = len(X)

        def visit(node, rows):
            if node.op == "ifelse":
                branch = None
                if track_ifelse:
                    branch = np.zeros(root_size, dtype=np.float32)
                    tracks.append(branch)
                # Preorder and short-circuit behavior match Node.evaluate,
                # including conditionals embedded in the predicate itself.
                route = node.children[0].evaluate(X[rows], track_ifelse, tracks,
                                                  leaf_buffer, rows, root_size) > .5
                if branch is not None:
                    branch[rows] = route
                visit(node.children[1], rows[route])
                visit(node.children[2], rows[~route])
            else:
                column = np.zeros(root_size, dtype=np.float32)
                column[rows] = node.evaluate(X[rows], track_ifelse, tracks,
                                             leaf_buffer, rows, root_size)
                values.append(column)

        for term in self.terms:
            visit(term, np.arange(root_size))
        Phi = np.column_stack(values).astype(np.float32) if values else np.zeros((root_size, 0), dtype=np.float32)
        T = np.column_stack(tracks).astype(np.float32) if tracks else np.zeros((root_size, 0), dtype=np.float32)
        return Phi, T

    def design_matrix(
        self,
        X: np.ndarray,
        track_ifelse: bool = False,
        leaf_buffer: dict[int, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.coefficient_mode == "conditional_affine":
            return self._conditional_design_matrix(X, track_ifelse, leaf_buffer)
        term_values: list[np.ndarray] = []
        track_cols: list[np.ndarray] = []
        for term in self.terms:
            local_tracks: list[np.ndarray] = []
            val = term.evaluate(X, track_ifelse=track_ifelse, track_buffer=local_tracks,
                                leaf_buffer=leaf_buffer)
            term_values.append(np.nan_to_num(val, nan=0.0, posinf=1e6, neginf=-1e6))
            if track_ifelse:
                track_cols.extend(local_tracks)
        Phi = np.stack(term_values, axis=1) if term_values else np.zeros((X.shape[0], 0), dtype=np.float32)
        if not track_cols:
            T = np.zeros((X.shape[0], 0), dtype=np.float32)
        else:
            T = np.stack(track_cols, axis=1).astype(np.float32)
        return Phi.astype(np.float32), T

    def predict(
        self,
        X: np.ndarray,
        track_ifelse: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        Phi, T = self.design_matrix(X, track_ifelse=track_ifelse)
        return self._combine_terms(Phi), T

    def _combine_terms(self, Phi: np.ndarray) -> np.ndarray:
        if self.coeffs is None:
            coeffs = np.ones(Phi.shape[1], dtype=np.float32)
            intercept = 0.0
        else:
            coeffs = self.coeffs.astype(np.float32)
            intercept = float(self.intercept)
        pred = intercept + Phi @ coeffs
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        return pred

    def execution_trace(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prediction, masked unique input coordinates, and preorder branches.

        Each serialized term is evaluated. A coordinate is active if any of
        its occurrences is evaluated, including occurrences in predicates.
        Protected numerical operators match ordinary prediction semantics.
        """
        active: dict[int, np.ndarray] = {}
        Phi, branches = self.design_matrix(X, track_ifelse=True, leaf_buffer=active)
        columns = [np.where(active.get(j, np.zeros(len(X), dtype=bool)), X[:, j], 0)
                   for j in sorted(self.used_features())]
        leaves = np.column_stack(columns).astype(np.float32) if columns else np.zeros((len(X), 0), dtype=np.float32)
        return self._combine_terms(Phi), leaves, branches

    def count_nodes(self) -> int:
        return sum(t.count_nodes() for t in self.terms)

    def count_unique_nodes(self) -> int:
        """Count a shared-subexpression DAG instead of repeated tree expansion."""
        signatures: set[str] = set()

        def _visit(node: Node) -> None:
            signatures.add(node.structural_signature())
            for child in node.children:
                _visit(child)

        for term in self.terms:
            _visit(term)
        return len(signatures)

    def count_ifelse(self) -> int:
        return sum(t.count_ifelse() for t in self.terms)

    def used_features(self) -> set[int]:
        feats: set[int] = set()
        for t in self.terms:
            feats |= t.collect_feature_indices()
        return feats

    def equation(self) -> str:
        if self.coefficient_mode == "conditional_affine":
            coefficients = iter(self.coeffs.tolist() if self.coeffs is not None else [1.] * self.coefficient_count())
            def render(node):
                if node.op == "ifelse":
                    return f"ifelse({node.children[0].to_string()}, {render(node.children[1])}, {render(node.children[2])})"
                return f"({next(coefficients):.5g})*{node.to_string()}"
            offset = float(self.intercept) if self.coeffs is not None else 0.
            return " + ".join([f"{offset:.5g}", *[render(t) for t in self.terms]])
        if self.coeffs is None:
            coeffs = [1.0] * len(self.terms)
            intercept = self.intercept
        else:
            coeffs = self.coeffs.tolist()
            intercept = self.intercept
        parts = [f"{intercept:.5g}"]
        for c, t in zip(coeffs, self.terms):
            parts.append(f"({c:.5g})*{t.to_string()}")
        return " + ".join(parts)

    def structural_hash(self) -> str:
        sigs = sorted(t.structural_signature() for t in self.terms)
        prefix = "conditional_affine|" if self.coefficient_mode == "conditional_affine" else ""
        return prefix + "|".join(sigs)

    def clone(self) -> "Program":
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        out = {
            "terms": [t.to_dict() for t in self.terms],
            "coeffs": self.coeffs.tolist() if self.coeffs is not None else None,
            "intercept": float(self.intercept),
        }
        if self.coefficient_mode != "term":
            out["coefficient_mode"] = self.coefficient_mode
        return out

    @staticmethod
    def from_dict(data: dict) -> "Program":
        terms = [Node.from_dict(t) for t in data.get("terms", [])]
        p = Program(terms=terms, coefficient_mode=data.get("coefficient_mode", "term"))
        coeffs = data.get("coeffs")
        if coeffs is not None:
            p.coeffs = np.asarray(coeffs, dtype=np.float32).reshape(-1)
        p.intercept = float(data.get("intercept", 0.0))
        return p

    def intervene_feature(self, feature_idx: int, value: float) -> "Program":
        out = self.clone()
        out.terms = [t.replace_var(feature_idx, float(value)) for t in out.terms]
        return out


UNARY_OPS = ["sin", "cos", "exp", "log"]
BINARY_OPS = ["+", "-", "*", "/"]
COND_OPS = ["gt", "lt"]


def _rand_feature(rng: np.random.Generator, feature_indices: Iterable[int]) -> Node:
    fid = int(rng.choice(list(feature_indices)))
    return Node(op="var", feature_idx=fid)


def sample_random_tree(
    rng: np.random.Generator,
    feature_indices: list[int],
    max_depth: int,
    ifelse_prob: float,
    unary_prob: float = 0.25,
    unary_ops: list[str] | None = None,
    binary_ops: list[str] | None = None,
    cond_ops: list[str] | None = None,
) -> Node:
    u_ops = unary_ops if unary_ops else UNARY_OPS
    b_ops = binary_ops if binary_ops else BINARY_OPS
    c_ops = cond_ops if cond_ops else COND_OPS

    if max_depth <= 0 or rng.random() < 0.25:
        return _rand_feature(rng, feature_indices)

    if rng.random() < ifelse_prob and max_depth >= 2:
        cond_op = str(rng.choice(c_ops))
        cond_left = sample_random_tree(
            rng,
            feature_indices,
            max_depth - 2,
            ifelse_prob=0.0,
            unary_ops=u_ops,
            binary_ops=b_ops,
            cond_ops=c_ops,
        )
        cond_right = _rand_feature(rng, feature_indices)
        cond = Node(cond_op, children=[cond_left, cond_right])
        t = sample_random_tree(
            rng,
            feature_indices,
            max_depth - 1,
            ifelse_prob=ifelse_prob * 0.5,
            unary_ops=u_ops,
            binary_ops=b_ops,
            cond_ops=c_ops,
        )
        f = sample_random_tree(
            rng,
            feature_indices,
            max_depth - 1,
            ifelse_prob=ifelse_prob * 0.5,
            unary_ops=u_ops,
            binary_ops=b_ops,
            cond_ops=c_ops,
        )
        return Node("ifelse", children=[cond, t, f])

    if rng.random() < unary_prob and u_ops:
        op = str(rng.choice(u_ops))
        child = sample_random_tree(
            rng,
            feature_indices,
            max_depth - 1,
            ifelse_prob=ifelse_prob,
            unary_ops=u_ops,
            binary_ops=b_ops,
            cond_ops=c_ops,
        )
        return Node(op, children=[child])

    op = str(rng.choice(b_ops))
    left = sample_random_tree(
        rng,
        feature_indices,
        max_depth - 1,
        ifelse_prob=ifelse_prob,
        unary_ops=u_ops,
        binary_ops=b_ops,
        cond_ops=c_ops,
    )
    right = sample_random_tree(
        rng,
        feature_indices,
        max_depth - 1,
        ifelse_prob=ifelse_prob,
        unary_ops=u_ops,
        binary_ops=b_ops,
        cond_ops=c_ops,
    )
    return Node(op, children=[left, right])
