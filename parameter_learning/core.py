"""Exact, smoothed discrete Bayesian-network parameter learning.

CPT layout matches structure/bn.py: (parent configurations, child states).
Parents are used in the supplied order, with the last parent varying fastest.
Missing values are marginalized, never treated as an extra category.
"""
from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd


def _validate_structure(dag, card):
    dag = {v: tuple(ps) for v, ps in dag.items()}
    if not dag or set(dag) != set(card):
        raise ValueError("dag and card must contain exactly the same nonempty node set")
    for v, ps in dag.items():
        if (not isinstance(card[v], (int, np.integer)) or card[v] < 1
                or len(set(ps)) != len(ps) or v in ps or not set(ps) <= set(dag)):
            raise ValueError(f"Invalid cardinality or parents for {v}")
    done = set()
    while len(done) < len(dag):
        ready = {v for v, ps in dag.items() if v not in done and set(ps) <= done}
        if not ready:
            raise ValueError("dag contains a directed cycle")
        done |= ready
    return dag, {v: int(card[v]) for v in dag}


def _array(data, nodes, card, latent_nodes=()):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame with named columns")
    if not data.columns.is_unique:
        raise ValueError("DataFrame columns must be unique")
    absent = set(nodes) - set(data.columns)
    if not set(latent_nodes) <= set(nodes) or not absent <= set(latent_nodes):
        raise ValueError(f"Missing model columns: {sorted(absent - set(latent_nodes))}")
    a = np.full((len(data), len(nodes)), -1, dtype=np.int64)
    for j, v in enumerate(nodes):
        if v not in data:
            continue
        x = data[v].to_numpy(dtype=float, na_value=np.nan)
        observed = ~np.isnan(x)
        z = x[observed]
        if np.any(~np.isfinite(z)) or np.any(z != np.floor(z)) or np.any((z < 0) | (z >= card[v])):
            raise ValueError(f"{v}: expected integer states 0..{card[v]-1} or NA")
        a[observed, j] = z.astype(np.int64)
    return a


def _shape(v, dag, card):
    return (int(np.prod([card[p] for p in dag[v]], dtype=np.int64)), card[v])


def _family_index(a, v, dag, card, positions):
    idx = np.zeros(len(a), dtype=np.int64)
    for p in (*dag[v], v):
        idx = idx * card[p] + a[:, positions[p]]
    return idx


def _normalize(counts, alpha, previous=None):
    n = counts.astype(float) + alpha
    sums = n.sum(axis=1, keepdims=True)
    # For pure ML, unobserved parent configurations are unconstrained. Keep the
    # previous row in EM, or use a documented uniform row for complete data.
    fallback = np.full_like(n, 1 / n.shape[1]) if previous is None else previous
    return np.divide(n, sums, out=fallback.copy(), where=sums > 0)


def _alpha(alpha):
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be a finite, nonnegative per-cell pseudocount")
    return float(alpha)


class _Completions:
    """Compress repeated evidence and enumerate only its missing variables.

    This is exact enumeration, not VE. A hard budget rejects infeasible masks
    before allocating their Cartesian products. Complete rows have one term.
    """
    def __init__(self, model, data, latent_nodes=(), max_completions=2_000_000):
        a = _array(data, list(model.dag), model.card, latent_nodes)
        self.n_rows = len(a)
        if not len(a):
            raise ValueError("Empty data")
        unique, self.inverse, self.frequency = np.unique(a, axis=0, return_inverse=True, return_counts=True)
        self.unique = unique
        cards = np.array(list(model.card.values()))
        # Python integers avoid overflow for unusually large models.
        sizes = np.array([int(np.prod([int(cards[j]) for j in np.flatnonzero(row < 0)], dtype=object))
                          for row in unique], dtype=object)
        total = sum(sizes)
        if total > max_completions:
            raise ValueError(f"Exact EM requires {total:,} completions; budget is {max_completions:,}. "
                             "Use fewer masked variables, a smaller batch/model, or a VE-based E-step.")
        sizes = sizes.astype(np.int64)
        self.starts = np.r_[0, np.cumsum(sizes)[:-1]]
        self.groups = np.repeat(np.arange(len(unique)), sizes)
        self.completed = np.repeat(unique, sizes, axis=0)
        masks, mask_inverse = np.unique(unique < 0, axis=0, return_inverse=True)
        for k, mask in enumerate(masks):
            missing = np.flatnonzero(mask)
            if not len(missing):
                continue
            ids = np.flatnonzero(mask_inverse == k)
            combos = np.array(list(product(*(range(cards[j]) for j in missing))), dtype=np.int64)
            indices = (self.starts[ids, None] + np.arange(len(combos))).ravel()
            self.completed[np.ix_(indices, missing)] = np.tile(combos, (len(ids), 1))
        positions = {v: i for i, v in enumerate(model.dag)}
        self.indices = {v: _family_index(self.completed, v, model.dag, model.card, positions) for v in model.dag}

    def expectation(self, cpts):
        log_joint = np.zeros(len(self.completed))
        with np.errstate(divide="ignore"):
            for v, idx in self.indices.items():
                log_joint += np.log(cpts[v].ravel()[idx])
        maximum = np.maximum.reduceat(log_joint, self.starts)
        if not np.isfinite(maximum).all():
            raise ValueError("Evidence has zero model probability; use positive smoothing/initial CPTs")
        unnormalized = np.exp(log_joint - maximum[self.groups])
        sums = np.add.reduceat(unnormalized, self.starts)
        log_evidence = maximum + np.log(sums)
        posterior = unnormalized / sums[self.groups]
        return log_evidence, posterior

    def counts(self, model, posterior):
        weights = posterior * self.frequency[self.groups]
        return {v: np.bincount(idx, weights=weights, minlength=model.cpts[v].size).reshape(model.cpts[v].shape)
                for v, idx in self.indices.items()}


@dataclass
class DiscreteBN:
    dag: dict
    card: dict
    cpts: dict

    def __post_init__(self):
        self.dag, self.card = _validate_structure(self.dag, self.card)
        if set(self.cpts) != set(self.dag):
            raise ValueError("CPT node set does not match DAG")
        self.cpts = {v: np.asarray(self.cpts[v], dtype=float).copy() for v in self.dag}
        for v, t in self.cpts.items():
            if (t.shape != _shape(v, self.dag, self.card) or not np.isfinite(t).all()
                    or np.any(t < 0) or not np.allclose(t.sum(axis=1), 1, rtol=0, atol=1e-10)):
                raise ValueError(f"Invalid CPT for {v}")

    def log_likelihood(self, data, *, latent_nodes=(), max_completions=2_000_000):
        """Per-row log P(observed cells); integrate all NA and declared absent latent nodes."""
        cache = _Completions(self, data, latent_nodes, max_completions)
        ll, _ = cache.expectation(self.cpts)
        return ll[cache.inverse]

    def posterior(self, data, node, *, latent_nodes=(), max_completions=2_000_000):
        """Return (rows, card[node]) exact posterior; row order matches input.

        Set the query column to NA (or declare it latent and omit it) to infer
        it. An observed query is correctly returned as a point mass.
        """
        if node not in self.dag:
            raise ValueError(f"Unknown query node: {node}")
        cache = _Completions(self, data, latent_nodes, max_completions)
        _, p = cache.expectation(self.cpts)
        states = cache.completed[:, list(self.dag).index(node)]
        flat = cache.groups * self.card[node] + states
        result = np.bincount(flat, weights=p, minlength=len(cache.unique) * self.card[node])
        return result.reshape(-1, self.card[node])[cache.inverse]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"format_version": 1, "parents": self.dag,
                                    "cardinality": self.card,
                                    "cpts": {v: t.tolist() for v, t in self.cpts.items()}}, indent=2), encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path):
        x = json.loads(Path(path).read_text(encoding="utf-8"))
        if x.get("format_version") != 1:
            raise ValueError("Unsupported model format")
        return cls(x["parents"], x["cardinality"], x["cpts"])


def fit_mle(dag, data, card, alpha=1.0, *, missing="raise"):
    """Add-alpha CPTs. missing='complete_case' explicitly deletes incomplete rows.

    alpha=0 gives ordinary ML (uniform rows for unseen parent configurations).
    alpha>0 gives add-alpha smoothing, equivalently MAP under Dirichlet(1+alpha).
    """
    dag, card = _validate_structure(dag, card)
    alpha = _alpha(alpha)
    a = _array(data, list(dag), card)
    if missing not in ("raise", "complete_case"):
        raise ValueError("missing must be 'raise' or 'complete_case'")
    if missing == "complete_case":
        a = a[(a >= 0).all(axis=1)]
    if not len(a) or np.any(a < 0):
        raise ValueError("MLE requires nonempty complete rows; use EM or missing='complete_case'")
    positions = {v: j for j, v in enumerate(dag)}
    cpts = {}
    for v in dag:
        shape = _shape(v, dag, card)
        idx = _family_index(a, v, dag, card, positions)
        n = np.bincount(idx, minlength=int(np.prod(shape))).reshape(shape)
        cpts[v] = _normalize(n, alpha)
    return DiscreteBN(dag, card, cpts)


@dataclass
class EMResult:
    model: DiscreteBN
    history: pd.DataFrame
    converged: bool
    seed: int
    n_iter: int

    @property
    def cpts(self):
        return self.model.cpts


def fit_em(dag, data, card, alpha=1.0, *, latent_nodes=(), seed=0,
           max_iter=500, tol=1e-7, init_cpts=None, max_completions=2_000_000):
    """Exact EM, one reproducible random start; retains every training row.

    The E-step enumerates missing states jointly conditional on all observed
    cells. M-step: theta=(expected_count+alpha)/(expected_parent_count+r*alpha).
    Tracked objective: observed LL + alpha * sum(log theta). Its increase per
    training row must fall below tol to stop. History includes iteration zero.
    Ordinary observed LL need not increase when alpha>0; the objective must.
    Rows missing every variable contribute no likelihood information.
    Run several seeds, choose by validation LL, and evaluate test only afterward.
    """
    dag, card = _validate_structure(dag, card)
    alpha = _alpha(alpha)
    if not isinstance(max_iter, int) or max_iter < 1 or not np.isfinite(tol) or tol < 0:
        raise ValueError("max_iter must be positive and tol finite and nonnegative")
    rng = np.random.default_rng(seed)
    cpts = ({v: rng.dirichlet(np.ones(card[v]), size=_shape(v, dag, card)[0]) for v in dag}
            if init_cpts is None else init_cpts)
    model = DiscreteBN(dag, card, cpts)
    if alpha > 0 and any(np.any(t <= 0) for t in model.cpts.values()):
        raise ValueError("Positive smoothing requires strictly positive initial CPTs")
    cache = _Completions(model, data, latent_nodes, max_completions)
    records, converged, started = [], False, time.perf_counter()
    for iteration in range(max_iter + 1):
        ll, posterior = cache.expectation(model.cpts)
        total_ll = float(ll @ cache.frequency)
        prior = alpha * sum(float(np.log(t).sum()) for t in model.cpts.values()) if alpha else 0.0
        objective = total_ll + prior
        delta = objective - records[-1]["objective"] if records else np.nan
        records.append({"iteration": iteration, "observed_ll": total_ll,
                        "ll_per_row": total_ll / len(data), "log_prior_penalty": prior,
                        "objective": objective, "objective_gain_per_row": delta / len(data),
                        "elapsed_seconds": time.perf_counter() - started})
        if iteration and delta < -1e-10 * max(1.0, abs(objective)):
            raise RuntimeError("EM objective decreased beyond numerical tolerance")
        if iteration and abs(delta) / len(data) <= tol:
            converged = True
            break
        if iteration == max_iter:
            break
        counts = cache.counts(model, posterior)
        model.cpts = {v: _normalize(counts[v], alpha, model.cpts[v]) for v in dag}
    return EMResult(model, pd.DataFrame(records), converged, int(seed), iteration)
