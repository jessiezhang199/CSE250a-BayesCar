"""Likelihood weighting, rejection, and single-site Gibbs posterior estimates."""
from dataclasses import dataclass
from numbers import Integral

import networkx as nx
import numpy as np

from structure.bn import to_nx
from .exact import query_ve, validate_query


@dataclass
class SampleResult:
    posterior: np.ndarray | None
    weight_ess: float | None = None
    accepted: int | None = None
    trace: np.ndarray | None = None


def _validate_count(count, minimum, name):
    if isinstance(count, bool) or not isinstance(count, Integral) or count < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _forward(model, evidence, n, rng, order):
    states = {}
    log_weights = np.zeros(n)
    for node in order:
        row = np.zeros(n, dtype=int)
        for parent in model.dag[node]:
            row = row * model.card[parent] + states[parent]
        cpt = model.cpts[node]
        if node in evidence:
            states[node] = np.full(n, evidence[node], dtype=int)
            with np.errstate(divide="ignore"):
                log_weights += np.log(cpt[row, evidence[node]])
        else:
            # The last state absorbs rounding error at the upper end of the CDF.
            cutoffs = np.cumsum(cpt, axis=1)[:, :-1]
            states[node] = (rng.random((n, 1)) >= cutoffs[row]).sum(axis=1)
    return states, log_weights


def likelihood_weighting(model, target, evidence, n, *, seed):
    validate_query(model, target, evidence)
    _validate_count(n, 1, "n")
    order = nx.topological_sort(to_nx(model.dag))
    states, logs = _forward(model, evidence, n, np.random.default_rng(seed), order)
    maximum = logs.max()
    if not np.isfinite(maximum):
        return SampleResult(None, weight_ess=0.0)
    weights = np.exp(logs - maximum)
    counts = np.bincount(states[target], weights=weights, minlength=model.card[target])
    return SampleResult(counts / counts.sum(), weight_ess=weights.sum()**2 / (weights @ weights))


def rejection_sampling(model, target, evidence, n, *, seed):
    validate_query(model, target, evidence)
    _validate_count(n, 1, "n")
    order = nx.topological_sort(to_nx(model.dag))
    states, _ = _forward(model, {}, n, np.random.default_rng(seed), order)
    matches = np.ones(n, dtype=bool)
    for node, value in evidence.items():
        matches &= states[node] == value
    accepted = int(matches.sum())
    if accepted == 0:
        return SampleResult(None, accepted=0)
    counts = np.bincount(states[target][matches], minlength=model.card[target])
    return SampleResult(counts / accepted, accepted=accepted)


def gibbs_sampling(model, target, evidence, n, *, seed, burn_in=0):
    """Systematic scans, one retained target state per post-burn-in sweep."""
    validate_query(model, target, evidence)
    _validate_count(n, 1, "n")
    _validate_count(burn_in, 0, "burn_in")
    rng = np.random.default_rng(seed)
    graph = to_nx(model.dag)
    order = list(nx.topological_sort(graph))
    free = [v for v in order if v not in evidence]
    initial, log_weight = _forward(model, evidence, 1, rng, order)
    state = {v: int(values[0]) for v, values in initial.items()}
    if not np.isfinite(log_weight[0]):
        # With structural zeros a clamped forward draw can start outside support.
        state = dict(evidence)
        query_ve(model, target, evidence)  # Detect impossible evidence even if all nodes are fixed.
        for v in free:
            state[v] = rng.choice(model.card[v], p=query_ve(model, v, state))

    # For each incident CPT, slice the updated node's axis and index every other
    # axis by its current state. A CPT can list its parents in any saved order.
    blankets = {}
    for node in free:
        terms = []
        for child in (node, *graph.successors(node)):
            scope = (*model.dag[child], child)
            axis = scope.index(node)
            tensor = model.cpts[child].reshape(tuple(model.card[v] for v in scope))
            others = scope[:axis] + scope[axis+1:]
            terms.append((np.moveaxis(tensor, axis, -1), others))
        blankets[node] = terms

    trace = np.empty(n, dtype=int)
    for sweep in range(burn_in + n):
        for node in free:
            conditional = np.ones(model.card[node])
            for table, others in blankets[node]:
                conditional *= table[tuple(state[v] for v in others)]
            cutoffs = conditional.cumsum()
            state[node] = int(np.searchsorted(cutoffs[:-1], rng.random() * cutoffs[-1], side="right"))
        if sweep >= burn_in:
            trace[sweep - burn_in] = state[target]
    counts = np.bincount(trace, minlength=model.card[target])
    return SampleResult(counts / n, trace=trace)
