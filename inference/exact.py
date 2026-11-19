"""Single-node variable elimination on Part C's ordered discrete CPTs."""
from math import prod
from numbers import Integral

import numpy as np


def validate_query(model, target, evidence):
    if target not in model.dag:
        raise ValueError(f"Unknown target: {target}")
    for node, state in evidence.items():
        if node not in model.card:
            raise ValueError(f"Unknown evidence node: {node}")
        if (isinstance(state, bool) or not isinstance(state, Integral)
                or not 0 <= state < model.card[node]):
            raise ValueError(f"{node} requires an integer state in 0..{model.card[node]-1}")


def _contract(factors, keep):
    # Integer einsum labels align axes by variable, including differently ordered CPTs.
    labels = {}
    operands = []
    for scope, values in factors:
        axes = [labels.setdefault(node, len(labels)) for node in scope]
        operands.extend((values, axes))
    return np.einsum(*operands, [labels[node] for node in keep])


def query_ve(model, target, evidence):
    """Return P(target | evidence) in schema order. Omit unobserved variables."""
    validate_query(model, target, evidence)
    factors = []
    for node, parents in model.dag.items():
        scope = (*parents, node)
        shape = tuple(model.card[v] for v in scope)
        index = tuple(evidence.get(v, slice(None)) for v in scope)
        free = tuple(v for v in scope if v not in evidence)
        factors.append((free, model.cpts[node].reshape(shape)[index]))

    hidden = [v for v in model.dag if v != target and v not in evidence]
    while hidden:
        # Prefer the smallest product of state counts in the current bucket.
        # Model node order breaks ties deterministically.
        scopes = {v: set().union(*(s for s, _ in factors if v in s)) for v in hidden}
        eliminate = min(hidden, key=lambda v: prod(model.card[x] for x in scopes[v]))
        bucket = [f for f in factors if eliminate in f[0]]
        factors = [f for f in factors if eliminate not in f[0]]
        keep = tuple(v for v in model.dag if v != eliminate and v in scopes[eliminate])
        factors.append((keep, _contract(bucket, keep)))
        hidden.remove(eliminate)

    keep = () if target in evidence else (target,)
    masses = _contract(factors, keep)
    total = masses.sum()
    if total <= 0:
        raise ValueError("Evidence has zero probability under the model")
    if target in evidence:
        result = np.zeros(model.card[target])
        result[evidence[target]] = 1
        return result
    return masses / total
