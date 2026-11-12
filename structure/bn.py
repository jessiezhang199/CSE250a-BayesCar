"""Discrete Bayesian network utilities for E1 (structure learning).

A DAG is a dict {node: sorted tuple of parents}. Data is a complete-case
integer DataFrame whose columns follow dataset/prepared/schema.json.
"""
import json
from itertools import permutations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "dataset" / "prepared"


# ---------- data ----------

def load_schema():
    schema = json.loads((PREPARED / "schema.json").read_text())["columns"]
    return {col: len(info["states"]) for col, info in schema.items()}


def load_split(name, columns):
    """Load one split, keep rows complete on `columns`. Returns (frame, n_raw)."""
    df = pd.read_csv(PREPARED / f"{name}.csv", index_col="source_row", dtype="Int64")
    complete = df[columns].dropna().astype(int)
    return complete, len(df)


# ---------- graph helpers ----------

def dag_from_edges(nodes, edges):
    parents = {v: [] for v in nodes}
    for u, v in edges:
        parents[v].append(u)
    dag = {v: tuple(sorted(ps)) for v, ps in parents.items()}
    assert nx.is_directed_acyclic_graph(to_nx(dag)), "edges contain a cycle"
    return dag


def edges_of(dag):
    return sorted((u, v) for v, ps in dag.items() for u in ps)


def to_nx(dag):
    g = nx.DiGraph()
    g.add_nodes_from(dag)
    g.add_edges_from(edges_of(dag))
    return g


def n_params(dag, card):
    return sum(
        (card[v] - 1) * int(np.prod([card[p] for p in ps], dtype=np.int64))
        for v, ps in dag.items()
    )


# ---------- counts, scores, likelihood ----------

def family_counts(df, card, node, parents):
    """Counts with shape (n_parent_configs, card[node])."""
    q = int(np.prod([card[p] for p in parents], dtype=np.int64))
    idx = np.zeros(len(df), dtype=np.int64)
    for p in parents:
        idx = idx * card[p] + df[p].to_numpy()
    flat = idx * card[node] + df[node].to_numpy()
    return np.bincount(flat, minlength=q * card[node]).reshape(q, card[node])


class BIC:
    """Decomposable BIC score with a family cache. Higher is better."""

    def __init__(self, df, card):
        self.df, self.card, self.cache = df, card, {}
        self.penalty = 0.5 * np.log(len(df))

    def family(self, node, parents):
        key = (node, tuple(sorted(parents)))
        if key not in self.cache:
            n = family_counts(self.df, self.card, node, key[1])
            row = n.sum(axis=1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                ll = np.where(n > 0, n * np.log(n / row), 0.0).sum()
            k = (n.shape[1] - 1) * n.shape[0]
            self.cache[key] = ll - self.penalty * k
        return self.cache[key]

    def total(self, dag):
        return sum(self.family(v, ps) for v, ps in dag.items())


def fit_cpts(dag, df, card, alpha=1.0):
    """Smoothed CPTs: P(x | pa) = (N + alpha) / (N_pa + alpha * card[x])."""
    cpts = {}
    for v, ps in dag.items():
        n = family_counts(df, card, v, ps) + alpha
        cpts[v] = n / n.sum(axis=1, keepdims=True)
    return cpts


def log_likelihood(dag, cpts, df, card):
    """Per-row joint log-likelihood (nats)."""
    ll = np.zeros(len(df))
    for v, ps in dag.items():
        idx = np.zeros(len(df), dtype=np.int64)
        for p in ps:
            idx = idx * card[p] + df[p].to_numpy()
        ll += np.log(cpts[v][idx, df[v].to_numpy()])
    return ll


# ---------- structure learning ----------

def hill_climb(score, nodes, start=None, max_parents=4):
    """Greedy search over add / delete / reverse moves. Returns (dag, trace)."""
    pa = {v: set(start[v]) if start else set() for v in nodes}
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from((u, v) for v in nodes for u in pa[v])
    fam = lambda v, ps: score.family(v, ps)
    trace = [score.total({v: tuple(pa[v]) for v in nodes})]

    while True:
        best, best_delta = None, 1e-9
        for u, v in permutations(nodes, 2):
            if u in pa[v]:
                d_del = fam(v, pa[v] - {u}) - fam(v, pa[v])
                if d_del > best_delta:
                    best, best_delta = ("del", u, v), d_del
                if len(pa[u]) < max_parents:
                    g.remove_edge(u, v)
                    ok = not nx.has_path(g, u, v)
                    g.add_edge(u, v)
                    if ok:
                        d_rev = d_del + fam(u, pa[u] | {v}) - fam(u, pa[u])
                        if d_rev > best_delta:
                            best, best_delta = ("rev", u, v), d_rev
            elif v not in pa[u] and len(pa[v]) < max_parents and not nx.has_path(g, v, u):
                d_add = fam(v, pa[v] | {u}) - fam(v, pa[v])
                if d_add > best_delta:
                    best, best_delta = ("add", u, v), d_add
        if best is None:
            break
        op, u, v = best
        if op in ("del", "rev"):
            pa[v].remove(u)
            g.remove_edge(u, v)
        if op == "add":
            pa[v].add(u)
            g.add_edge(u, v)
        if op == "rev":
            pa[u].add(v)
            g.add_edge(v, u)
        trace.append(trace[-1] + best_delta)
    return {v: tuple(sorted(pa[v])) for v in nodes}, trace


def random_dag(nodes, n_edges, max_parents, rng):
    order = list(rng.permutation(nodes))
    pairs = [(order[i], order[j]) for i in range(len(order)) for j in range(i + 1, len(order))]
    pa = {v: [] for v in nodes}
    for k in rng.permutation(len(pairs)):
        u, v = pairs[k]
        if len(pa[v]) < max_parents and sum(map(len, pa.values())) < n_edges:
            pa[v].append(u)
    return {v: tuple(sorted(ps)) for v, ps in pa.items()}


def cond_mutual_info(df, card, x, y, z=()):
    """Empirical I(X; Y | Z) in nats and the G-test degrees of freedom."""
    n = family_counts(df, card, x, (*z, y)).reshape(-1, card[y], card[x]).astype(float)
    nz = n.sum(axis=(1, 2), keepdims=True)
    expected = n.sum(axis=2, keepdims=True) * n.sum(axis=1, keepdims=True) / np.where(nz > 0, nz, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cmi = np.where(n > 0, n * np.log(n / expected), 0.0).sum() / len(df)
    dof = (card[x] - 1) * (card[y] - 1) * int((nz > 0).sum())
    return cmi, dof


def chow_liu(df, card, nodes, root):
    """Maximum mutual-information spanning tree, edges directed away from root."""
    g = nx.Graph()
    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            g.add_edge(u, v, weight=cond_mutual_info(df, card, u, v)[0])
    tree = nx.maximum_spanning_tree(g)
    return dag_from_edges(nodes, nx.bfs_edges(tree, root))


def naive_bayes(nodes, target):
    return dag_from_edges(nodes, [(target, v) for v in nodes if v != target])


# ---------- conditional independence checks ----------

def local_markov_tests(dag, df, card):
    """G-test of X indep Y | Pa(X) for every non-descendant, non-parent Y of X."""
    g, n, rows, seen = to_nx(dag), len(df), [], set()
    for x in dag:
        blocked = nx.descendants(g, x) | set(dag[x]) | {x}
        for y in dag:
            if y in blocked:
                continue
            key = (frozenset((x, y)), dag[x])
            if key in seen:
                continue
            seen.add(key)
            cmi, dof = cond_mutual_info(df, card, x, y, dag[x])
            rows.append({
                "x": x, "y": y, "given": ", ".join(dag[x]),
                "cmi_nats": cmi, "g_stat": 2 * n * cmi, "dof": dof,
                "p_value": chi2.sf(2 * n * cmi, dof),
            })
    return pd.DataFrame(rows).sort_values("cmi_nats", ascending=False)


def shd(a, b):
    """Structural Hamming distance: missing, extra and reversed edges each count 1."""
    ea, eb = set(edges_of(a)), set(edges_of(b))
    ska, skb = {frozenset(e) for e in ea}, {frozenset(e) for e in eb}
    reversed_ = sum(1 for u, v in ea if (v, u) in eb)
    return len(ska ^ skb) + reversed_, len(ska & skb), reversed_
