"""E1: expert DAG vs BIC hill-climbing (plus Chow-Liu, Naive Bayes, empty graph).

Run from the repo root: python structure/run_e1.py
All structures share the same variables, bins, complete-case rows and smoothing.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

import bn

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

cfg = json.loads((HERE / "config.json").read_text())
card = bn.load_schema()
nodes = list(card)
target, max_pa, alpha = cfg["target"], cfg["max_parents"], cfg["alpha"]

# ---------- data: same complete-case rule for every structure ----------
data, counts = {}, []
for split in ("train", "val", "test"):
    data[split], n_raw = bn.load_split(split, nodes)
    counts.append({"split": split, "rows": n_raw, "complete_rows": len(data[split]),
                   "retained_pct": round(100 * len(data[split]) / n_raw, 2)})
train = data["train"]
pd.DataFrame(counts).to_csv(OUT / "retained_counts.csv", index=False)
print(pd.DataFrame(counts).to_string(index=False), "\n")

# ---------- structures ----------
score = bn.BIC(train, card)
expert = bn.dag_from_edges(nodes, cfg["expert_dag"])
assert max(map(len, expert.values())) <= max_pa

starts = {"empty": None, "expert": expert}
for seed in cfg["hc_restart_seeds"]:
    rng = np.random.default_rng(seed)
    starts[f"random_seed{seed}"] = bn.random_dag(nodes, 15, max_pa, rng)

runs, hc_rows = {}, []
for name, start in starts.items():
    dag, trace = bn.hill_climb(score, nodes, start, max_pa)
    runs[name] = dag
    hc_rows.append({"start": name, "moves": len(trace) - 1, "edges": len(bn.edges_of(dag)),
                    "train_bic": trace[-1]})
hc_table = pd.DataFrame(hc_rows)
best_start = hc_table.loc[hc_table["train_bic"].idxmax(), "start"]
hc_table["shd_to_best"] = [bn.shd(runs[s], runs[best_start])[0] for s in hc_table["start"]]
hc_table.to_csv(OUT / "hc_restarts.csv", index=False)
print(hc_table.to_string(index=False), "\n")

dags = {
    "expert": expert,
    "hill_climb_bic": runs[best_start],
    "chow_liu": bn.chow_liu(train, card, nodes, target),
    "naive_bayes": bn.naive_bayes(nodes, target),
    "empty": bn.dag_from_edges(nodes, []),
}

# ---------- evaluation ----------
rows, test_ll = [], {}
for name, dag in dags.items():
    cpts = bn.fit_cpts(dag, train, card, alpha)
    row = {"structure": name, "edges": len(bn.edges_of(dag)),
           "max_in_degree": max(map(len, dag.values())),
           "params": bn.n_params(dag, card), "train_bic": score.total(dag)}
    for split in ("train", "val", "test"):
        ll = bn.log_likelihood(dag, cpts, data[split], card)
        row[f"{split}_ll_per_row"] = ll.mean()
        if split == "test":
            test_ll[name] = ll
            row["test_ll_total"] = ll.sum()
            row["test_ll_se"] = ll.std(ddof=1) / np.sqrt(len(ll))
    rows.append(row)
scores = pd.DataFrame(rows)
# paired per-record gain over the expert DAG on the same test rows
gains = {name: ll - test_ll["expert"] for name, ll in test_ll.items()}
scores["test_gain_vs_expert"] = [gains[s].mean() for s in scores["structure"]]
scores["test_gain_se"] = [gains[s].std(ddof=1) / np.sqrt(len(gains[s])) for s in scores["structure"]]
scores.to_csv(OUT / "e1_scores.csv", index=False)
print(scores.round(4).to_string(index=False), "\n")

learned = dags["hill_climb_bic"]
dist, shared, rev = bn.shd(expert, learned)
e_exp, e_hc = set(bn.edges_of(expert)), set(bn.edges_of(learned))
sk_exp, sk_hc = {frozenset(e) for e in e_exp}, {frozenset(e) for e in e_hc}
diff = {
    "shd": dist, "shared_skeleton_edges": shared, "reversed": rev,
    "same_direction": sorted(e_exp & e_hc),
    "reversed_in_learned": sorted(e for e in e_hc if e[::-1] in e_exp),
    "only_expert": sorted(e for e in e_exp if frozenset(e) not in sk_hc),
    "only_learned": sorted(e for e in e_hc if frozenset(e) not in sk_exp),
}
print(json.dumps(diff, indent=1), "\n")

(OUT / "dags.json").write_text(json.dumps({
    "nodes": nodes, "cardinality": card, "max_parents": max_pa, "alpha": alpha,
    "missing_rule": cfg["missing_rule"], "hc_best_start": best_start,
    "edges": {name: bn.edges_of(dag) for name, dag in dags.items()},
    "expert_vs_hill_climb": diff,
}, indent=2))

# ---------- conditional independence checks ----------
for name in ("expert", "hill_climb_bic"):
    ci = bn.local_markov_tests(dags[name], train, card)
    ci.to_csv(OUT / f"ci_tests_{name}.csv", index=False)
    print(f"{name}: {len(ci)} implied independencies, "
          f"{(ci.p_value < 0.01).sum()} rejected at 0.01, max CMI {ci.cmi_nats.max():.4f} nats")
    print(ci.head(6).round(5).to_string(index=False), "\n")

# ---------- figures ----------
POS = {
    "body_type": (0, 3), "brand_segment": (1.6, 3.3), "fuel_type": (3.3, 3.75), "vehicle_age": (5, 3),
    "engine_power_hp": (2.4, 1.9), "odometer_km": (3.9, 1.6), "number_of_previous_owners": (6, 1.9),
    "urban_rural": (0, 0.9), "accident_history": (4.8, 0.8), "service_history_quality": (5.8, 0.4),
    "listing_price": (1.2, 0), "overall_condition_score": (3.9, 0),
}
LABEL = {v: v.replace("number_of_", "").replace("_", "\n", 1).replace("_", " ") for v in nodes}


def draw(ax, dag, title, colors):
    g = bn.to_nx(dag)
    nx.draw_networkx_nodes(g, POS, ax=ax, node_size=2500, node_color="#eef2f7", edgecolors="#4a5568")
    nx.draw_networkx_labels(g, POS, LABEL, ax=ax, font_size=7)
    nx.draw_networkx_edges(g, POS, ax=ax, node_size=2500, arrowsize=14, width=1.4,
                           edge_color=[colors(e) for e in g.edges],
                           connectionstyle="arc3,rad=0.08")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-0.6, 6.6)
    ax.set_ylim(-0.5, 4.25)
    ax.axis("off")


GREY, RED, BLUE = "#4a5568", "#c53030", "#2b6cb0"
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
draw(axes[0], expert, f"Expert DAG ({len(e_exp)} edges)",
     lambda e: GREY if frozenset(e) in sk_hc else RED)
draw(axes[1], learned, f"BIC hill-climbing DAG ({len(e_hc)} edges)",
     lambda e: BLUE if frozenset(e) not in sk_exp else (GREY if e in e_exp else "#b7791f"))
fig.text(0.5, 0.02, "grey: in both graphs    red: expert only    blue: learned only    "
                    "amber: same edge, reversed direction", ha="center", fontsize=9)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "dag_comparison.png", dpi=180)

fig, ax = plt.subplots(figsize=(6.5, 3.2))
order = scores.sort_values("test_ll_per_row")
ax.errorbar(order["test_ll_per_row"], order["structure"], xerr=1.96 * order["test_ll_se"],
            fmt="o", color="#2b6cb0", capsize=4)
for x, y in zip(order["test_ll_per_row"], order["structure"]):
    ax.annotate(f"{x:.3f}", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)
ax.margins(y=0.25)
ax.grid(axis="x", alpha=0.3)
ax.set_xlabel("test joint log-likelihood per record (nats, 95% CI)")
fig.tight_layout()
fig.savefig(OUT / "test_loglik.png", dpi=180)
print("Saved outputs to", OUT)
