"""E1: expert DAG vs BIC hill climbing, with Chow-Liu, naive Bayes and empty-graph baselines.

python -m structure.run_e1
Every structure uses the same variables, bins, complete-case rows and smoothing.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from common import SPLITS, load_card, load_split
from structure import bn

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"

POS = {
    "body_type": (0, 3), "brand_segment": (1.6, 3.3), "fuel_type": (3.3, 3.75), "vehicle_age": (5, 3),
    "engine_power_hp": (2.4, 1.9), "odometer_km": (3.9, 1.6), "number_of_previous_owners": (6, 1.9),
    "urban_rural": (0, 0.9), "accident_history": (4.8, 0.8), "service_history_quality": (5.8, 0.4),
    "listing_price": (1.2, 0), "overall_condition_score": (3.9, 0),
}
LABEL = {v: v.replace("number_of_", "").replace("_", "\n", 1).replace("_", " ") for v in POS}
GREY, RED, BLUE, AMBER = "#4a5568", "#c53030", "#2b6cb0", "#b7791f"


def load_data(nodes, out):
    """Complete-case rows for every split, the same rule for every structure."""
    data, counts = {}, []
    for split in SPLITS:
        frame = load_split(split)
        data[split] = frame[nodes].dropna().astype(int)
        counts.append({"split": split, "rows": len(frame), "complete_rows": len(data[split]),
                       "retained_pct": round(100 * len(data[split]) / len(frame), 2)})
    counts = pd.DataFrame(counts)
    counts.to_csv(out / "retained_counts.csv", index=False)
    print(counts.to_string(index=False), "\n")
    return data


def search_structures(score, nodes, card, train, cfg, out):
    """Hill climbing from the empty graph, the expert DAG and random starts, plus the baselines."""
    max_pa = cfg["max_parents"]
    expert = bn.dag_from_edges(nodes, cfg["expert_dag"])
    assert max(map(len, expert.values())) <= max_pa

    starts = {"empty": None, "expert": expert}
    for seed in cfg["hc_restart_seeds"]:
        starts[f"random_seed{seed}"] = bn.random_dag(nodes, 15, max_pa, np.random.default_rng(seed))

    runs, rows = {}, []
    for name, start in starts.items():
        dag, trace = bn.hill_climb(score, nodes, start, max_pa)
        runs[name] = dag
        rows.append({"start": name, "moves": len(trace) - 1, "edges": len(bn.edges_of(dag)),
                     "train_bic": trace[-1]})
    restarts = pd.DataFrame(rows)
    best_start = restarts.loc[restarts["train_bic"].idxmax(), "start"]
    restarts["shd_to_best"] = [bn.shd(runs[s], runs[best_start])[0] for s in restarts["start"]]
    restarts.to_csv(out / "hc_restarts.csv", index=False)
    print(restarts.to_string(index=False), "\n")

    dags = {
        "expert": expert,
        "hill_climb_bic": runs[best_start],
        "chow_liu": bn.chow_liu(train, card, nodes, cfg["target"]),
        "naive_bayes": bn.naive_bayes(nodes, cfg["target"]),
        "empty": bn.dag_from_edges(nodes, []),
    }
    return dags, best_start


def evaluate(dags, data, card, score, alpha, out):
    """Smoothed CPTs on train, then per-row joint log-likelihood and paired gains over the expert DAG."""
    rows, test_ll = [], {}
    for name, dag in dags.items():
        cpts = bn.fit_cpts(dag, data["train"], card, alpha)
        row = {"structure": name, "edges": len(bn.edges_of(dag)),
               "max_in_degree": max(map(len, dag.values())),
               "params": bn.n_params(dag, card), "train_bic": score.total(dag)}
        for split in SPLITS:
            ll = bn.log_likelihood(dag, cpts, data[split], card)
            row[f"{split}_ll_per_row"] = ll.mean()
            if split == "test":
                test_ll[name] = ll
                row["test_ll_total"] = ll.sum()
                row["test_ll_se"] = ll.std(ddof=1) / np.sqrt(len(ll))
        rows.append(row)
    scores = pd.DataFrame(rows)
    gains = {name: ll - test_ll["expert"] for name, ll in test_ll.items()}
    scores["test_gain_vs_expert"] = [gains[s].mean() for s in scores["structure"]]
    scores["test_gain_se"] = [gains[s].std(ddof=1) / np.sqrt(len(gains[s])) for s in scores["structure"]]
    scores.to_csv(out / "e1_scores.csv", index=False)
    print(scores.round(4).to_string(index=False), "\n")
    return scores


def compare(expert, learned):
    """Shared, reversed and unique edges between the expert and learned DAGs."""
    dist, shared, rev = bn.shd(expert, learned)
    e_exp, e_hc = set(bn.edges_of(expert)), set(bn.edges_of(learned))
    sk_exp, sk_hc = {frozenset(e) for e in e_exp}, {frozenset(e) for e in e_hc}
    return {
        "shd": dist, "shared_skeleton_edges": shared, "reversed": rev,
        "same_direction": sorted(e_exp & e_hc),
        "reversed_in_learned": sorted(e for e in e_hc if e[::-1] in e_exp),
        "only_expert": sorted(e for e in e_exp if frozenset(e) not in sk_hc),
        "only_learned": sorted(e for e in e_hc if frozenset(e) not in sk_exp),
    }


def ci_tests(dags, train, card, out):
    for name in ("expert", "hill_climb_bic"):
        ci = bn.local_markov_tests(dags[name], train, card)
        ci.to_csv(out / f"ci_tests_{name}.csv", index=False)
        print(f"{name}: {len(ci)} implied independencies, "
              f"{(ci.p_value < 0.01).sum()} rejected at 0.01, max CMI {ci.cmi_nats.max():.4f} nats")
        print(ci.head(6).round(5).to_string(index=False), "\n")


def draw(ax, dag, title, colors):
    g = bn.to_nx(dag)
    nx.draw_networkx_nodes(g, POS, ax=ax, node_size=2500, node_color="#eef2f7", edgecolors=GREY)
    nx.draw_networkx_labels(g, POS, LABEL, ax=ax, font_size=7)
    nx.draw_networkx_edges(g, POS, ax=ax, node_size=2500, arrowsize=14, width=1.4,
                           edge_color=[colors(e) for e in g.edges],
                           connectionstyle="arc3,rad=0.08")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-0.6, 6.6)
    ax.set_ylim(-0.5, 4.25)
    ax.axis("off")


def dag_figure(expert, learned, path):
    e_exp, e_hc = set(bn.edges_of(expert)), set(bn.edges_of(learned))
    sk_exp, sk_hc = {frozenset(e) for e in e_exp}, {frozenset(e) for e in e_hc}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    draw(axes[0], expert, f"Expert DAG ({len(e_exp)} edges)",
         lambda e: GREY if frozenset(e) in sk_hc else RED)
    draw(axes[1], learned, f"BIC hill-climbing DAG ({len(e_hc)} edges)",
         lambda e: BLUE if frozenset(e) not in sk_exp else (GREY if e in e_exp else AMBER))
    fig.text(0.5, 0.02, "grey: in both graphs    red: expert only    blue: learned only    "
                        "amber: same edge, reversed direction", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def loglik_figure(scores, path):
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    order = scores.sort_values("test_ll_per_row")
    ax.errorbar(order["test_ll_per_row"], order["structure"], xerr=1.96 * order["test_ll_se"],
                fmt="o", color=BLUE, capsize=4)
    for x, y in zip(order["test_ll_per_row"], order["structure"]):
        ax.annotate(f"{x:.3f}", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)
    ax.margins(y=0.25)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlabel("test joint log-likelihood per record (nats, 95% CI)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(output_dir=OUTPUT_DIR):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((HERE / "config.json").read_text())
    card = load_card()
    nodes = list(card)

    data = load_data(nodes, out)
    score = bn.BIC(data["train"], card)
    dags, best_start = search_structures(score, nodes, card, data["train"], cfg, out)
    scores = evaluate(dags, data, card, score, cfg["alpha"], out)
    diff = compare(dags["expert"], dags["hill_climb_bic"])
    print(json.dumps(diff, indent=1), "\n")

    (out / "dags.json").write_text(json.dumps({
        "nodes": nodes, "cardinality": card, "max_parents": cfg["max_parents"], "alpha": cfg["alpha"],
        "missing_rule": cfg["missing_rule"], "hc_best_start": best_start,
        "edges": {name: bn.edges_of(dag) for name, dag in dags.items()},
        "expert_vs_hill_climb": diff,
    }, indent=2))

    ci_tests(dags, data["train"], card, out)
    dag_figure(dags["expert"], dags["hill_climb_bic"], out / "dag_comparison.png")
    loglik_figure(scores, out / "test_loglik.png")
    print("Saved outputs to", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    run(parser.parse_args().output_dir)
