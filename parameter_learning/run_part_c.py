"""Reproduce Part C using the fixed upstream data/DAGs; no structure search.

python -m parameter_learning.run_part_c
All labels are read only AFTER fitting and validation-based selection finish.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from parameter_learning import fit_mle, fit_em
from parameter_learning.metrics import normalized_mutual_information as nmi
from structure.bn import dag_from_edges, n_params

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def hash_file(p):
    with p.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def plot_detail(out):
    """Expose slow latent convergence after the large initialization jump."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    cfg = json.loads((HERE / "config.json").read_text())
    for seed in cfg["seeds"]:
        history = pd.read_csv(out / "histories" / f"latent_H3_seed{seed}.csv")
        tail = history[history.iteration >= 10]
        axes[0].plot(tail.iteration, tail.ll_per_row, label=f"seed {seed}")
        axes[1].semilogy(tail.iteration, tail.objective_gain_per_row.clip(lower=1e-15), label=f"seed {seed}")
    axes[0].set(title="Latent H: likelihood after iteration 10", xlabel="EM iteration", ylabel="observed LL / row (nats)")
    axes[1].axhline(cfg["tol"], color="black", ls="--", label="stopping tolerance")
    axes[1].set(title="Latent H: regularized objective gain", xlabel="EM iteration", ylabel="gain per row (log scale)")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out / "em_convergence_detail.png", dpi=160)
    plt.close(fig)


def run(output=None):
    out = Path(output) if output else HERE / "outputs"
    for folder in (out, out / "models", out / "histories"):
        folder.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((HERE / "config.json").read_text())
    artifact = json.loads((ROOT / "structure/outputs/dags.json").read_text())
    card = artifact["cardinality"]
    data = {s: pd.read_csv(ROOT / f"dataset/prepared/{s}.csv", index_col="source_row", dtype="Int64")
            for s in ("train", "val", "test")}
    dags = {name: dag_from_edges(card, artifact["edges"][name]) for name in ("expert", "hill_climb_bic")}
    spec = cfg["latent_graph"]
    latent_edges = [e for e in artifact["edges"][spec["base"]] if e not in spec["remove_edges"]] + spec["add_edges"]
    latent_card = {**card, "H": cfg["latent_cardinality"]}
    latent_dag = dag_from_edges(latent_card, latent_edges)
    if max(map(len, latent_dag.values())) > artifact["max_parents"]:
        raise ValueError("E3 violates shared parent cap")
    (out / "latent_dag.json").write_text(json.dumps({"parents": latent_dag, "cardinality": latent_card,
                                                   "specification": spec}, indent=2), newline="\n")
    fits, records, selected = {}, [], {}
    started = time.perf_counter()
    # Fit on train only. Test likelihoods are deliberately computed later.
    for name, dag in [*dags.items(), ("latent_H3", latent_dag)]:
        local_card = latent_card if name == "latent_H3" else card
        latent_nodes = ("H",) if name == "latent_H3" else ()
        if not latent_nodes:
            mle = fit_mle(dag, data["train"], card, cfg["alpha"], missing="complete_case")
            mle.save(out / "models" / f"{name}_complete_case_mle.json")
            fits[f"{name}_complete_case_mle"] = mle
        for seed in cfg["seeds"]:
            t = time.perf_counter()
            result = fit_em(dag, data["train"], local_card, cfg["alpha"], latent_nodes=latent_nodes,
                            seed=seed, max_iter=cfg["max_iter"], tol=cfg["tol"])
            elapsed = time.perf_counter() - t
            key = f"{name}_seed{seed}"
            fits[key] = result.model
            result.model.save(out / "models" / f"{key}.json")
            result.history.to_csv(out / "histories" / f"{key}.csv", index=False, lineterminator="\n")
            val_ll = result.model.log_likelihood(data["val"], latent_nodes=latent_nodes).mean()
            row = {"model": name, "seed": seed, "params": n_params(dag, local_card),
                   "iterations": result.n_iter, "converged": result.converged,
                   "train_ll_per_row": result.history.ll_per_row.iloc[-1],
                   "final_objective": result.history.objective.iloc[-1],
                   "val_ll_per_row": val_ll, "fit_seconds": elapsed}
            records.append(row)
            print(json.dumps(row), flush=True)
        candidates = [r for r in records if r["model"] == name]
        best = max(candidates, key=lambda r: r["val_ll_per_row"])
        selected[name] = {"seed": best["seed"], "converged": best["converged"],
                          "criterion": cfg["selection"]}
        chosen_model = fits[f"{name}_seed{best['seed']}"]
        chosen_model.save(out / "models" / f"{name}_selected.json")
    (out / "selection.json").write_text(json.dumps(selected, indent=2), newline="\n")
    # All seeds are reported; primary results use only the selected seed.
    for row in records:
        row["selected"] = row["seed"] == selected[row["model"]]["seed"]
        model = fits[f"{row['model']}_seed{row['seed']}"]
        row["test_ll_per_row"] = model.log_likelihood(data["test"], latent_nodes=("H",) if "H" in model.dag else ()).mean()
    pd.DataFrame(records).to_csv(out / "restarts.csv", index=False, lineterminator="\n")
    summary = []
    likelihoods = {}
    # Compare complete-case MLE and observed-data EM on EXACTLY the same rows
    # and observed cell patterns, integrating the natural missing values.
    for name in dags:
        for method, key in [("complete_case_mle", f"{name}_complete_case_mle"),
                            ("em", f"{name}_seed{selected[name]['seed']}")]:
            model = fits[key]
            row = {"model": name, "method": method, "params": n_params(model.dag, model.card),
                   "training_rows": len(data["train"].dropna()) if method == "complete_case_mle" else len(data["train"])}
            for split, frame in data.items():
                ll = model.log_likelihood(frame)
                row[f"{split}_ll_per_row"] = ll.mean()
                row[f"{split}_ll_se"] = ll.std(ddof=1) / np.sqrt(len(ll))
                likelihoods[(name, method, split)] = ll
            summary.append(row)
    key = f"latent_H3_seed{selected['latent_H3']['seed']}"
    model = fits[key]
    row = {"model": "latent_H3", "method": "em", "params": n_params(model.dag, model.card), "training_rows": len(data["train"])}
    for split, frame in data.items():
        ll = model.log_likelihood(frame, latent_nodes=("H",))
        row[f"{split}_ll_per_row"] = ll.mean()
        row[f"{split}_ll_se"] = ll.std(ddof=1) / np.sqrt(len(ll))
        likelihoods[("latent_H3", "em", split)] = ll
    summary.append(row)
    reference = likelihoods[("expert", "em", "test")]
    for row in summary:
        gain = likelihoods[(row["model"], row["method"], "test")] - reference
        row["test_gain_vs_expert_em"] = gain.mean()
        row["paired_gain_se"] = gain.std(ddof=1) / np.sqrt(len(gain))
    pd.DataFrame(summary).to_csv(out / "likelihood_comparison.csv", index=False, lineterminator="\n")
    # Evaluation-only labels enter here, after all model selection decisions.
    evaluations, contingencies, indicator_rows = [], [], []
    for split in ("val", "test"):
        frame = data[split]
        labels = pd.read_csv(HERE / f"labels/{split}.csv", index_col="source_row").reindex(frame.index)
        for variable in ("overall_condition_score", "vehicle_age", "listing_price"):
            available = labels.condition_grade.notna().to_numpy() & frame[variable].notna().to_numpy()
            indicator_rows.append({"split": split, "variable": variable, "paired_rows": int(available.sum()),
                                   "nmi_with_grade": nmi(labels.condition_grade[available].to_numpy(dtype=str),
                                                         frame.loc[available, variable].to_numpy(dtype=int))})
    pd.DataFrame(indicator_rows).to_csv(out / "observed_indicator_nmi.csv", index=False, lineterminator="\n")
    for seed in cfg["seeds"]:
        model = fits[f"latent_H3_seed{seed}"]
        for split in ("val", "test"):
            frame = data[split]
            labels = pd.read_csv(HERE / f"labels/{split}.csv", index_col="source_row").reindex(frame.index)
            if not labels.index.is_unique or len(labels) != len(frame):
                raise ValueError("Invalid evaluation label alignment")
            available = labels.condition_grade.notna().to_numpy()
            grade = labels.condition_grade[available].to_numpy(dtype=str)
            for evidence, hide in [("all_observed", []), ("without_condition_score", ["overall_condition_score"]),
                                   ("without_score_or_price", ["overall_condition_score", "listing_price"])]:
                evidence_frame = frame.copy()
                for col in hide:
                    evidence_frame[col] = pd.NA
                p = model.posterior(evidence_frame, "H", latent_nodes=("H",))
                pred = p.argmax(axis=1)
                score = nmi(grade, pred[available])
                rng = np.random.default_rng(2026)
                null = np.array([nmi(rng.permutation(grade), pred[available]) for _ in range(100)])
                row = {"seed": seed, "selected": seed == selected["latent_H3"]["seed"], "split": split,
                       "evidence": evidence, "labeled_rows": int(available.sum()), "nmi": score,
                       "permutation_nmi_mean": null.mean(), "permutation_nmi_max": null.max(),
                       "mean_posterior_entropy": float(-(p * np.log(np.clip(p, 1e-300, 1))).sum(axis=1).mean())}
                row.update({f"hard_H{h}_fraction": float(np.mean(pred == h)) for h in range(3)})
                row.update({f"soft_H{h}_fraction": float(p[:, h].mean()) for h in range(3)})
                evaluations.append(row)
                if row["selected"] and split == "test":
                    for h in range(3):
                        for g in sorted(np.unique(grade)):
                            contingencies.append({"evidence": evidence, "H": h, "grade": g,
                                                  "count": int(np.sum((pred[available] == h) & (grade == g)))})
                    prediction = pd.DataFrame({"H_map": pred, **{f"p_H{h}": p[:, h] for h in range(3)}}, index=frame.index)
                    prediction.to_csv(out / f"test_H_{evidence}.csv", lineterminator="\n")
    ev = pd.DataFrame(evaluations)
    ev.to_csv(out / "e3_nmi.csv", index=False, lineterminator="\n")
    ct = pd.DataFrame(contingencies)
    ct.to_csv(out / "e3_contingency.csv", index=False, lineterminator="\n")
    group = ev[ev.split == "test"].groupby("evidence").nmi.agg(["mean", "std", "min", "max"])
    group.to_csv(out / "e3_nmi_across_seeds.csv", lineterminator="\n")
    # Figures: ML and regularized objective are both shown, not conflated.
    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    for i, name in enumerate([*dags, "latent_H3"]):
        for seed in cfg["seeds"]:
            history = pd.read_csv(out / "histories" / f"{name}_seed{seed}.csv")
            axes[i, 0].plot(history.iteration, history.ll_per_row, label=f"seed {seed}")
            axes[i, 1].plot(history.iteration, history.objective / len(data["train"]), label=f"seed {seed}")
        for j, metric in enumerate(["Observed-data LL / row", "Regularized objective / row"]):
            axes[i, j].set(title=f"{name}: {metric}", xlabel="EM iteration", ylabel="nats")
            axes[i, j].grid(alpha=0.25)
            axes[i, j].legend()
    fig.tight_layout()
    fig.savefig(out / "em_convergence.png", dpi=160)
    plt.close(fig)
    plot_detail(out)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for seed in cfg["seeds"]:
        rows = ev[(ev.seed == seed) & (ev.split == "test")]
        axes[0].plot(range(len(rows)), rows.nmi, "o-", label=f"seed {seed}")
    axes[0].set(xticks=range(3), xticklabels=["All observed", "Hide score", "Hide score + price"],
                ylabel="NMI with held-out condition_grade", title="E3: sensitivity to posterior evidence")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    matrix = ct[ct.evidence == "all_observed"].pivot(index="H", columns="grade", values="count")
    normalized = matrix.div(matrix.sum(axis=1), axis=0).fillna(0)
    image = axes[1].imshow(normalized, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    axes[1].set(xticks=range(len(matrix.columns)), xticklabels=matrix.columns, yticks=range(3),
                yticklabels=[f"H={h}" for h in range(3)], xlabel="condition_grade (evaluation only)",
                title="Selected seed: grade fractions within H")
    for i in range(3):
        for j in range(len(matrix.columns)):
            axes[1].text(j, i, f"{normalized.iloc[i,j]:.0%}", ha="center", va="center",
                         color="white" if normalized.iloc[i,j] > .5 else "black")
    fig.colorbar(image, ax=axes[1], shrink=.8)
    fig.tight_layout()
    fig.savefig(out / "e3_association.png", dpi=160)
    plt.close(fig)
    inputs = [ROOT / f"dataset/prepared/{s}.csv" for s in data] + [ROOT / "dataset/prepared/schema.json",
              ROOT / "structure/outputs/dags.json", HERE / "config.json", HERE / "labels/provenance.json"]
    inputs += [HERE / f"labels/{s}.csv" for s in data]
    manifest = {"upstream_commit": "d0446966a9ed62b77807999dcd80e76d0df5edcb", "config": cfg,
                "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                "pandas": pd.__version__, "elapsed_seconds": time.perf_counter() - started,
                "rows": {s: len(f) for s, f in data.items()},
                "input_sha256": {str(p.relative_to(ROOT)): hash_file(p) for p in inputs}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), newline="\n")
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
    print(group.to_string(), flush=True)
    print(f"Outputs: {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args()
    run(args.output)
