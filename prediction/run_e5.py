"""E5: predict the listing price tier on the test split with exact VE and compare models.

python -m prediction.run_e5
Writes metrics, paired gains, calibration tables and figures.
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from common import MODELS, ROOT, SPLITS, TARGET, load_dags, load_split, sha256
from parameter_learning import DiscreteBN, fit_em
from prediction.core import (EVIDENCE_SETS, fit_logistic, labels, one_hot, per_row, predict_bn,
                             reliability, scores)
from prediction.plots import calibration_figure, metric_figure
from structure.bn import dag_from_edges

OUTPUT_DIR = ROOT / "prediction/outputs"
PART_C = ROOT / "parameter_learning"
FITTED = ("expert", "hill_climb_bic")
BASELINE_GRAPHS = ("chow_liu", "naive_bayes", "empty")
LOGISTIC_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
REFERENCES = ("expert", "hill_climb_bic")


def fit_baselines(artifact, cfg, train, val):
    """EM fits of the baseline graphs with Part C's settings and selection rule."""
    fits, selection = {}, {}
    for name in BASELINE_GRAPHS:
        dag = dag_from_edges(artifact["nodes"], artifact["edges"][name])
        val_ll = {}
        for seed in cfg["seeds"]:
            result = fit_em(dag, train, artifact["cardinality"], cfg["alpha"], seed=seed,
                            max_iter=cfg["max_iter"], tol=cfg["tol"])
            if not result.converged:
                raise RuntimeError(f"EM did not converge for {name} seed {seed}")
            fits[name, seed] = result.model
            val_ll[seed] = float(result.model.log_likelihood(val).mean())
        best = max(val_ll, key=val_ll.get)
        selection[name] = {"seed": best, "val_ll_per_row": val_ll}
    return fits, selection


def load_fitted(cfg):
    selected = json.loads((PART_C / "outputs/selection.json").read_text())
    models = {(name, seed): DiscreteBN.load(MODELS / f"{name}_seed{seed}.json")
              for name in FITTED for seed in cfg["seeds"]}
    return models, {name: {"seed": selected[name]["seed"]} for name in FITTED}


def paired_gains(rows, evidence):
    """Per-row gains over each reference on the same test rows. Positive means better."""
    out = []
    for reference in REFERENCES:
        base = rows[reference]
        n = len(base)
        for model, frame in rows.items():
            if model == reference:
                continue
            acc = frame.correct - base.correct
            nll = base.nll - frame.nll
            out.append({"evidence": evidence, "reference": reference, "model": model,
                        "accuracy_gain": acc.mean(), "accuracy_gain_se": acc.std(ddof=1) / np.sqrt(n),
                        "log_loss_gain": nll.mean(), "log_loss_gain_se": nll.std(ddof=1) / np.sqrt(n)})
    return out


def run(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((PART_C / "config.json").read_text())
    artifact = load_dags()
    card = artifact["cardinality"]
    train, val, test = (load_split(s) for s in SPLITS)
    y = labels(test)

    models, selection = load_fitted(cfg)
    baseline_models, baseline_selection = fit_baselines(artifact, cfg, train, val)
    models.update(baseline_models)
    selection.update(baseline_selection)

    runs, paired, calibration, logistic = [], [], [], {}
    for evidence, features in EVIDENCE_SETS.items():
        selected_rows = {}
        for (name, seed), model in models.items():
            probs = predict_bn(model, test, features)
            is_selected = seed == selection[name]["seed"]
            runs.append({"evidence": evidence, "model": name, "seed": seed, "selected": is_selected,
                         **scores(probs, y)})
            if is_selected:
                selected_rows[name] = per_row(probs, y)
                calibration.append(reliability(probs, y).assign(evidence=evidence, model=name))

        fit, c, losses = fit_logistic(train, val, features, card, LOGISTIC_GRID)
        logistic[evidence] = {"C": c, "val_log_loss": {str(k): v for k, v in losses.items()}}
        probs = fit.predict_proba(one_hot(test, features, card))
        runs.append({"evidence": evidence, "model": "logistic", "seed": pd.NA, "selected": True,
                     **scores(probs, y)})
        selected_rows["logistic"] = per_row(probs, y)
        calibration.append(reliability(probs, y).assign(evidence=evidence, model="logistic"))
        paired.extend(paired_gains(selected_rows, evidence))

    runs = pd.DataFrame(runs).astype({"seed": "Int64"})
    runs.to_csv(output_dir / "runs.csv", index=False, lineterminator="\n")
    summary = summarize(runs)
    summary.to_csv(output_dir / "summary.csv", index=False, lineterminator="\n")
    pd.DataFrame(paired).to_csv(output_dir / "paired.csv", index=False, lineterminator="\n")
    calibration = pd.concat(calibration)[["evidence", "model", "bin", "lower", "upper", "count",
                                          "mean_confidence", "accuracy"]]
    calibration.to_csv(output_dir / "calibration.csv", index=False, lineterminator="\n")
    write_settings(output_dir, cfg, selection, logistic, len(test))
    metric_figure(summary, output_dir / "e5_metrics.png")
    calibration_figure(calibration, output_dir / "e5_calibration.png")


def summarize(runs):
    metrics = ["accuracy", "log_loss", "tier_mae", "ece"]
    chosen = runs[runs.selected].set_index(["evidence", "model"])[metrics]
    spread = runs.groupby(["evidence", "model"]).agg(
        n_fits=("accuracy", "size"), accuracy_seed_sd=("accuracy", "std"),
        log_loss_seed_sd=("log_loss", "std"))
    return chosen.join(spread).reset_index()


def write_settings(output_dir, cfg, selection, logistic, n_test):
    hashes = {p.name: sha256(p) for p in sorted(MODELS.glob("*_seed*.json"))
              if p.name.split("_seed")[0] in FITTED}
    settings = {
        "target": TARGET, "test_rows": n_test, "evidence_sets": {k: list(v) for k, v in EVIDENCE_SETS.items()},
        "paired_references": list(REFERENCES), "em": {k: cfg[k] for k in ("alpha", "seeds", "max_iter", "tol")},
        "selection": selection, "logistic": {"grid": list(LOGISTIC_GRID), **logistic},
        "calibration_bins": 10, "part_c_model_sha256": hashes,
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "matplotlib")},
    }
    (output_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    run(parser.parse_args().output_dir)
