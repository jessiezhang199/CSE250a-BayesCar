"""E2: complete-case MLE, mode imputation + MLE and EM under the same MCAR masks.

python -m complete-case_MLE_and_EM.run_e2
Each rate and seed draws one training mask that all three methods share. Every model
is scored on the same raw test evidence and on one fixed 20% test mask for recovery.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_dags, load_split
from parameter_learning import fit_em, fit_mle
from parameter_learning.e2_handoff import MASK_COLUMNS, mcar_mask
from structure.bn import dag_from_edges

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
ALPHA = 1.0
RATES = (0.0, 0.2, 0.4)
SEEDS = (0, 1, 2)
METHODS = ("Complete-case MLE", "Mode + MLE", "EM")
TEST_MASK_RATE, TEST_MASK_SEED = 0.2, 2026
EM_MAX_ITER, EM_TOL = 500, 1e-7


@dataclass
class Setup:
    dag: dict
    card: dict
    train: pd.DataFrame
    test: pd.DataFrame
    reference: object
    masked_test: pd.DataFrame
    hidden_test: pd.DataFrame


def prepare(structure="expert"):
    artifact = load_dags()
    card = artifact["cardinality"]
    dag = dag_from_edges(card, artifact["edges"][structure])
    train, test = load_split("train"), load_split("test")
    reference = fit_mle(dag, train, card, alpha=ALPHA, missing="complete_case")
    masked_test, hidden_test = mcar_mask(test, MASK_COLUMNS, TEST_MASK_RATE, TEST_MASK_SEED)
    return Setup(dag, card, train, test, reference, masked_test, hidden_test)


def mean_cpt_kl(reference, model):
    """KL(reference || model) averaged over parent configurations, then over nodes."""
    node_kl = [np.sum(p * (np.log(p) - np.log(model.cpts[node])), axis=1).mean()
               for node, p in reference.cpts.items()]
    return float(np.mean(node_kl))


def run_comparison(setup, rate, seed):
    """Fit all three methods on one shared training mask. Returns (score rows, EM diagnostics)."""
    s = setup
    masked_train, hidden_train = mcar_mask(s.train, MASK_COLUMNS, rate, seed)
    assert masked_train.isna().equals(s.train.isna() | hidden_train)
    assert not (hidden_train & s.train.isna()).any().any()

    modes = masked_train.mode(dropna=True).iloc[0]
    if modes.isna().any():
        raise ValueError("Mode imputation requires an observed value in every column")

    complete_case = fit_mle(s.dag, masked_train, s.card, alpha=ALPHA, missing="complete_case")
    mode_mle = fit_mle(s.dag, masked_train.fillna(modes), s.card, alpha=ALPHA)
    em = fit_em(s.dag, masked_train, s.card, alpha=ALPHA, init_cpts=mode_mle.cpts,
                seed=seed, max_iter=EM_MAX_ITER, tol=EM_TOL)
    models = (complete_case, mode_mle, em.model)
    rows_used = (len(masked_train.dropna()), len(masked_train), len(masked_train))

    scores = []
    for method, model, used in zip(METHODS, models, rows_used):
        row = {"mask_rate": rate, "mask_seed": seed, "method": method, "train_rows_used": used,
               "test_ll_per_row": model.log_likelihood(s.test).mean(),
               "mean_cpt_kl": mean_cpt_kl(s.reference, model)}
        for col in MASK_COLUMNS:
            hidden = s.hidden_test[col]
            truth = s.test.loc[hidden, col].to_numpy(dtype=int)
            predicted = model.posterior(s.masked_test.loc[hidden], col).argmax(axis=1)
            row[f"{col}_accuracy"] = np.mean(predicted == truth)
        scores.append(row)

    diagnostic = {"mask_rate": rate, "mask_seed": seed,
                  "em_converged": em.converged, "em_iterations": em.n_iter}
    for col in MASK_COLUMNS:
        diagnostic[f"{col}_added_fraction"] = hidden_train[col].sum() / s.train[col].count()
    return scores, diagnostic


def summarize(results):
    """Mean and sd across masking seeds, and how often EM beats complete-case MLE on the same mask."""
    metrics = ["train_rows_used", "test_ll_per_row", "mean_cpt_kl",
               *(f"{col}_accuracy" for col in MASK_COLUMNS)]
    summary = results.groupby(["mask_rate", "method"], sort=False)[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    paired = results.pivot(index=["mask_rate", "mask_seed"], columns="method",
                           values=["test_ll_per_row", "mean_cpt_kl"])
    wins = pd.DataFrame({
        "em_higher_test_ll": paired["test_ll_per_row"]["EM"] > paired["test_ll_per_row"]["Complete-case MLE"],
        "em_lower_cpt_kl": paired["mean_cpt_kl"]["EM"] < paired["mean_cpt_kl"]["Complete-case MLE"],
    }).groupby(level="mask_rate").sum()
    return summary.reset_index(), wins.reset_index()


def run(output_dir=OUTPUT_DIR):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    setup = prepare()
    runs, diagnostics = [], []
    for rate in RATES:
        for seed in SEEDS:
            scores, diagnostic = run_comparison(setup, rate, seed)
            runs.extend(scores)
            diagnostics.append(diagnostic)
            print(f"Rate {rate:.0%}, seed {seed}: {diagnostic['em_iterations']} EM iterations", flush=True)

    results = pd.DataFrame(runs)
    summary, wins = summarize(results)
    results.to_csv(out / "runs.csv", index=False, lineterminator="\n")
    pd.DataFrame(diagnostics).to_csv(out / "em_diagnostics.csv", index=False, lineterminator="\n")
    summary.to_csv(out / "summary.csv", index=False, lineterminator="\n")
    wins.to_csv(out / "paired_wins.csv", index=False, lineterminator="\n")
    table = summary.pivot(index="mask_rate", columns="method", values="test_ll_per_row_mean")[list(METHODS)]
    print("\nTest log-likelihood per row, mean over masking seeds")
    print(table.round(5).to_string())
    print(f"\nSaved outputs to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    run(parser.parse_args().output_dir)
