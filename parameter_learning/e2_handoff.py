"""Runnable integration example for A; this is not the full E2 experiment.

Runs C's observed-condition EM on one shared MCAR mask, and compares it with
complete-case and training-mode MLE using the same validation evidence.
python -m parameter_learning.e2_handoff --rate 0.2 --seed 0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from parameter_learning import fit_em, fit_mle
from structure.bn import dag_from_edges

ROOT = Path(__file__).resolve().parents[1]
MASK_COLUMNS = ["service_history_quality", "overall_condition_score"]


def mcar_mask(frame, columns, rate, seed):
    if not np.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("rate must be finite and in [0, 1]")
    mask = pd.DataFrame(False, index=frame.index, columns=frame.columns)
    draws = np.random.default_rng(seed).random((len(frame), len(columns)))
    mask[columns] = (draws < rate) & frame[columns].notna().to_numpy()
    return frame.mask(mask), mask


def run(rate=0.2, seed=0, structure="expert", output=None):
    artifact = json.loads((ROOT / "structure/outputs/dags.json").read_text())
    card = artifact["cardinality"]
    dag = dag_from_edges(card, artifact["edges"][structure])
    train = pd.read_csv(ROOT / "dataset/prepared/train.csv", index_col="source_row", dtype="Int64")
    val = pd.read_csv(ROOT / "dataset/prepared/val.csv", index_col="source_row", dtype="Int64")
    masked, mask = mcar_mask(train, MASK_COLUMNS, rate, seed)
    if not masked.isna().equals(train.isna() | mask):
        raise RuntimeError("Mask changed originally missing cells")
    result = fit_em(dag, masked, card, alpha=1, seed=seed)
    models = {"em": result.model,
              "complete_case_mle": fit_mle(dag, masked, card, alpha=1, missing="complete_case")}
    # Modes use only the masked training data. Break ties by smallest state.
    modes = {v: int(masked[v].mode().iloc[0]) for v in masked if not masked[v].dropna().empty}
    if len(modes) != len(card):
        raise ValueError("Mode baseline is undefined for an entirely missing training column")
    models["mode_mle"] = fit_mle(dag, masked.fillna(modes), card, alpha=1)
    rows = []
    for name, model in models.items():
        row = {"method": name, "mask_seed": seed, "additional_mask_rate": rate, "structure": structure,
               "val_observed_ll_per_row": model.log_likelihood(val).mean()}
        for v in MASK_COLUMNS:
            hidden = mask[v].to_numpy()
            # For each query, every deliberately hidden cell stays hidden.
            # Never restore another masked column's truth while scoring.
            if hidden.any():
                p = model.posterior(masked.loc[hidden], v)
                truth = train.loc[hidden, v].to_numpy(dtype=int)
                row[f"{v}_hidden_accuracy"] = float(np.mean(p.argmax(axis=1) == truth))
        rows.append(row)
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    if output:
        out = Path(output)
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / f"handoff_{structure}_rate{rate}_seed{seed}.csv", index=False, lineterminator="\n")
        result.history.to_csv(out / f"handoff_history_{structure}_rate{rate}_seed{seed}.csv", index=False, lineterminator="\n")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate", type=float, default=.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--structure", choices=["expert", "hill_climb_bic"], default="expert")
    parser.add_argument("--output")
    args = parser.parse_args()
    run(**vars(args))
