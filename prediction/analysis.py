"""E5 sensitivity to the Dirichlet pseudocount and training size, and price-tier posteriors given one feature.

python -m prediction.analysis
Writes sensitivity.csv and feature_posteriors.csv.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from common import MODELS, PREPARED, ROOT, TARGET, load_dags, load_split
from inference import query_ve
from parameter_learning import DiscreteBN, fit_em
from prediction.core import EVIDENCE_SETS, FEATURES, labels, per_row, predict_bn
from structure.bn import dag_from_edges

OUTPUT_DIR = ROOT / "prediction/outputs"
PART_C = ROOT / "parameter_learning"
GRAPHS = ("expert", "hill_climb_bic")
PSEUDOCOUNTS = (0.1, 1.0, 10.0, 100.0)
TRAIN_FRACTIONS = (0.1, 0.25, 0.5, 1.0)
SUBSAMPLE_SEED = 42
EM_SEED = 0


def validation_scores(dag, train, val, card, alpha, cfg):
    """Fit by EM, then score validation accuracy and log-loss with all features."""
    model = fit_em(dag, train, card, alpha, seed=EM_SEED, max_iter=cfg["max_iter"], tol=cfg["tol"]).model
    rows = per_row(predict_bn(model, val, EVIDENCE_SETS["all_features"]), labels(val))
    return {"accuracy": rows.correct.mean(), "log_loss": rows.nll.mean()}


def sensitivity(cfg):
    artifact = load_dags()
    card = artifact["cardinality"]
    train, val = load_split("train"), load_split("val")
    out = []
    for name in GRAPHS:
        dag = dag_from_edges(artifact["nodes"], artifact["edges"][name])
        for alpha in PSEUDOCOUNTS:
            out.append({"graph": name, "setting": "pseudocount", "value": alpha, "train_rows": len(train),
                        **validation_scores(dag, train, val, card, alpha, cfg)})
        for fraction in TRAIN_FRACTIONS:
            subset = train.sample(frac=fraction, random_state=SUBSAMPLE_SEED)
            out.append({"graph": name, "setting": "train_fraction", "value": fraction, "train_rows": len(subset),
                        **validation_scores(dag, subset, val, card, cfg["alpha"], cfg)})
    return pd.DataFrame(out)


def feature_posteriors():
    """P(listing_price tier | one observed feature) under the selected learned DAG."""
    states = json.loads((PREPARED / "schema.json").read_text())["columns"]
    model = DiscreteBN.load(MODELS / "hill_climb_bic_selected.json")
    out = []
    for feature in FEATURES:
        for index, label in enumerate(states[feature]["states"]):
            probs = query_ve(model, TARGET, {feature: index})
            out.append({"feature": feature, "state": label, **{f"tier_{t}": p for t, p in enumerate(probs)}})
    return pd.DataFrame(out)


def run(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((PART_C / "config.json").read_text())
    sensitivity(cfg).to_csv(output_dir / "sensitivity.csv", index=False, lineterminator="\n")
    feature_posteriors().to_csv(output_dir / "feature_posteriors.csv", index=False, lineterminator="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    run(parser.parse_args().output_dir)
