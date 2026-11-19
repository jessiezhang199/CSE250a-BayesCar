"""Check the A/B/C handoff and print reference values for the two E4 queries."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from inference import query_ve
from parameter_learning import DiscreteBN
from structure.bn import dag_from_edges, load_schema, to_nx

ROOT = Path(__file__).resolve().parents[1]
TARGET = "listing_price"
EVIDENCE_FIELDS = ["odometer_km", "overall_condition_score"]
CASES = {"common": (2, 0), "rare": (0, 0)}


def pgmpy_reference(model):
    network = DiscreteBayesianNetwork(to_nx(model.dag))
    for node, parents in model.dag.items():
        # Part C stores one row per parent configuration. pgmpy needs one column.
        network.add_cpds(TabularCPD(
            node, model.card[node], model.cpts[node].T,
            evidence=list(parents), evidence_card=[model.card[p] for p in parents],
        ))
    network.check_model()
    return VariableElimination(network)


def main():
    val = pd.read_csv(ROOT / "dataset/prepared/val.csv", index_col="source_row", dtype="Int64")
    counts = val[EVIDENCE_FIELDS].value_counts()
    edges = json.loads((ROOT / "structure/outputs/dags.json").read_text())["edges"]
    card = load_schema()
    for name in ("expert", "hill_climb_bic"):
        model = DiscreteBN.load(ROOT / f"parameter_learning/outputs/models/{name}_selected.json")
        if model.card != card or model.dag != dag_from_edges(card, edges[name]):
            raise ValueError(f"{name} disagrees with the prepared schema or Part B DAG")
        reference = pgmpy_reference(model)
        joint = reference.query(EVIDENCE_FIELDS, show_progress=False)
        for label, states in CASES.items():
            evidence = dict(zip(EVIDENCE_FIELDS, states))
            exact = reference.query([TARGET], evidence=evidence, show_progress=False).values
            np.testing.assert_allclose(query_ve(model, TARGET, evidence), exact, atol=1e-10, rtol=0)
            row = pd.DataFrame([evidence], columns=list(model.dag), dtype="Int64")
            enumerated = model.posterior(row, TARGET)[0]
            np.testing.assert_allclose(exact, enumerated, atol=1e-10, rtol=0)
            print(f"{name} {label}: val={counts[states]}/{counts.sum()}, "
                  f"P(e)={joint.get_value(**evidence):.6f}, P(price|e)={exact.round(6)}")


if __name__ == "__main__":
    main()
