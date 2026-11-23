"""Check that Part C's models match the schema and Part B's DAGs, and that VE matches pgmpy.

python -m inference.check_setup
Prints the reference posteriors for the two E4 evidence cases on both graphs.
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning, module="pgmpy")
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from common import MODELS, TARGET, load_card, load_dags, load_split
from inference import query_ve
from parameter_learning import DiscreteBN
from structure.bn import dag_from_edges, to_nx

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
    counts = load_split("val")[EVIDENCE_FIELDS].value_counts()
    edges = load_dags()["edges"]
    card = load_card()
    for name in ("expert", "hill_climb_bic"):
        model = DiscreteBN.load(MODELS / f"{name}_selected.json")
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
