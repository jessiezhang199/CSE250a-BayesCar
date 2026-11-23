import numpy as np
import pandas as pd
import pytest

from common import MODELS, TARGET, load_split
from inference import query_ve
from inference.check_setup import pgmpy_reference
from inference.sampling import gibbs_sampling, likelihood_weighting, rejection_sampling
from parameter_learning import DiscreteBN


@pytest.fixture
def collider():
    # Nonalphabetical parent order, unequal cardinalities, child stored first.
    return DiscreteBN(
        {"C": ("B", "A"), "A": (), "B": ()}, {"C": 2, "A": 3, "B": 2},
        {"C": [[0.95, 0.05], [0.55, 0.45], [0.30, 0.70],
               [0.70, 0.30], [0.40, 0.60], [0.10, 0.90]],
         "A": [[0.15, 0.55, 0.30]], "B": [[0.25, 0.75]]},
    )


@pytest.mark.parametrize("name", ["expert", "hill_climb_bic"])
def test_real_queries_against_pgmpy(name):
    model = DiscreteBN.load(MODELS / f"{name}_selected.json")
    oracle = pgmpy_reference(model)
    val = load_split("val")
    row = val.loc[val.overall_condition_score.isna()].iloc[0]
    row_evidence = row.drop(TARGET).dropna().to_dict()
    queries = [(TARGET, row_evidence), (TARGET, {}), ("vehicle_age", {TARGET: 4}),
               (TARGET, {"odometer_km": 2, "overall_condition_score": 0}),
               (TARGET, {"odometer_km": 0, "overall_condition_score": 0}),
               ("service_history_quality", {TARGET: 1})]
    for target, evidence in queries:
        actual = query_ve(model, target, evidence)
        expected = oracle.query([target], evidence=evidence, show_progress=False).values
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual.sum(), 1, atol=1e-12, rtol=0)
    masked = row.to_frame().T.copy()
    masked[TARGET] = pd.NA
    np.testing.assert_allclose(query_ve(model, TARGET, row_evidence),
                               model.posterior(masked, TARGET)[0], atol=1e-10, rtol=0)


def test_exact_on_collider(collider):
    # P(A | B=1,C=1) is proportional to P(A)P(C=1 | B=1,A).
    masses = np.array([0.15 * 0.30, 0.55 * 0.60, 0.30 * 0.90])
    np.testing.assert_allclose(query_ve(collider, "A", {"B": 1, "C": 1}), masses / masses.sum())
    np.testing.assert_allclose(query_ve(collider, "A", {}), collider.cpts["A"][0])
    np.testing.assert_array_equal(query_ve(collider, "A", {"A": np.int64(2)}), [0, 0, 1])


@pytest.mark.parametrize("method", [likelihood_weighting, rejection_sampling, gibbs_sampling])
def test_sampling_against_exact_and_reproducible(collider, method):
    evidence = {"C": 1}
    options = {"burn_in": 100} if method is gibbs_sampling else {}
    result = method(collider, "A", evidence, 12_000, seed=7, **options)
    repeat = method(collider, "A", evidence, 12_000, seed=7, **options)
    exact = query_ve(collider, "A", evidence)
    assert np.abs(result.posterior - exact).sum() / 2 < 0.025
    np.testing.assert_array_equal(result.posterior, repeat.posterior)
    if method is gibbs_sampling:
        assert len(result.trace) == 12_000
        fixed = method(collider, "C", evidence, 20, seed=5, burn_in=3)
        assert np.all(fixed.trace == 1)
    elif method is likelihood_weighting:
        assert 0 < result.weight_ess < 12_000
        assert method(collider, "A", {}, 10, seed=2).weight_ess == 10
    else:
        assert 0 < result.accepted < 12_000


def test_structural_zeros():
    model = DiscreteBN({"A": (), "B": ("A",)}, {"A": 2, "B": 2},
                       {"A": [[0.99, 0.01]], "B": [[1.0, 0.0], [0.0, 1.0]]})
    impossible = {"A": 0, "B": 1}
    with pytest.raises(ValueError, match="zero probability"):
        query_ve(model, "A", impossible)
    with pytest.raises(ValueError, match="zero probability"):
        gibbs_sampling(model, "A", impossible, 1, seed=0)
    np.testing.assert_array_equal(gibbs_sampling(model, "A", {"B": 1}, 10, seed=0).trace, 1)
    weighted = likelihood_weighting(model, "A", impossible, 100, seed=0)
    assert weighted.posterior is None and weighted.weight_ess == 0
    rejected = rejection_sampling(model, "A", {"B": 1}, 1, seed=0)
    assert rejected.posterior is None and rejected.accepted == 0


@pytest.mark.parametrize("target,evidence", [
    ("unknown", {}), ("A", {"foo": 1}), ("A", {"A": -1}), ("A", {"B": 2}),
    ("A", {"B": 0.5}), ("A", {"B": True}), ("A", {"C": pd.NA}),
])
def test_invalid_queries(collider, target, evidence):
    with pytest.raises(ValueError):
        query_ve(collider, target, evidence)
