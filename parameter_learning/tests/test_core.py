"""Independent oracles for indexing, marginalization, EM and handoff behavior."""
from itertools import product

import numpy as np
import pandas as pd
import pytest

from parameter_learning import DiscreteBN, fit_em, fit_mle
from structure.bn import fit_cpts, log_likelihood
from parameter_learning.metrics import normalized_mutual_information as nmi
from parameter_learning.e2_handoff import mcar_mask


def tiny():
    return DiscreteBN({"A": (), "B": ("A",)}, {"A": 2, "B": 2},
                      {"A": [[0.6, 0.4]], "B": [[0.9, 0.1], [0.2, 0.8]]})


def brute(model, row):
    assignments, weights = [], []
    nodes = list(model.dag)
    for values in product(*(range(model.card[v]) for v in nodes)):
        x = dict(zip(nodes, values))
        if any(pd.notna(row.get(v, pd.NA)) and x[v] != row[v] for v in nodes):
            continue
        p = 1.0
        for v, parents in model.dag.items():
            j = 0
            for parent in parents:
                j = j * model.card[parent] + x[parent]
            p *= model.cpts[v][j, x[v]]
        assignments.append(x)
        weights.append(p)
    return assignments, np.array(weights)


def test_exact_marginalization_and_posterior():
    model = tiny()
    data = pd.DataFrame({"A": [0, None, 1, None, 0], "B": [1, 1, None, None, 1]})
    ll = model.log_likelihood(data)
    posterior = model.posterior(data, "A")
    for i, (_, row) in enumerate(data.iterrows()):
        assignments, weights = brute(model, row)
        assert ll[i] == pytest.approx(np.log(weights.sum()))
        expected = np.bincount([x["A"] for x in assignments], weights=weights, minlength=2) / weights.sum()
        np.testing.assert_allclose(posterior[i], expected, atol=1e-14)
    assert np.exp(ll[3]) == pytest.approx(1.0)


def test_one_em_step_matches_independent_expected_counts():
    initial = tiny()
    data = pd.DataFrame({"A": [0, None, 1, None], "B": [1, 1, None, 0]})
    counts = {v: np.zeros_like(t) for v, t in initial.cpts.items()}
    for _, row in data.iterrows():
        assignments, weights = brute(initial, row)
        for x, w in zip(assignments, weights / weights.sum()):
            counts["A"][0, x["A"]] += w
            counts["B"][x["A"], x["B"]] += w
    result = fit_em(initial.dag, data, initial.card, alpha=0.7, init_cpts=initial.cpts, max_iter=1)
    for v, n in counts.items():
        expected = (n + 0.7) / (n.sum(axis=1, keepdims=True) + 1.4)
        np.testing.assert_allclose(result.cpts[v], expected, atol=1e-14)


def test_mle_compatibility_and_complete_data_em():
    rng = np.random.default_rng(42)
    card = {"A": 2, "B": 3, "C": 2}
    # Deliberately unsorted parents: indexing must follow the supplied order.
    dag = {"A": (), "B": (), "C": ("B", "A")}
    data = pd.DataFrame({v: rng.integers(r, size=800) for v, r in card.items()})
    mle = fit_mle(dag, data, card)
    old = fit_cpts(dag, data, card)
    em = fit_em(dag, data, card, max_iter=3)
    assert em.converged
    for v in dag:
        np.testing.assert_allclose(mle.cpts[v], old[v])
        np.testing.assert_allclose(em.cpts[v], old[v])
    np.testing.assert_allclose(mle.log_likelihood(data), log_likelihood(dag, old, data, card))


@pytest.mark.parametrize("alpha", [0, 1])
def test_em_monotonic_reproducible_and_na_forms(alpha):
    dag, card = tiny().dag, tiny().card
    data = pd.DataFrame({"A": [0, None, 1, 0, None] * 30, "B": [0, 1, None, 1, 0] * 30}, dtype="Int64")
    a = fit_em(dag, data, card, alpha=alpha)
    b = fit_em(dag, data.astype(float), card, alpha=alpha)
    assert a.converged
    assert (a.history.objective.diff().dropna() >= -1e-8).all()
    for v in dag:
        np.testing.assert_allclose(a.cpts[v], b.cpts[v])
    if alpha == 0:
        assert (a.history.observed_ll.diff().dropna() >= -1e-8).all()


def test_latent_column_absence_and_labels_ignored():
    model = tiny()
    data = pd.DataFrame({"B": [0, 1, 1, 0] * 10, "condition_grade": ["D", "B", "A", "C"] * 10})
    with pytest.raises(ValueError, match="Missing model columns"):
        fit_em(model.dag, data, model.card)
    a = fit_em(model.dag, data, model.card, latent_nodes=["A"])
    b = fit_em(model.dag, data.drop(columns="condition_grade"), model.card, latent_nodes=["A"])
    for v in model.dag:
        np.testing.assert_allclose(a.cpts[v], b.cpts[v])
    p = a.model.posterior(data, "A", latent_nodes=["A"])
    np.testing.assert_allclose(p.sum(axis=1), 1)


def test_serialization_and_guards(tmp_path):
    model = tiny()
    path = tmp_path / "cpt.json"
    model.save(path)
    restored = DiscreteBN.load(path)
    assert restored.dag == model.dag
    for v in model.dag:
        np.testing.assert_allclose(restored.cpts[v], model.cpts[v])
    for invalid in [-1, 0.2, 2, np.inf]:
        with pytest.raises(ValueError, match="integer states"):
            model.log_likelihood(pd.DataFrame({"A": [invalid], "B": [0]}))
    with pytest.raises(ValueError, match="budget"):
        model.log_likelihood(pd.DataFrame({"A": [None], "B": [None]}), max_completions=3)
    with pytest.raises(ValueError, match="cycle"):
        fit_mle({"A": ("B",), "B": ("A",)}, pd.DataFrame(), model.card)
    with pytest.raises(ValueError, match="nonnegative"):
        fit_em(model.dag, pd.DataFrame(), model.card, alpha=-1)


def test_complete_case_and_unseen_parents():
    model = tiny()
    data = pd.DataFrame({"A": [0, 0, None], "B": [0, 1, 1]})
    with pytest.raises(ValueError, match="requires"):
        fit_mle(model.dag, data, model.card)
    fit = fit_mle(model.dag, data, model.card, alpha=0, missing="complete_case")
    np.testing.assert_allclose(fit.cpts["B"][1], [0.5, 0.5])


def test_latent_recovery_up_to_label_permutation():
    # Three independent noisy indicators identify two synthetic components.
    rng = np.random.default_rng(71)
    h = rng.integers(2, size=5000)
    data = pd.DataFrame({v: np.where(rng.random(len(h)) < 0.94, h, 1-h) for v in ("X", "Y", "Z")})
    dag = {"H": (), "X": ("H",), "Y": ("H",), "Z": ("H",)}
    result = fit_em(dag, data, dict.fromkeys(dag, 2), latent_nodes=["H"], seed=2)
    pred = result.model.posterior(data, "H", latent_nodes=["H"]).argmax(axis=1)
    assert max(np.mean(pred == h), np.mean(pred != h)) > 0.97


def test_zero_probability_evidence_fails_clearly():
    model = DiscreteBN({"A": ()}, {"A": 2}, {"A": [[1, 0]]})
    with pytest.raises(ValueError, match="zero model probability"):
        model.log_likelihood(pd.DataFrame({"A": [1]}))


def test_nmi_sparse_pairs_constants_and_permutations():
    # The largest (x, y) joint cell can be empty: bincount needs minlength.
    assert nmi([0, 0, 1, 1, 2], [1, 1, 0, 0, 0]) == pytest.approx(nmi([1, 1, 0, 0, 0], [0, 0, 1, 1, 2]))
    assert nmi([0, 0, 1, 1], [7, 7, 3, 3]) == pytest.approx(1)
    assert nmi([0, 0, 0], [1, 1, 1]) == 1
    assert nmi([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(0)
    assert nmi([0, 1, 1], [0, 0, 0]) == pytest.approx(0)


def test_mask_only_hides_observed_cells_and_is_reproducible():
    data = pd.DataFrame({"A": [0, 1, None, 0] * 30, "B": [1, 1, 0, 0] * 30}, dtype="Int64")
    a, mask = mcar_mask(data, ["A"], .5, 0)
    b, second_mask = mcar_mask(data, ["A"], .5, 0)
    assert a.equals(b) and mask.equals(second_mask)
    assert a.isna().equals(data.isna() | mask)
    assert not mask.B.any()
