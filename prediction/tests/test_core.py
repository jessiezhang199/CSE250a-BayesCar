import json

import numpy as np
import pandas as pd
import pytest

from common import MODELS, PREPARED, TARGET, load_dags, load_split
from parameter_learning import DiscreteBN, fit_mle
from prediction.core import (EVIDENCE_SETS, FEATURES, ece, evidence_rows, one_hot, predict_bn,
                             reliability, scores)
from structure.bn import dag_from_edges

ARTIFACT = load_dags()
CARD = ARTIFACT["cardinality"]


@pytest.fixture(scope="module")
def val():
    return load_split("val")


def test_feature_sets_exclude_target_and_match_schema():
    schema = json.loads((PREPARED / "schema.json").read_text())
    assert TARGET not in FEATURES
    assert set(FEATURES) | {TARGET} == set(schema["columns"]) == set(ARTIFACT["nodes"])
    assert set(EVIDENCE_SETS["no_condition_history"]) < set(FEATURES)


def test_evidence_rows_skip_missing_cells(val):
    rows = val.loc[val["overall_condition_score"].isna()].head(3)
    for evidence in evidence_rows(rows, FEATURES):
        assert "overall_condition_score" not in evidence
        assert all(isinstance(v, int) for v in evidence.values())
        assert TARGET not in evidence


def test_predict_bn_matches_enumeration_with_missing_values(val):
    model = DiscreteBN.load(MODELS / "hill_climb_bic_selected.json")
    rows = pd.concat([val.loc[val["service_history_quality"].isna()].head(20), val.head(20)])
    hidden = rows.copy()
    hidden[TARGET] = pd.NA
    expected = model.posterior(hidden, TARGET)
    np.testing.assert_allclose(predict_bn(model, rows, FEATURES), expected, atol=1e-10)


def test_naive_bayes_prediction_is_bayes_rule(val):
    train = load_split("train").dropna()
    dag = dag_from_edges(ARTIFACT["nodes"], ARTIFACT["edges"]["naive_bayes"])
    model = fit_mle(dag, train, CARD)
    row = val.dropna().head(1)
    evidence = evidence_rows(row, FEATURES)[0]
    joint = model.cpts[TARGET].reshape(-1).copy()
    for node, state in evidence.items():
        joint *= model.cpts[node][:, state]
    np.testing.assert_allclose(predict_bn(model, row, FEATURES)[0], joint / joint.sum())


def test_one_hot_gives_missing_its_own_column(val):
    rows = val.loc[val["overall_condition_score"].isna()].head(5)
    x = one_hot(rows, FEATURES, CARD)
    assert x.shape == (5, sum(CARD[f] + 1 for f in FEATURES))
    np.testing.assert_array_equal(x.sum(axis=1), len(FEATURES))
    start = sum(CARD[f] + 1 for f in FEATURES[:FEATURES.index("overall_condition_score")])
    np.testing.assert_array_equal(x[:, start + CARD["overall_condition_score"]], 1)


def test_scores_on_perfect_and_uniform_predictions():
    y = np.array([0, 1, 2, 3, 4, 4])
    perfect = np.eye(5)[y]
    result = scores(perfect, y)
    assert result["accuracy"] == 1 and result["tier_mae"] == 0 and result["ece"] == 0
    assert result["log_loss"] == pytest.approx(0, abs=1e-12)
    assert scores(np.full((6, 5), 0.2), y)["log_loss"] == pytest.approx(np.log(5))


def test_reliability_and_ece():
    probs = np.array([[0.9, 0.1], [0.9, 0.1], [0.6, 0.4], [0.4, 0.6]])
    y = np.array([0, 1, 0, 1])
    table = reliability(probs, y)
    assert table["count"].sum() == len(y)
    assert ece(probs, y) == pytest.approx((2 * abs(0.5 - 0.9) + 2 * abs(1.0 - 0.6)) / 4)
