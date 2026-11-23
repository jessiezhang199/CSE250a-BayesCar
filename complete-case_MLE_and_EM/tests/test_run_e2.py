import importlib

import numpy as np
import pytest

from parameter_learning import DiscreteBN

mean_cpt_kl = importlib.import_module("complete-case_MLE_and_EM.run_e2").mean_cpt_kl


def test_cpt_kl_is_zero_for_identical_models_and_positive_otherwise():
    def model(p):
        return DiscreteBN({"A": (), "B": ("A",)}, {"A": 2, "B": 2},
                          {"A": [[0.5, 0.5]], "B": [[p, 1 - p], [0.3, 0.7]]})

    reference = model(0.9)
    assert mean_cpt_kl(reference, model(0.9)) == 0
    expected = (0.9 * np.log(0.9 / 0.6) + 0.1 * np.log(0.1 / 0.4)) / 2 / 2
    assert mean_cpt_kl(reference, model(0.6)) == pytest.approx(expected)
