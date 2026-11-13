"""Verify persisted experiment results and integration contracts without refitting."""
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from parameter_learning import DiscreteBN

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "parameter_learning/outputs"


def hash_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify():
    manifest = json.loads((OUT / "manifest.json").read_text())
    for relative, expected in manifest["input_sha256"].items():
        assert hash_file(ROOT / relative) == expected, f"Input changed: {relative}"
    runs = pd.read_csv(OUT / "restarts.csv")
    assert len(runs) == 9 and runs.converged.all()
    selected = json.loads((OUT / "selection.json").read_text())
    for name, group in runs.groupby("model"):
        winner = group.loc[group.val_ll_per_row.idxmax()]
        assert selected[name]["seed"] == winner.seed
        chosen = DiscreteBN.load(OUT / "models" / f"{name}_selected.json")
        assert max(map(len, chosen.dag.values())) <= 4
        for row in group.itertuples():
            key = f"{name}_seed{row.seed}"
            history = pd.read_csv(OUT / "histories" / f"{key}.csv")
            assert history.iteration.tolist() == list(range(row.iterations+1))
            assert (history.objective.diff().dropna() >= -1e-7).all()
            assert abs(history.objective_gain_per_row.iloc[-1]) <= manifest["config"]["tol"]
            restored = DiscreteBN.load(OUT / "models" / f"{key}.json")
            if row.seed == winner.seed:
                for v in chosen.dag:
                    np.testing.assert_array_equal(chosen.cpts[v], restored.cpts[v])
            test = pd.read_csv(ROOT / "dataset/prepared/test.csv", index_col="source_row", dtype="Int64")
            # Confirms that serialized CPTs reproduce the recorded test metric.
            ll = restored.log_likelihood(test, latent_nodes=("H",) if "H" in restored.dag else ())
            np.testing.assert_allclose(ll.mean(), row.test_ll_per_row, atol=1e-11, rtol=0)
    for path in OUT.glob("test_H_*.csv"):
        probabilities = pd.read_csv(path, index_col="source_row")
        assert len(probabilities) == 10000 and probabilities.index.is_unique
        np.testing.assert_allclose(probabilities[["p_H0", "p_H1", "p_H2"]].sum(axis=1), 1, atol=1e-12)
    nmi = pd.read_csv(OUT / "e3_nmi.csv")
    assert len(nmi) == 18 and nmi.nmi.between(0, 1).all()
    assert (nmi.query('split == "test"').labeled_rows == 9917).all()
    print("Verified: input hashes, 9 converged fits, monotone objectives, validation selection, "
          "serialized test likelihoods, CPT parent cap, 3 posterior exports and 18 NMI evaluations.")


if __name__ == "__main__":
    verify()
