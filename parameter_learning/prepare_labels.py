"""Extract evaluation-only condition_grade after verifying source-row alignment.

python -m parameter_learning.prepare_labels --raw-csv dataset/cars_dataset.csv
No raw dataset or label enters an estimator; only these small sidecars are saved.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def prepare(raw_csv, output=None):
    raw_csv = Path(raw_csv)
    output = Path(output) if output else ROOT / "parameter_learning" / "labels"
    schema = json.loads((ROOT / "dataset/prepared/schema.json").read_text())["columns"]
    raw = pd.read_csv(raw_csv, usecols=[*schema, "condition_grade"])
    raw.index.name = "source_row"
    encoded = pd.DataFrame(index=raw.index)
    for v, info in schema.items():
        x = raw[v]
        if v == "body_type":
            x = x.mask(x.notna() & ~x.isin(["Sedan", "SUV", "Pickup"]), "Other")
        if "cut_points" in info:
            encoded[v] = pd.cut(x, [-np.inf, *info["cut_points"], np.inf], labels=False).astype("Int64")
        else:
            encoded[v] = x.map({s: i for i, s in enumerate(info["states"])}).astype("Int64")
    prepared, indices = {}, []
    for split in ("train", "val", "test"):
        frame = pd.read_csv(ROOT / f"dataset/prepared/{split}.csv", index_col="source_row", dtype="Int64")
        # Compare every model cell and missingness bit, not only row counts.
        expected = encoded.loc[frame.index, frame.columns]
        # read_csv(dtype='Int64') also makes the index nullable. .loc preserves
        # the raw integer index dtype, so normalize dtype after checking IDs.
        np.testing.assert_array_equal(expected.index.to_numpy(), frame.index.to_numpy())
        expected.index = frame.index
        if not frame.equals(expected):
            different = (frame != expected).fillna(False) | (frame.isna() != expected.isna())
            raise ValueError(f"Original data does not match {split}; refusing to assign labels: "
                             f"{different.sum().to_dict()}; sample rows "
                             f"{frame.index[different.any(axis=1)][:5].tolist()}")
        prepared[split] = raw.loc[frame.index, ["condition_grade"]]
        indices.extend(frame.index)
    if len(indices) != len(set(indices)) or set(indices) != set(raw.index):
        raise ValueError("Splits are not a disjoint partition of the source")
    output.mkdir(parents=True, exist_ok=True)
    for split, labels in prepared.items():
        labels.to_csv(output / f"{split}.csv", lineterminator="\n")
    sha = hashlib.file_digest(raw_csv.open("rb"), "sha256").hexdigest()
    provenance = {"source": "https://www.kaggle.com/datasets/datascikhan/cars-dataset",
                  "raw_sha256": sha, "raw_rows": len(raw), "verified_model_cells": len(raw) * len(schema),
                  "alignment": "All encoded model cells, missingness and source_row checked against prepared splits",
                  "use": "Post-hoc evaluation only; never training, initialization or model selection",
                  "grade_counts": raw.condition_grade.value_counts(dropna=False).to_dict()}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    prepare(args.raw_csv, args.output)
