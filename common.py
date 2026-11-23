"""Paths and loaders shared by every part of the project."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PREPARED = ROOT / "dataset" / "prepared"
DAGS = ROOT / "structure" / "outputs" / "dags.json"
MODELS = ROOT / "parameter_learning" / "outputs" / "models"
TARGET = "listing_price"
SPLITS = ("train", "val", "test")


def load_split(name):
    """One prepared split with nullable integer state codes, indexed by source_row."""
    return pd.read_csv(PREPARED / f"{name}.csv", index_col="source_row", dtype="Int64")


def load_card():
    """Number of states for each variable, in schema order."""
    schema = json.loads((PREPARED / "schema.json").read_text())["columns"]
    return {col: len(info["states"]) for col, info in schema.items()}


def load_dags():
    """Part B's shared structure file with the node list, cardinalities and edge lists."""
    return json.loads(DAGS.read_text())


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
