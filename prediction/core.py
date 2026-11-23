"""Evidence encoding, model predictions and scores for E5 price-tier prediction."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from common import TARGET
from inference import query_ve

FEATURES = (
    "brand_segment", "body_type", "fuel_type", "accident_history", "service_history_quality",
    "urban_rural", "vehicle_age", "number_of_previous_owners", "engine_power_hp",
    "odometer_km", "overall_condition_score",
)
HISTORY_FIELDS = ("accident_history", "service_history_quality", "overall_condition_score")
EVIDENCE_SETS = {
    "all_features": FEATURES,
    "no_condition_history": tuple(f for f in FEATURES if f not in HISTORY_FIELDS),
}


def evidence_rows(frame, features):
    """One feature-to-state dict per row, leaving out missing cells."""
    values = frame[list(features)].to_numpy(dtype=float, na_value=np.nan)
    return [{f: int(v) for f, v in zip(features, row) if not np.isnan(v)} for row in values]


def predict_bn(model, frame, features):
    """P(listing_price | observed features) for each row by exact VE."""
    cache = {}
    out = np.empty((len(frame), model.card[TARGET]))
    for i, evidence in enumerate(evidence_rows(frame, features)):
        key = tuple(evidence.items())
        if key not in cache:
            cache[key] = query_ve(model, TARGET, evidence)
        out[i] = cache[key]
    return out


def one_hot(frame, features, card):
    """Indicator columns per feature with an extra column for a missing value."""
    blocks = []
    for f in features:
        codes = frame[f].to_numpy(dtype=float, na_value=np.nan)
        index = np.where(np.isnan(codes), card[f], codes).astype(int)
        block = np.zeros((len(frame), card[f] + 1))
        block[np.arange(len(frame)), index] = 1
        blocks.append(block)
    return np.hstack(blocks)


def fit_logistic(train, val, features, card, grid):
    """Multinomial logistic regression with C chosen by validation log-loss."""
    x_train, x_val = one_hot(train, features, card), one_hot(val, features, card)
    y_train, y_val = labels(train), labels(val)
    losses = {}
    for c in grid:
        fit = LogisticRegression(C=c, max_iter=5000).fit(x_train, y_train)
        losses[c] = log_loss(y_val, fit.predict_proba(x_val), labels=range(card[TARGET]))
    best = min(losses, key=losses.get)
    return LogisticRegression(C=best, max_iter=5000).fit(x_train, y_train), best, losses


def labels(frame):
    return frame[TARGET].to_numpy(dtype=int)


def per_row(probs, y):
    """Per-row correctness, negative log-likelihood and absolute tier error."""
    pred = probs.argmax(axis=1)
    p_true = np.clip(probs[np.arange(len(y)), y], 1e-15, None)
    return pd.DataFrame({"correct": (pred == y).astype(float), "nll": -np.log(p_true),
                         "tier_error": np.abs(pred - y).astype(float)})


def reliability(probs, y, bins=10):
    """Top-label reliability table over equal-width confidence bins."""
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    which = np.minimum((confidence * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        mask = which == b
        rows.append({"bin": b, "lower": b / bins, "upper": (b + 1) / bins, "count": int(mask.sum()),
                     "mean_confidence": confidence[mask].mean() if mask.any() else np.nan,
                     "accuracy": correct[mask].mean() if mask.any() else np.nan})
    return pd.DataFrame(rows)


def ece(probs, y, bins=10):
    table = reliability(probs, y, bins).dropna()
    return float((table["count"] * (table["accuracy"] - table["mean_confidence"]).abs()).sum() / len(y))


def scores(probs, y):
    rows = per_row(probs, y)
    return {"accuracy": rows.correct.mean(), "log_loss": rows.nll.mean(),
            "tier_mae": rows.tier_error.mean(), "ece": ece(probs, y)}
