"""Small label-invariant diagnostics without a scikit-learn dependency."""
import numpy as np


def normalized_mutual_information(x, y):
    """NMI with arithmetic-mean entropy normalization, range [0, 1]."""
    if len(x) != len(y) or not len(x):
        raise ValueError("NMI requires paired, nonempty label arrays")
    _, a = np.unique(x, return_inverse=True)
    _, b = np.unique(y, return_inverse=True)
    n = np.bincount(a * (b.max()+1) + b, minlength=(a.max()+1)*(b.max()+1)).reshape(a.max()+1, b.max()+1).astype(float)
    p = n / n.sum()
    px, py = p.sum(axis=1), p.sum(axis=0)
    h = lambda q: -np.sum(q[q > 0] * np.log(q[q > 0]))
    hx, hy = h(px), h(py)
    if hx + hy == 0:
        return 1.0
    expected = px[:, None] * py[None, :]
    observed = p > 0
    mi = np.sum(p[observed] * np.log(p[observed] / expected[observed]))
    return float(np.clip(2 * mi / (hx + hy), 0, 1))
