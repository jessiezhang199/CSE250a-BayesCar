"""E4: compare likelihood weighting, Gibbs and rejection sampling with exact VE.

python -m inference.run_e4
Runs the fixed validation queries on the expert network and writes the benchmark,
the Gibbs burn-in pilot and the comparison figure.
"""
import argparse
from importlib.metadata import version
from itertools import product
import json
from pathlib import Path
import platform
from statistics import median
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import MODELS, ROOT, TARGET, load_split, sha256
from inference import query_ve
from inference.check_setup import CASES, EVIDENCE_FIELDS, pgmpy_reference
from inference.sampling import gibbs_sampling, likelihood_weighting, rejection_sampling
from parameter_learning import DiscreteBN

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
MODEL = MODELS / "expert_selected.json"
BUDGETS = (100, 1_000, 10_000)
SEEDS = (0, 1, 2)
METHODS = {"Likelihood weighting": likelihood_weighting, "Gibbs": gibbs_sampling,
           "Rejection": rejection_sampling}
PILOT_SEEDS = (101, 102, 103)
BURN_CANDIDATES = (50, 200, 500)
PILOT_DRAWS = 2_000
PILOT_TV_LIMIT = 0.08


def tv(p, q):
    return float(np.abs(p - q).sum() / 2)


def timed(fn, *args, repeats=3, **kwargs):
    elapsed = []
    for _ in range(repeats):
        start = perf_counter()
        result = fn(*args, **kwargs)
        elapsed.append((perf_counter() - start) * 1_000)
    return result, median(elapsed)


def select_burn_in(model, cases, exact):
    rows = []
    for case, evidence in cases.items():
        for seed in PILOT_SEEDS:
            result = gibbs_sampling(model, TARGET, evidence, max(BURN_CANDIDATES) + PILOT_DRAWS,
                                    seed=seed)
            for skip in BURN_CANDIDATES:
                window = result.trace[skip:skip + PILOT_DRAWS]
                p = np.bincount(window, minlength=model.card[TARGET]) / PILOT_DRAWS
                rows.append(dict(case=case, seed=seed, burn_in=skip, tv=tv(p, exact[case])))
    pilot = pd.DataFrame(rows)
    worst = pilot.groupby("burn_in").tv.max()
    passing = worst[worst <= PILOT_TV_LIMIT]
    if passing.empty:
        raise RuntimeError(f"Gibbs pilot failed TV criterion. Worst TV by burn-in: {worst.to_dict()}")
    return int(passing.index.min()), pilot


def summarize(runs):
    groups = runs.groupby(["case", "method", "budget"], sort=False)
    return groups.agg(
        runs=("seed", "size"), valid_posteriors=("tv", "count"),
        mean_tv=("tv", "mean"), sd_tv=("tv", "std"), min_tv=("tv", "min"), max_tv=("tv", "max"),
        median_ms=("runtime_ms", "median"), min_ms=("runtime_ms", "min"), max_ms=("runtime_ms", "max"),
        mean_weight_ess=("weight_ess", "mean"), mean_accepted=("accepted", "mean"),
        mean_tier0_lag1=("tier0_lag1", "mean"),
    ).reset_index().sort_values(["case", "method", "budget"])


def plot(summary, references, path):
    styles = {"Likelihood weighting": ("#157a78", "o"), "Gibbs": ("#96521c", "s"),
              "Rejection": ("#585399", "^")}
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey="col", layout="constrained")
    for i, case in enumerate(CASES):
        error_ax, time_ax = axes[i]
        for method, (color, marker) in styles.items():
            rows = summary[(summary.case == case) & (summary.method == method)]
            for ax, center, low, high in ((error_ax, "mean_tv", "min_tv", "max_tv"),
                                          (time_ax, "median_ms", "min_ms", "max_ms")):
                ax.plot(rows.budget, rows[center], marker=marker, color=color, label=method)
                ax.fill_between(rows.budget, rows[low], rows[high], color=color, alpha=0.13)
            partial = rows[rows.valid_posteriors < rows.runs]
            if not partial.empty:
                notes = [f"{row.budget:,} draws   {row.valid_posteriors}/{row.runs}"
                         for row in partial.itertuples()]
                error_ax.text(0.04, 0.96, "\n".join([f"{method} estimates", *notes]),
                              transform=error_ax.transAxes, va="top", fontsize=8, color=color)
        ref = references.loc[case]
        time_ax.axhline(ref.ve_ms, color="#31363b", linestyle="--", linewidth=1.2, label="Exact VE")
        error_ax.set_ylabel(f"{case.capitalize()} evidence\nTV distance")
        time_ax.set_ylabel("Query time (ms)")
        for ax in axes[i]:
            ax.set(xscale="log", yscale="log", xticks=BUDGETS)
            ax.set_xticklabels(["100", "1,000", "10,000"])
            ax.grid(axis="y", which="major", alpha=0.22)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_title("Posterior error against exact VE")
    axes[0, 1].set_title("Wall time per query")
    for ax in axes[-1]:
        ax.set_xlabel("Draws / retained Gibbs sweeps")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run(output_dir=OUTPUT_DIR):
    out = Path(output_dir)
    model, load_ms = timed(DiscreteBN.load, MODEL)
    reference = pgmpy_reference(model)
    joint = reference.query(EVIDENCE_FIELDS, show_progress=False)
    counts = load_split("val")[EVIDENCE_FIELDS].value_counts()
    cases = {k: dict(zip(EVIDENCE_FIELDS, v)) for k, v in CASES.items()}
    exact = {}
    baselines = []
    for case, evidence in cases.items():
        ref = reference.query([TARGET], evidence=evidence, show_progress=False).values
        np.testing.assert_allclose(query_ve(model, TARGET, evidence), ref, atol=1e-10, rtol=0)
        exact[case], ve_ms = timed(query_ve, model, TARGET, evidence, repeats=31)
        baselines.append(dict(case=case, **evidence, val_count=counts[CASES[case]],
                              val_observed=counts.sum(), model_p_e=joint.get_value(**evidence),
                              ve_ms=ve_ms, **dict(zip(tier_columns(model), exact[case]))))
    references = pd.DataFrame(baselines).set_index("case")
    burn_in, pilot = select_burn_in(model, cases, exact)

    schedules = list(product(cases, METHODS, BUDGETS, SEEDS))
    np.random.default_rng(42).shuffle(schedules)
    rows = []
    for case, method, budget, seed in schedules:
        options = {"burn_in": burn_in} if method == "Gibbs" else {}
        estimate, ms = timed(METHODS[method], model, TARGET, cases[case], budget, seed=seed, **options)
        p = estimate.posterior
        lag1 = np.nan
        if estimate.trace is not None:
            indicator = estimate.trace == 0
            if indicator[:-1].std() > 0 and indicator[1:].std() > 0:
                lag1 = np.corrcoef(indicator[:-1], indicator[1:])[0, 1]
        rows.append(dict(case=case, method=method, budget=budget, seed=seed,
                         burn_in=options.get("burn_in", 0), runtime_ms=ms,
                         weight_ess=estimate.weight_ess, accepted=estimate.accepted, tier0_lag1=lag1,
                         tv=tv(p, exact[case]) if p is not None else np.nan,
                         **(dict(zip(tier_columns(model), p)) if p is not None else {})))

    runs = pd.DataFrame(rows).sort_values(["case", "method", "budget", "seed"])
    summary = summarize(runs)
    out.mkdir(parents=True, exist_ok=True)
    references.to_csv(out / "exact.csv")
    pilot.to_csv(out / "gibbs_pilot.csv", index=False)
    runs.to_csv(out / "runs.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    plot(summary, references, out / "e4_comparison.png")
    metadata = dict(model=str(MODEL.relative_to(ROOT).as_posix()),
                    model_sha256=sha256(MODEL),
                    model_load_ms=load_ms, target=TARGET, evidence=cases, budgets=BUDGETS, seeds=SEEDS,
                    burn_in=burn_in, pilot_seeds=PILOT_SEEDS, pilot_draws=PILOT_DRAWS,
                    pilot_burn_candidates=BURN_CANDIDATES, pilot_max_tv=PILOT_TV_LIMIT,
                    sampler_timing_repeats=3, ve_timing_repeats=31,
                    platform=platform.platform(), python=platform.python_version(),
                    packages={p: version(p) for p in ("numpy", "pandas", "networkx", "matplotlib", "pgmpy")})
    (out / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{len(runs)} sampling runs, burn-in {burn_in}, results in {out}")


def tier_columns(model):
    return [f"p_tier{i}" for i in range(model.card[TARGET])]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    run(parser.parse_args().output_dir)
