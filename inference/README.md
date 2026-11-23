# Part D · Inference and E4

Part D provides a single-node posterior query for Part E and compares variable elimination (VE), likelihood weighting, Gibbs and rejection sampling on two fixed price-tier queries. The [E4 report](REPORT.md) includes the measured results and figure.

## Use VE from Part E

Run from the repository root. This example uses the same validation row and available evidence for both models. It selects a row with a missing condition score to illustrate marginalization.

```python
from pathlib import Path
import pandas as pd
from inference import query_ve
from parameter_learning import DiscreteBN

val = pd.read_csv("dataset/prepared/val.csv", index_col="source_row", dtype="Int64")
row = val.loc[val["overall_condition_score"].isna()].iloc[0]
features = ["vehicle_age", "odometer_km", "overall_condition_score"]
evidence = row[features].dropna().astype(int).to_dict()

models = Path("parameter_learning/outputs/models")
for name in ("expert", "hill_climb_bic"):
    model = DiscreteBN.load(models / f"{name}_selected.json")
    probabilities = query_ve(model, "listing_price", evidence)
    print(name, probabilities, probabilities.argmax())
```

Load each model once when scoring multiple rows. Pass the agreed listing-time features as a dictionary of integer state codes, skipping missing values. Keep the true price separately for E5 scoring. The return value is a normalized NumPy vector in [schema state order](../dataset/prepared/schema.json). Empty evidence gives the marginal. Including an observed target gives a point mass when the evidence is possible. Unknown nodes, invalid states and impossible evidence raise a `ValueError`.

## Reproduce E4

Run these from the repository root after installing the root `requirements.txt`.

```bash
python -m inference.check_setup
python -m pytest -q inference/tests parameter_learning/tests
python -m inference.run_e4
```

The setup check confirms the [DAGs](../structure/outputs/dags.json) and schema match the [selected Part C models](../parameter_learning/README.md). It compares VE, pgmpy and Part C enumeration on both evidence cases and graphs. The tests also cover missing fields in a real validation row, downstream evidence, empty evidence, CPT axis order, structural zeros and the samplers.

The benchmark uses the expert graph, reads the existing validation split, and writes the following files. Use `python -m inference.run_e4 --output-dir /path/to/results` to reproduce in a separate directory. Random seeds fix the posterior estimates. Runtime depends on the machine and its load.

| File | Contents |
| --- | --- |
| [exact.csv](outputs/exact.csv) | Evidence counts, model evidence probabilities, exact posteriors and VE timing |
| [gibbs_pilot.csv](outputs/gibbs_pilot.csv) | Pilot TV distances for each burn-in, evidence case and independent seed |
| [runs.csv](outputs/runs.csv) | All 54 estimates, timings, weight ESS, accepted counts and Gibbs lag-one diagnostics |
| [summary.csv](outputs/summary.csv) | Valid-run counts and statistics across the three evaluation seeds |
| [e4_comparison.png](outputs/e4_comparison.png) | Error and runtime by evidence case and budget |
| [settings.json](outputs/settings.json) | Seeds, selected burn-in, timing settings, model hash and software versions |

The three sampling functions in [sampling.py](sampling.py) accept the same model, target and evidence plus `n` and a keyword `seed`. Gibbs also accepts `burn_in`. Each returns a `SampleResult`. A rejection run with zero accepted samples or a weighting run with zero total weight returns `posterior=None`, so downstream summaries can count failures explicitly.
