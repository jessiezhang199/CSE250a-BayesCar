# Part E · Price-tier prediction and E5

E5 predicts the five-tier `listing_price` on the test split with exact VE and compares the expert DAG, the learned DAG and simple baselines. Results are in the [E5 report](REPORT.md).

## Reproduce

Run these from the repository root after installing the root `requirements.txt`.

```bash
python -m pytest -q prediction/tests
python -m prediction.run_e5
```

The run takes about three minutes and writes to `prediction/outputs/`. Pass `--output-dir /path/to/results` to write somewhere else. The outputs are deterministic.

## How the run works

- Every model scores the same 10,000 test rows. A missing feature is left out of the evidence and [`query_ve`](../inference/exact.py) sums it out. 526 test rows have at least one missing feature.
- There are two evidence sets. `all_features` uses all 11 non-price variables, and `no_condition_history` drops `accident_history`, `service_history_quality` and `overall_condition_score`. Part A already removed `days_on_market` and `demand_category`.
- The expert and learned DAGs use Part C's EM fits for seeds 0, 1 and 2. The primary numbers come from the seed chosen in [`selection.json`](../parameter_learning/outputs/selection.json).
- The Chow-Liu, naive Bayes and empty-graph baselines take their edges from [`dags.json`](../structure/outputs/dags.json). Each is fit by EM with Part C's settings and seeds, and its seed is picked by the same validation log-likelihood rule.
- Logistic regression is multinomial on one-hot features, with an extra column marking a missing value. Validation log-loss picks `C` separately for each evidence set.

## Outputs

| File | Contents |
| --- | --- |
| [summary.csv](outputs/summary.csv) | Accuracy, log-loss, mean absolute tier error and ECE for the selected models, plus the spread across seeds |
| [paired.csv](outputs/paired.csv) | Per-row paired gains over the expert and learned DAGs with standard errors |
| [runs.csv](outputs/runs.csv) | Metrics for every model, seed and evidence set |
| [calibration.csv](outputs/calibration.csv) | Top-label reliability bins for the selected models |
| [e5_metrics.png](outputs/e5_metrics.png) | Accuracy and log-loss for both evidence sets |
| [e5_calibration.png](outputs/e5_calibration.png) | Reliability diagrams |
| [settings.json](outputs/settings.json) | Evidence sets, EM settings, seed and `C` selection, Part C model hashes and package versions |

[`core.py`](core.py) holds the functions other parts can reuse, such as `predict_bn` for VE predictions over a data frame, `one_hot` and `fit_logistic` for the baseline, and `scores`, `reliability` and `ece` for the metrics.
