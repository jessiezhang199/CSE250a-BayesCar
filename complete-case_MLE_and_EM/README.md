# Part A · Missing data and E2

E2 asks how complete-case MLE, mode imputation followed by MLE, and EM compare when more of the training data goes missing. [run_e2.py](run_e2.py) runs the full sweep and writes the tables in [outputs/](outputs/). Jessie's notebook [MLE_EM.ipynb](MLE_EM.ipynb) is the original version with the discussion, and the script gives the same numbers. Part A's data preparation is in [dataset/](../dataset/).

## Reproduce

Run from the repository root after installing the root `requirements.txt`. It takes about 20 seconds.

```bash
python -m pytest -q complete-case_MLE_and_EM/tests
python -m complete-case_MLE_and_EM.run_e2
```

Opening the notebook needs Jupyter, which is not part of `requirements.txt`.

## How the run works

- The model is the expert DAG from [dags.json](../structure/outputs/dags.json) with add-one smoothing, fit with [Part C's](../parameter_learning/README.md) `fit_mle` and `fit_em`.
- For each rate (0%, 20% and 40%) and masking seed (0, 1 and 2), `mcar_mask` hides that share of the observed `service_history_quality` and `overall_condition_score` cells in the training split, independently and completely at random. The original missing cells stay missing, so 0% means the data as it is.
- All three methods fit the same masked copy. Mode imputation learns its modes from the masked training data only. EM starts from the mode-imputed CPTs and stops when the objective gains less than 1e-7 nats per row.
- Every model is scored on the full test split, with missing test cells summed out. Recovery accuracy comes from one fixed 20% test mask (seed 2026), and only the cells hidden by that mask are scored.
- CPT KL compares each fit with complete-case MLE on the unmasked training split, averaged over parent configurations and then over nodes.

## Outputs

| File | Contents |
| --- | --- |
| [summary.csv](outputs/summary.csv) | Mean and standard deviation across seeds of rows used, test log-likelihood, CPT KL and recovery accuracy for each rate and method |
| [runs.csv](outputs/runs.csv) | The same metrics for every rate, seed and method |
| [paired_wins.csv](outputs/paired_wins.csv) | How many of the three seeds EM beats complete-case MLE on test log-likelihood and CPT KL |
| [em_diagnostics.csv](outputs/em_diagnostics.csv) | EM convergence, iteration count and the share of cells actually hidden for each mask |
