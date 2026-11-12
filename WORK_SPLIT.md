# Work split

Five parts, one per person. Fill in your name next to your part and update the status when it changes.

| Part | Owner | Status | Code |
| --- | --- | --- | --- |
| A. Data + E2 missing-data experiment | Jessie Zhang | Data prepared, E2 pending | `dataset/` |
| B. E1 structure learning | Kevin Nie | Done, on branch `e1-structure-learning` | `structure/` |
| C. Parameter learning + E3 latent variable | TBD | Pending | |
| D. E4 exact and approximate inference | TBD | Pending | |
| E. E5 price prediction + final integration | TBD | Pending | |

## A. Data + E2 missing-data experiment

Select variables and exclude columns derived from price or near-duplicate price columns. Fix the 80/10/10 split and fit bin edges on the training set only. Call C's EM and compare complete-case deletion, mode imputation and EM under the same MCAR masks.

Deliverables: data processing and masking scripts, missing-rate comparison table, report §2 and E2.

## B. E1 structure learning

Maintain the expert DAG and learn a structure with BIC hill climbing. Compare the two graphs under the same variables, bins and parameter estimation rule. If time allows, add Chow-Liu and conditional independence tests.

Deliverables: DAG figure, structure config, test-set log-likelihood comparison, report structure section and E1.

## C. Parameter learning + E3 latent variable

Implement smoothed MLE and a reusable EM. First hand the EM interface for the observed-condition model to A, then fit a three-state latent variable `H`. Hold `condition_grade` out for post-hoc evaluation and measure its association with `H` using NMI.

Deliverables: CPT learning code, EM convergence curves, multi-initialization results, report learning section and E3.

## D. E4 exact and approximate inference

Implement VE and check it against `pgmpy`. Run likelihood weighting and Gibbs first, and compare posterior error and runtime under common and rare evidence on the same DAG and CPTs. Add rejection sampling if time allows.

Deliverables: reusable posterior query interface, TV distance and runtime plots, report inference section and E4.

## E. E5 price prediction + final integration

Use D's VE to compare the expert DAG, the learned DAG and simple baselines on predicting the `listing_price` tier. Use the same test rows and available evidence for all models, and exclude variables not yet known at prediction time, such as `days_on_market`.

Deliverables: accuracy, log-loss and calibration plots, reproduction notes, report E5 and conclusion, final submission package.

## Shared conventions

- Load splits with `pd.read_csv(path, index_col="source_row", dtype="Int64")` and take state order from `dataset/prepared/schema.json`.
- Read the expert and learned DAGs from `structure/outputs/dags.json` so every part uses the same structures.
- At most 4 parents per node, smoothed CPTs, validation set for choosing settings, test set only for final numbers.
- Stochastic experiments use three seeds and report variability.
