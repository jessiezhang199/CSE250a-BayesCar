# Work split

Five parts, one per person. The table lists who owns each part, its status and where its code lives.

| Part | Owner | Status | Code |
| --- | --- | --- | --- |
| A. Data + E2 missing-data experiment | Jessie Zhang | Data prepared, E2 complete in notebook | [dataset/](dataset/), [complete-case_MLE_and_EM/](complete-case_MLE_and_EM/) |
| B. E1 structure learning | Kevin Nie | E1 complete | [structure/](structure/) |
| C. Parameter learning + E3 latent variable | Chenghao Xiong | Parameter learning and E3 complete | [parameter_learning/](parameter_learning/) |
| D. E4 exact and approximate inference | Yijun Wang | E4 complete | [inference/](inference/) |
| E. E5 price prediction + final integration | Chloe Huang | E5 complete, code integrated | [prediction/](prediction/) |

## A. Data + E2 missing-data experiment

Select variables and exclude columns derived from price or near-duplicate price columns. Fix the 80/10/10 split and fit bin edges on the training set only. Call C's EM and compare complete-case deletion, mode imputation and EM under the same MCAR masks.

A delivers the data processing and masking scripts, the missing-rate comparison table, and report §2 and E2.

## B. E1 structure learning

Maintain the expert DAG and learn a structure with BIC hill climbing. Compare the two graphs under the same variables, bins and parameter estimation rule. If time allows, add Chow-Liu and conditional independence tests.

B delivers the DAG figure, the structure config, the test-set log-likelihood comparison, and the report's structure section and E1.

## C. Parameter learning + E3 latent variable

Implement smoothed MLE and a reusable EM. First hand the EM interface for the observed-condition model to A, then fit a three-state latent variable `H`. Hold `condition_grade` out for post-hoc evaluation and measure its association with `H` using NMI.

C delivers the CPT learning code, EM convergence curves, multi-initialization results, and the report's learning section and E3.

## D. E4 exact and approximate inference

Implement VE and check it against `pgmpy`. Run likelihood weighting and Gibbs first, and compare posterior error and runtime under common and rare evidence on the same DAG and CPTs. Add rejection sampling if time allows.

D delivers a reusable posterior query interface, TV distance and runtime plots, and the report's inference section and E4.

## E. E5 price prediction + final integration

Use D's VE to compare the expert DAG, the learned DAG and simple baselines on predicting the `listing_price` tier. Use the same test rows and available evidence for all models, and exclude variables not yet known at prediction time, such as `days_on_market`.

E delivers accuracy, log-loss and calibration plots, reproduction notes, report E5 and the conclusion, and the final submission package.

## Shared conventions

- Load splits with `common.load_split(name)` and take state order from `dataset/prepared/schema.json`.
- Read the expert and learned DAGs from `structure/outputs/dags.json` so every part uses the same structures.
- Use at most 4 parents per node and smoothed CPTs. Choose settings on the validation set and keep the test set for final numbers.
- Stochastic experiments use three seeds and report variability.
