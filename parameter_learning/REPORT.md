# Part C — Parameter estimation and E3 latent-variable analysis

## Parameter learning

We retain the upstream 80,000/10,000/10,000 train/validation/test split, twelve
discrete variables, training-derived state encodings, and expert/learned DAGs.
Natural blanks in service history and overall condition remain unknown values,
rather than extra categories. All model parameters are estimated on training
rows only. All three initializations use seeds 0, 1, 2. Validation observed-data
log-likelihood selects the initialization; test data and condition-grade labels
do not enter that decision.

For node X with r states and parent configuration u, smoothed complete-data
estimation is

\[
\theta_{x|u} = \frac{N(x,u)+\alpha}{N(u)+r\alpha},\quad\alpha=1.
\]

This matches the add-one convention used in E1. The complete-case baseline uses
76,081 training rows. Incomplete-data EM retains all 80,000 rows. Its E-step
computes the exact posterior over the **joint** missing assignment conditional
on every observed variable, then accumulates expected family counts. Its M-step
uses the formula above with expected counts. The implementation compresses
duplicate evidence, enumerates only missing states, and caches family indices.
A completion budget fails clearly rather than silently approximating inference.

Add-one EM is MAP estimation under independent Dirichlet(2) priors, optimizing

\[
J(\theta)=\sum_i\log P_\theta(x^{(i)}_{obs})
             +\alpha\sum_{X,u,x}\log\theta_{x|u}.
\]

We record both observed-data likelihood and J. Only J is guaranteed to be
monotone for positive alpha. We stop when the absolute change in J per training
row is at most 1e-7. At alpha=0 the same implementation becomes ordinary ML EM.
Initialization zero is included in every history. Unseen parent rows under pure
ML are uniform for complete-data fitting, or preserve their previous EM row.

## Observed-model results

Every entry below scores the same 10,000 validation or test records, integrating
their naturally missing cells. These likelihoods therefore differ from E1's
complete-case-only evaluation and must not be copied into E1's table.

| Structure | Estimator | Training rows | Parameters | Validation LL/row | Test LL/row |
| --- | --- | ---: | ---: | ---: | ---: |
| Expert | Complete-case add-one MLE | 76,081 | 611 | -10.875967 | -10.864066 |
| Expert | EM | 80,000 | 611 | -10.875807 | -10.863744 |
| BIC hill climbing | Complete-case add-one MLE | 76,081 | 851 | -10.737514 | -10.725022 |
| BIC hill climbing | EM | 80,000 | 851 | -10.737294 | -10.725265 |

All six observed-model EM fits converge in four M-steps, with essentially
identical likelihoods across initializations. Validation selects seed 2 for both
structures; the differences between these seeds are numerical rather than
substantively different solutions. EM improves the expert's test likelihood by
0.000322 nats/row relative to deletion, but its learned-DAG test likelihood is
0.000243 nats/row lower than deletion. Thus retaining missing records does not
guarantee a test improvement. Learned-DAG EM improves on expert EM by
0.138480 nats/row, with paired standard error 0.005164.

The E2 handoff also runs successfully on the full training set with one shared
20% additional MCAR mask (seed 0) in the two specified columns. Validation
LL/row is -10.876601 for EM, -10.877733 for complete-case MLE, and -10.975283 for
mode-imputed MLE. This is an integration check; A owns the full missing-rate and
three-seed E2 comparison. Modes are learned only from the masked training data.

## E3 specification

The work split specifies three states for H but does not fix its edges. We
therefore explicitly define a separate E3 graph, based on the expert graph:
H is a root parent of accident history, service history, overall condition score
and listing price. Remove service_history_quality->overall_condition_score to
respect the four-parent cap, and replace overall_condition_score->listing_price
with H->listing_price. Retain all other expert edges. H is always unobserved;
condition_grade is never a model variable. The graph has 13 nodes and 701 nominal
free parameters. It is a modeling hypothesis, not a causal interpretation or a
new structure-learning result.

The original dataset was recovered and all 1,200,000 encoded model cells,
missingness bits and split memberships verified before attaching grade labels
by source_row. Raw-file SHA-256 is recorded in labels/provenance.json. Missing
grade labels are excluded only from NMI evaluation; all rows remain in fitting
and likelihood evaluation. The test split has 9,917 observed grades. Labels are
read only after training and validation-based initialization selection finish.

| Initialization | M-steps | Converged | Validation LL/row | Selected |
| --- | ---: | --- | ---: | --- |
| Seed 0 | 1,282 | Yes | -10.891996 | No |
| Seed 1 | 1,942 | Yes | -10.891739 | Yes |
| Seed 2 | 1,374 | Yes | -10.892122 | No |

All three runs meet the same convergence tolerance. The selected H model's test
LL/row is -10.883721, which is **0.019977 worse** than expert EM (paired standard
error 0.001583), despite having more parameters. This experiment provides no
evidence that this particular latent graph improves the held-out density model.

## Post-hoc association with condition grade

For each fitted H model, infer the posterior and assign each row its MAP H
state. We use label-permutation-invariant NMI, normalized by the arithmetic mean
of the two entropies. H's state numbers have no inherent good/bad ordering. No
grade-based state relabeling or tuning is performed.

| Posterior evidence | Selected seed test NMI | Three-seed mean ± sample SD |
| --- | ---: | ---: |
| All available observed variables | 0.115133 | 0.084943 ± 0.026218 |
| Hide overall condition score | 0.019023 | 0.016615 ± 0.003947 |
| Hide both condition score and price | 0.128422 | 0.084294 ± 0.043166 |

These are evidence-sensitivity checks on the same fitted model; they do not
retrain without the hidden indicators. Price is observed in the first two rows,
so these results are not price-prediction metrics. With full evidence, 100 label
permutations give mean NMI 0.000395 and maximum 0.001043 for the selected seed.
The association is above that finite-sample null, but modest in magnitude.

For context, the observed binned condition score itself has test NMI 0.719368
with grade on 9,816 jointly observed rows; age and price have NMI 0.381046 and
0.246522 on 9,917 rows. These are post-hoc indicator comparisons, not additional
training criteria. H does not recover a grade-like partition comparable to the
direct condition-score indicator.

The selected model's full-evidence hard state occupancies are 42.36%, 56.99% and
0.65%; posterior-average occupancies are 41.00%, 57.37% and 1.63%. The third state
is weakly used. Mean posterior entropy is 0.436617 nats (maximum log(3)). Without
both score and price, hard assignments put 91.05% of records into one state.
NMI can rise when these variables are removed because MAP assignment and
entropy normalization change; it is not a monotonic measure of evidence
informativeness. The contingency tables and posterior probabilities are
included so this behavior is visible.

The defensible E3 conclusion is that this three-state factor has weak and
initialization-sensitive association with grade and does not improve held-out
likelihood. We do not claim discovery of three meaningful condition tiers.
All conclusions concern this synthetic dataset. A stronger latent design would
require a separately specified model and validation study, rather than tuning
this model against the already reported test NMI.

## Verification and artifacts

The test suite checks exact evidence marginalization against independent full
joint enumeration, a hand-calculated EM update, parent-order compatibility with
E1, complete-data EM/MLE equivalence, reproducibility, objective monotonicity,
synthetic latent recovery up to label permutation, serialization, missing-column
guards, invalid state rejection, completion limits, sparse NMI and MCAR masks.

- `outputs/em_convergence.png`: likelihood and regularized-objective curves for all nine fits.
- `outputs/em_convergence_detail.png`: latent likelihood and stopping-tolerance detail after the initialization jump.
- `outputs/restarts.csv`: convergence, timing and likelihood by initialization.
- `outputs/likelihood_comparison.csv`: common-row comparisons and paired gains.
- `outputs/e3_association.png`: NMI sensitivity and selected-state grade composition.
- `outputs/e3_nmi.csv`, `e3_nmi_across_seeds.csv`, `e3_contingency.csv`: detailed E3 diagnostics.
- `outputs/observed_indicator_nmi.csv`: indicator/grade comparison denominators.
- `outputs/models/`: all fitted CPTs plus validation-selected models, ordered parent metadata included.
- `outputs/test_H_*.csv`: selected-model test posteriors, keyed by source_row, without grade labels.
- `outputs/manifest.json`: fixed inputs, hashes, configuration and runtime metadata.

Reproduce from the repository root with `python -m parameter_learning.run_part_c`.
Use the package README and `e2_handoff.py` for A/D/E integration instructions.
