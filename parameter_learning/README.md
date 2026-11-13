# Part C: parameter learning and E3

This directory contains smoothed CPT MLE, exact EM for incomplete discrete
records, and the three-state latent-factor experiment. It uses the upstream
splits, schema and DAGs unchanged. `condition_grade` is evaluation-only.

## Reproduce

From the repository root, with Python 3.11 or newer:

```bash
pip install -r parameter_learning/requirements.txt
python -m pytest parameter_learning/tests -q
python -m parameter_learning.run_part_c
python -m parameter_learning.verify_outputs
```

The label sidecars are included. To regenerate them, obtain `cars_dataset.csv`
from the Kaggle source linked in the main README, then run:

```bash
python -m parameter_learning.prepare_labels --raw-csv dataset/cars_dataset.csv
```

The extractor verifies all 1,200,000 encoded data cells, their missingness and
the source-row split membership before attaching any label. It refuses a
different source version. See `labels/provenance.json` for the source hash.

## EM interface for A

```python
import json
import pandas as pd
from parameter_learning import fit_em, fit_mle, DiscreteBN
from structure.bn import dag_from_edges

artifact = json.load(open("structure/outputs/dags.json"))
card = artifact["cardinality"]
dag = dag_from_edges(card, artifact["edges"]["expert"])
train = pd.read_csv("dataset/prepared/train.csv", index_col="source_row", dtype="Int64")

# Replace train with A's masked training frame. Preserve all original NAs.
result = fit_em(dag, train, card, alpha=1.0, seed=0, max_iter=500, tol=1e-7)
model = result.model
cpts = result.cpts                 # compatible with B/D's dict-of-arrays
history = result.history          # iteration 0 + every completed M-step
print(result.converged, result.n_iter)

validation = pd.read_csv("dataset/prepared/val.csv", index_col="source_row", dtype="Int64")
ll = model.log_likelihood(validation)  # one marginal log probability per row
model.save("parameter_learning/outputs/my_em.json")
restored = DiscreteBN.load("parameter_learning/outputs/my_em.json")
```

Run seeds 0, 1, 2 and report variability. Settings and initialization selection
use validation likelihood, never test or grade labels. The entry point above
fits one seed deliberately; A controls masks, seeds and E2 comparisons.

A runnable integration example is provided:

```bash
python -m parameter_learning.e2_handoff --rate 0.2 --seed 0
```

It runs the three estimators on one shared MCAR mask. Mask rates mean additional
masking among observed training cells. Mode values are learned from the masked
training data only. It checks recovery only on deliberately hidden entries,
while retaining every other mask. This example is not A's full E2 sweep.

For complete-case MLE:

```python
model = fit_mle(dag, masked_train, card, alpha=1, missing="complete_case")
```

For mode imputation, compute each mode from `masked_train`, fill training NAs,
then call `fit_mle`. Score each method on the same validation/test rows and
evidence. Do not silently drop validation rows for one method or fill them for
another. `model.log_likelihood` integrates missing test entries exactly.

## CPT and posterior conventions for D/E

`dag[node]` is an ordered parent tuple. CPT shape is
`(product(card[parent]), card[node])`. Parent configuration indexing is
lexicographic in that tuple, with the last parent varying fastest. State order
is exactly `dataset/prepared/schema.json`; no new missing-value category exists.

JSON models include the exact ordered parents, cardinalities and CPT arrays.
Observed selected models are in `outputs/models/expert_selected.json` and
`hill_climb_bic_selected.json`. The E3 model is separate in
`latent_H3_selected.json`; it must not silently replace E1's expert DAG.

For imputation, keep the query column NA and call:

```python
posterior = model.posterior(masked_frame, "overall_condition_score")
imputed = posterior.argmax(axis=1)
```

An observed query returns a point mass. To infer H, omit its column and pass
`latent_nodes=("H",)` to `fit_em`, `log_likelihood` or `posterior`. Undeclared
absent columns raise errors, so a spelling mistake cannot become a latent node.

## Estimation and convergence

Add-alpha MLE uses `(count + alpha)/(parent_count + r*alpha)`; default alpha=1
matches B. At alpha=0 an unseen parent row is uniform. EM uses the same formula
with exact expected counts, conditioning jointly on all observed variables.
Missing rows are retained, including their observed components. A fully missing
record carries no observed likelihood information.

With alpha>0 this is MAP EM under independent Dirichlet(1+alpha) priors. The
monotone objective is `observed_LL + alpha * sum(log(CPT entries))`, rather than
the unregularized likelihood alone. Histories and plots show both. At alpha=0
this becomes ordinary maximum-likelihood EM. Iteration zero is initialization;
stopping uses absolute objective change per training row <= 1e-7. Hitting the
iteration cap is reported explicitly by `converged=False`.

EM enumerates only missing states, compresses repeated evidence patterns, and
caches family indices. This is exact, not sampling or an independent-column
approximation. A two-million-completion limit rejects infeasible inputs before
allocation. This supports the current two-column MCAR design plus H, but does
not promise arbitrary high-dimensional masks. Posterior enumeration is a Part C
helper; D's VE remains the scalable inference interface.

## E3 design and interpretation

The screenshot/work split does not prescribe H's edges. `config.json` therefore
records a concrete experimental graph: a three-state root H is a shared parent
of accident history, service history, overall condition score and listing price.
Starting from the expert DAG, service->condition_score is removed to keep the
four-parent cap, and condition_score->price is replaced by H->price. All other
expert edges remain. This is a separate modeling hypothesis with 701 nominal
parameters; it is not an E1 structure-learning result or a causal claim.

Fit three random initializations on all 80,000 training rows, choose by observed
validation LL, then report test LL and NMI. Grade is never an input, initializer,
or selection criterion. H state numbers have no inherent order; NMI is invariant
to their permutation and uses arithmetic-mean entropy normalization.

Because grade is closely related to observed condition score, full-evidence NMI
does not establish independent recovery of a true condition construct. We also
infer H with score hidden, and with both score and price hidden, without
retraining or selecting on these diagnostic results. Report each NMI,
state occupancy, posterior entropy and the 100-permutation null. These are
posterior-evidence sensitivity checks, not a score-free training ablation or
out-of-sample price prediction. E's prediction experiment must hide price.

See `REPORT.md` for measured results and report-ready text.
