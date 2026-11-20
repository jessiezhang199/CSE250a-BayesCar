# CSE250a-BayesCar

CSE 250A final project by the BayesCar group: Kevin Nie, Chloe Huang, Chenghao Xiong, Yijun Wang and Jessie Zhang.

We model vehicle listing price tiers with a discrete Bayesian network on the Kaggle [Cars Dataset (2026)](https://www.kaggle.com/datasets/datascikhan/cars-dataset), which has 100,000 synthetic records. The project asks four things: whether a learned graph fits better than a hand-built one, how parameter learning holds up when data is missing, how approximate inference compares with exact inference, and how well the networks predict the price tier. Because the data is synthetic, the results say nothing about real car prices or causal effects.

Who does what is in [WORK_SPLIT.md](WORK_SPLIT.md).

## Experiments

| Experiment | Question | Headline result | Part | Code | Write-up |
| --- | --- | --- | --- | --- | --- |
| [E1](#e1--structure-learning) | Does a learned DAG fit better than the expert DAG? | Yes, by 0.14 nats per test record | B | [structure/](structure/) | [README](structure/README.md) |
| [E2](#e2--missing-data) | How do deletion, imputation and EM compare under missing data? | EM is best at every masking rate, mode imputation degrades fastest | A | [complete-case_MLE_and_EM/](complete-case_MLE_and_EM/) | [MLE_EM.ipynb](complete-case_MLE_and_EM/MLE_EM.ipynb) |
| [E3](#e3--latent-variable) | Does a latent factor H recover vehicle condition? | No, H barely tracks condition grade and does not improve fit | C | [parameter_learning/](parameter_learning/) | [REPORT.md](parameter_learning/REPORT.md) |
| [E4](#e4--exact-and-approximate-inference) | How do sampling methods compare with exact VE? | Likelihood weighting is the most accurate sampler, rejection sampling fails on rare evidence | D | [inference/](inference/) | [REPORT.md](inference/REPORT.md) |
| [E5](#e5--price-tier-prediction) | How well do the networks predict the price tier? | Learned DAG 67.1%, expert DAG 63.9%, chance 20% | E | [prediction/](prediction/) | [REPORT.md](prediction/REPORT.md) |

## How the parts fit together

Each part reads what the earlier ones wrote, so the same splits, graphs and fitted models are used everywhere.

1. Part A prepares the data in `dataset/prepared/`.
2. Part B writes the expert and learned graphs to `structure/outputs/dags.json`.
3. Part C fits CPTs for both graphs and saves them in `parameter_learning/outputs/models/`.
4. Part D provides exact and approximate posterior queries on those models.
5. Part E uses Part D's exact queries to predict the price tier, and Part A's E2 notebook uses Part C's EM.

## Reproduce

Use Python 3.11 or newer and run everything from the repository root. `prediction/requirements.txt` pulls in the Part C and Part D requirements, so one install is enough.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r prediction/requirements.txt
python -m pytest -q parameter_learning/tests inference/tests prediction/tests
```

Run the experiments in this order.

```bash
python structure/run_e1.py                # E1, writes structure/outputs/
python -m parameter_learning.run_part_c   # CPTs, EM and E3, writes parameter_learning/outputs/
python -m inference.check_setup           # checks VE against pgmpy on both graphs
python -m inference.run_e4                # E4, writes inference/outputs/
python -m prediction.run_e5               # E5, writes prediction/outputs/
```

E2 lives in the notebook [MLE_EM.ipynb](complete-case_MLE_and_EM/MLE_EM.ipynb). It finds the repository root on its own, so it runs from any folder inside the checkout.

Parts C, D and E record the package versions behind their committed outputs in `outputs/manifest.json` or `outputs/settings.json`. Timings will differ on your machine, and the logistic regression baseline in E5 can move in the fourth decimal with a different scikit-learn version.

## Repository layout

| Path | Part | Contents |
| --- | --- | --- |
| [dataset/](dataset/) | A | Cleaning notebook, prepared `train.csv`, `val.csv`, `test.csv`, `schema.json` and `column_audit.csv` |
| [complete-case_MLE_and_EM/](complete-case_MLE_and_EM/) | A | E2 notebook comparing complete-case MLE, mode imputation and EM under MCAR masks |
| [structure/](structure/) | B | BIC score, hill climbing, Chow-Liu, CI tests and the shared `outputs/dags.json` |
| [parameter_learning/](parameter_learning/) | C | Smoothed MLE, exact missing-data EM, fitted models and E3 |
| [inference/](inference/) | D | `query_ve`, likelihood weighting, Gibbs and rejection sampling, E4 |
| [prediction/](prediction/) | E | Price-tier prediction, baselines, metrics and E5 |

## Data and shared conventions

- The data has 12 discrete variables and an 80/10/10 split (seed 42). Bins and category codes come from the training rows only.
- Load a split with `pd.read_csv(path, index_col="source_row", dtype="Int64")` and take state order from `dataset/prepared/schema.json`.
- Blanks are real missing values in `service_history_quality` (about 4%) and `overall_condition_score` (about 1%).
- We dropped `days_on_market` and `demand_category` because nobody knows them until after the car is listed.
- Every network has at most 4 parents per node and smoothed CPTs. Validation data picks settings and seeds, and the test split is only used for final numbers.
- Stochastic experiments use three seeds and report the spread.

## E1 · Structure learning

```bash
python structure/run_e1.py
```

Part B ([guide and results](structure/README.md)) writes an expert DAG by hand and learns a second one with BIC hill climbing from 12 starting graphs. Both are compared with a Chow-Liu tree, naive Bayes and the empty graph on the same complete-case rows. The learned DAG reaches -10.781 nats per test record against -10.919 for the expert DAG, a paired gain of 0.138 ± 0.005, while the expert DAG uses 240 fewer parameters. The two graphs have 11 edges in common, 5 of them with the direction reversed. Two variables, `service_history_quality` and `urban_rural`, turn out to be independent of everything else in this dataset, so the learned graph leaves them isolated.

## E2 · Missing data

Part A's [notebook](complete-case_MLE_and_EM/MLE_EM.ipynb) hides an extra 0%, 20% or 40% of the observed `service_history_quality` and `overall_condition_score` cells in the training split, completely at random. It fits the expert DAG on each masked copy with complete-case MLE, with mode imputation followed by MLE, and with Part C's EM. The three methods see the same mask, and each rate is repeated with three masking seeds.

| Added masking | Complete-case MLE | Mode imputation + MLE | EM | Complete-case training rows kept |
| --- | ---: | ---: | ---: | ---: |
| 0% | -10.86407 | -10.86682 | -10.86374 | 76,081 / 80,000 (95.1%) |
| 20% | -10.86595 | -10.99005 | -10.86431 | 48,699 / 80,000 (60.9%) |
| 40% | -10.87056 | -11.22263 | -10.86666 | 27,441 / 80,000 (34.3%) |

Values are test log-likelihood in nats per row, averaged over the three seeds. EM comes out on top at every rate and beats complete-case MLE on both test log-likelihood and CPT KL in all six masked runs, though complete-case MLE stays close even when it keeps only a third of the rows. Mode imputation gets much worse as more cells are hidden. The notebook also has condition-score recovery accuracy and per-seed results.

## E3 · Latent variable

```bash
python -m parameter_learning.run_part_c
```

Part C ([guide](parameter_learning/README.md), [report](parameter_learning/REPORT.md)) fits smoothed MLE and exact missing-data EM for both graphs. It then adds a three-state latent root H as a shared parent of accident history, service history, condition score and price. `condition_grade` never enters training and is only used afterwards to measure how well H lines up with it. All nine EM fits converge. The latent graph scores slightly worse on held-out data than the expert EM fit (-0.020 ± 0.002 nats per row). H also has little to do with condition grade, with a test NMI of 0.115 for the selected seed compared with 0.719 for the observed condition score.

## E4 · Exact and approximate inference

```bash
python -m inference.run_e4
```

Part D ([guide](inference/README.md), [report](inference/REPORT.md)) implements variable elimination as `inference.query_ve`. It matches pgmpy to within `1e-10` on both graphs and returns the exact price posterior in about 0.1 ms. On the expert network it is compared with likelihood weighting, Gibbs and rejection sampling. At 10,000 draws likelihood weighting has the smallest error, with mean TV 0.006 under common evidence and 0.010 under rare evidence. Rejection sampling falls apart under rare evidence (TV 0.15).

## E5 · Price-tier prediction

```bash
python -m prediction.run_e5
```

Part E ([guide](prediction/README.md), [report](prediction/REPORT.md)) predicts the price tier of every test row with exact VE. It uses the expert DAG, the learned DAG, Chow-Liu, naive Bayes and the prior, plus a logistic regression baseline, and runs each one with all features and again without the condition and history fields. With all features the learned DAG gets 67.1% right and the expert DAG 63.9%, where chance is 20%. Logistic regression lands at 67.9%, about the same as the learned DAG. The expert DAG has the best calibrated probabilities.
