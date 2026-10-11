# Football Prediction on Big 5 Leagues

A reproducible Python pipeline for football match-result prediction across Europe's Big Five leagues: England, Spain, Germany, Italy, and France. The project supports pre-match and half-time prediction, time-aware validation, feature selection, model comparison, hyperparameter tuning, class-imbalance handling, and momentum-curve analysis.

> This repository is an engineering and research project. Predictions are statistical estimates, not betting advice or guarantees of future results.

## Features

- Pre-match and half-time three-class prediction: home win (`H`), draw (`D`), away win (`A`).
- Time-ordered train/validation/test splitting to reduce temporal leakage.
- Historical, ranking, odds, performance, rest/fatigue, half-time, and momentum features.
- Random Forest, Logistic Regression, XGBoost, SVM, Naive Bayes, feed-forward neural network, Vanilla RNN, voting, and stacking models.
- Feature selection based on variance, correlation, and model importance.
- Accuracy, log loss, precision, recall, F1, ROC-AUC, confusion matrices, and comparison reports.
- Random/grid hyperparameter search with imbalance samplers and scalers.
- Momentum-equation and causal-curve utilities for event-level analysis.

## Repository layout

```text
.
├── data/                 # Data cleaning and exploratory analysis code
├── merged_data/          # Prepared feature tables and intermediate data
├── src/
│   ├── config/           # Paths, labels, model defaults, feature groups
│   ├── data/              # Loading, cleaning, and time-based splitting
│   ├── evaluation/       # Metrics and visual reports
│   ├── features/         # Pre-match, half-time, and feature selection logic
│   ├── momentum/         # Momentum equations and curve features
│   ├── training/         # Trainers, ensembles, and imbalance handling
│   └── utils/             # Shared helpers
├── scripts/              # Command-line entry points
├── outputs/              # Generated models, reports, and tuning results
├── requirements.txt
└── README.md
```

Generated files under `outputs/`, Python caches, local environments, and editor files should not be committed.

## Requirements

- Python 3.10–3.13 recommended.
- A 64-bit operating system.
- Enough disk space for feature tables and generated reports.
- TensorFlow is required only for the Vanilla RNN model and is included in `requirements.txt`; on platforms where TensorFlow wheels are unavailable, use the other models or install a compatible TensorFlow version separately.

## Installation

From the repository root:

```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run all commands from the directory that contains `scripts/`, `src/`, and `merged_data/`.

## Data preparation

The default configuration reads:

```text
merged_data/feature_engineer5/England.csv
```

The supplied league tables follow the same layout and are expected at:

```text
merged_data/feature_engineer5/{England,Spain,Germany,Italy,France}.csv
```

To use another CSV, pass `--input-csv` (or the legacy `--data-path`) to `scripts/run.py`, or `--data-path` to the dedicated training scripts. A custom table must contain the target column `target_match_result`; the cleaning and feature modules also recognize the target columns listed in `src/config/config.py`. The labels are `H`, `D`, and `A`.

Do not commit private, licensed, or personally identifiable source data without permission. Check the license and redistribution terms of any football, odds, or event dataset before publishing it.

## Quick start

Unified entry point:

```bash
# Pre-match prediction using the selected league file
python scripts/run.py --task prematch --league England

# Half-time prediction
python scripts/run.py --task halftime --league England

# Enable variance/correlation feature selection
python scripts/run.py --task prematch --league England --version selected

# Choose models and an output directory
python scripts/run.py --task prematch --league Spain \
  --models random_forest,logistic_regression,xgboost,svm \
  --output-dir outputs/spain_run
```

The command writes trained models to `<output-dir>/models` and evaluation reports to `<output-dir>/reports`. If `--output-dir` is omitted, the project uses `outputs/`.

Dedicated entry points are also available:

```bash
python scripts/run_prematch.py --help
python scripts/run_halftime.py --help
```

Use `--help` on every script to see the complete and current argument list.

## Hyperparameter tuning

Tune one model/task/league:

```bash
python scripts/tune_hyperparameters.py \
  --model xgboost \
  --task prematch \
  --league England \
  --search random \
  --n-iter 60
```

Run the configured model/league/task combinations:

```bash
python scripts/run_all_tuning.py \
  --models logistic_regression,xgboost,random_forest \
  --leagues England,Spain \
  --tasks prematch,halftime \
  --search random \
  --n-iter 60
```

Aggregate tuning outputs and create a summary:

```bash
python scripts/aggregate_tuning_results.py --help
python scripts/plot_tuning_summary.py --help
```

Tuning can be computationally expensive. Reduce `--n-iter`, `--cv-splits`, or the number of leagues/models when testing locally.

## Analysis and momentum tools

```bash
# Feature importance analysis
python scripts/analyze_feature_importance.py --task prematch --league England

# Plot momentum curves from prepared event data
python scripts/plot_momentum_curves.py --help

# Compare half-time and pre-match curve features for the 2015–16 experiment
python scripts/run_halftime_curve_compare_1516.py --help
python scripts/run_prematch_curve_compare_1516.py --help
python scripts/run_next_goal__1516.py --help
```

These experiment scripts expect the event/curve data paths configured in `src/momentum/config.py`. Adjust those paths before running them on a new dataset.

## Reproducibility and evaluation notes

- The default random seed is defined in `src/config/config.py`.
- Splits are time-aware; do not randomly shuffle matches when comparing against the project defaults.
- Fit preprocessing and feature selection on training data only when extending the pipeline.
- Report more than accuracy: inspect macro-F1, per-class recall, log loss, ROC-AUC, and the confusion matrix, especially for the draw class.
- Results depend on the input data version, available seasons, feature-engineering stage, and installed library versions.

## Development checks

Before opening a pull request, run:

```bash
python -m compileall src scripts
python scripts/run.py --help
```

Keep generated artifacts out of commits, use clear module/function names, and update this README when command-line interfaces or data paths change.

## License and citation

No license is declared yet. Add a license file before distributing the repository or accepting external contributions. If this project or its feature-engineering ideas support published work, add the appropriate citation here.

## Contact

Open a GitHub issue with the operating system, Python version, command, and full traceback when reporting a reproducible problem. Do not upload private datasets or credentials.
