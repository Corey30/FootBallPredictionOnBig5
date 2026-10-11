# Football Prediction on Big 5 Leagues

An end-to-end football prediction research project for Europe?s Big Five leagues: England, Spain, Germany, Italy, and France. The project combines structured match-level modelling with event-derived first-half momentum analysis to study football result prediction under chronological evaluation.

> This project is for research and engineering purposes. Its outputs are statistical estimates, not betting advice or guarantees of future results.

## Project overview

The project is organized into two connected components:

1. **Football Prediction on Big 5 Leagues** (`FootBallPredictionOnBig5/`)
   - General data loading, cleaning, feature engineering, training, evaluation, and reporting.
   - Supports pre-match and half-time prediction.
   - Provides reusable model, metric, feature-selection, imbalance-handling, and momentum utilities.

2. **Football Real-Time Momentum Prediction** (`FootBallRealPre/`)
   - Event-level analysis based on StatsBomb data.
   - Builds first-half event-state tables and momentum-curve features.
   - Runs controlled CatBoost and Optuna experiments comparing baseline, half-time, momentum, and combined feature sets.

The first component provides the broader modelling framework. The second component focuses on event-derived momentum features and controlled experiments that can be used to extend or validate the broader framework.

## Research workflow

```text
Historical match and event data
            ?
            ??? FootBallRealPre
            ?   ??? StatsBomb match/event ingestion
            ?   ??? Event-state construction
            ?   ??? First-half momentum curves
            ?   ??? Momentum feature tables
            ?
            ??? FootBallPredictionOnBig5
                ??? Data cleaning and feature preparation
                ??? Pre-match and half-time feature construction
                ??? Time-aware train/validation/test splits
                ??? Model training and tuning
                ??? Evaluation and visual reports
                ??? Reproducible experiment outputs
```

Both components use the three-class match-result target:

- `H`: home win
- `D`: draw
- `A`: away win

## Repository structure

```text
.
??? README.md
??? FootBallPredictionOnBig5/
?   ??? README.md
?   ??? requirements.txt
?   ??? .gitignore
?   ??? data/
?   ??? merged_data/
?   ??? src/
?   ?   ??? config/
?   ?   ??? data/
?   ?   ??? evaluation/
?   ?   ??? features/
?   ?   ??? momentum/
?   ?   ??? training/
?   ?   ??? utils/
?   ??? scripts/
?   ??? outputs/
?
??? FootBallRealPre/
    ??? README.md
    ??? requirements.txt
    ??? .gitignore
    ??? FootBallRealPre/
        ??? FootBallRealTimePrediction/
            ??? build_momentum_features.py
            ??? run_momentum_prediction.py
            ??? tune_england_pre_momentum.py
            ??? 15-16_official_data/
            ??? real_time_data/
            ??? sb_events/
            ??? src/
```

The nested `FootBallRealPre/FootBallRealPre/FootBallRealTimePrediction/` path is the current runnable code directory for the event-level component.

## Installation

Use separate virtual environments for the two components because they have different modelling and data-access dependencies.

### General modelling component

```bash
cd FootBallPredictionOnBig5
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Event-level momentum component

```bash
cd FootBallRealPre/FootBallRealPre/FootBallRealTimePrediction
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Python 3.10?3.13 is recommended. Some third-party packages, especially data-access or machine-learning packages, may require a platform-specific compatible version.

## Quick start

### Train pre-match or half-time models

```bash
cd FootBallPredictionOnBig5

python scripts/run.py --task prematch --league England
python scripts/run.py --task halftime --league England
python scripts/run.py --task prematch --league England --version selected
```

The default league files are expected under:

```text
FootBallPredictionOnBig5/merged_data/feature_engineer5/
??? England.csv
??? France.csv
??? Germany.csv
??? Italy.csv
??? Spain.csv
```

Generated models and reports are written to `outputs/` unless another output directory is supplied.

### Build momentum features

```bash
cd FootBallRealPre/FootBallRealPre/FootBallRealTimePrediction
python build_momentum_features.py
```

This reads the first-half event-state table and writes the momentum feature table under `real_time_data/`.

### Run the momentum comparison experiment

```bash
python run_momentum_prediction.py
```

The experiment compares:

- `pre_baseline`: pre-match features only
- `pre_momentum`: pre-match plus momentum features
- `ht_baseline`: pre-match plus half-time features
- `ht_momentum`: pre-match, half-time, and momentum features

Results are written to `momentum_prediction_results/`.

### Tune the England momentum model

```bash
python tune_england_pre_momentum.py
```

This runs the focused England pre-match momentum experiment using CatBoost and Optuna.

## Main capabilities

### Match-level modelling

- Pre-match and half-time prediction.
- Time-ordered train, validation, and test splitting.
- Random Forest, Logistic Regression, XGBoost, SVM, Naive Bayes, feed-forward neural network, Vanilla RNN, voting, and stacking models.
- Feature selection using variance, correlation, and model importance.
- Class-imbalance analysis, weighting, threshold adjustment, and resampling.
- Accuracy, log loss, precision, recall, F1, ROC-AUC, confusion matrices, and comparison reports.
- Grid and randomized hyperparameter tuning.

### Event-level momentum analysis

- StatsBomb match and event ingestion.
- Event-state and form-table construction.
- First-half directional and intensity momentum curves.
- Uniform-grid curve resampling and summary statistics.
- Team-name normalization and match-level feature merging.
- CatBoost classification with Optuna hyperparameter search.
- Controlled ablation experiments to measure the contribution of momentum features.

## Data and licensing

The project contains historical football tables, event-derived files, and generated experiment artifacts. Before publishing or redistributing data, verify the terms of the original data providers, including StatsBomb attribution and redistribution requirements.

Do not commit private data, credentials, API keys, or data that you are not authorized to redistribute. Generated outputs, caches, local environments, and logs should remain untracked according to the component `.gitignore` files.

## Reproducibility guidance

- Record the Python version and installed package versions.
- Keep the input data version and feature-engineering stage with each experiment.
- Preserve chronological train/validation/test splits.
- Avoid using post-match or second-half information in first-half prediction tasks.
- Fit preprocessing and feature selection using training data only when extending the pipeline.
- Report macro-F1, per-class recall, log loss, and ROC-AUC alongside accuracy.
- Inspect the draw class separately because class imbalance can hide poor minority-class performance.
- Confirm match-key coverage after merging official and event-derived tables.

## Development checks

Run these checks before opening a pull request:

```bash
cd FootBallPredictionOnBig5
python -m compileall src scripts

cd ../FootBallRealPre/FootBallRealPre/FootBallRealTimePrediction
python -m compileall src build_momentum_features.py run_momentum_prediction.py tune_england_pre_momentum.py
```

Keep command-line interfaces, input paths, feature definitions, and dependency files documented when changing the code.

## Component documentation

- [General modelling component](FootBallPredictionOnBig5/README.md)
- [Event-level momentum component](FootBallRealPre/README.md)
- [Runnable event-level project directory](FootBallRealPre/FootBallRealPre/FootBallRealTimePrediction/README.md)

## License and citation

No software license is currently declared for this project. Add a license file before distributing the code or accepting external contributions. Add the required dataset citations and provider-specific attribution before publishing derived or raw data.

## Contact and issue reports

When opening an issue, include the operating system, Python version, component path, command, input-data version, and full traceback. Never include private datasets, credentials, or restricted event files in an issue.
