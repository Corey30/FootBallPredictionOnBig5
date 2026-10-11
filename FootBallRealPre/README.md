# Football Real-Time Momentum Prediction

A focused research pipeline for evaluating whether first-half, event-derived momentum features improve football match-result prediction across Europe's Big Five leagues: England, Spain, Germany, Italy, and France.

This repository is the event-level companion to **[Football Prediction on Big 5 Leagues](../FootBallPredictionOnBig5/README.md)**. The companion repository provides the general pre-match/half-time modelling pipeline and reusable model/evaluation modules; this repository builds first-half momentum features from StatsBomb event data and runs controlled CatBoost experiments that compare baseline, half-time, momentum, and combined feature sets.

> This is a research and engineering project. Predictions are statistical estimates, not betting advice or guarantees of future results.

## How the two repositories are related

| Repository | Main responsibility | Typical output |
|---|---|---|
| `Football Prediction on Big 5 Leagues` (`FootBallPredictionOnBig5`) | General data preparation, feature engineering, time-aware modelling, evaluation, tuning, and reporting | Trained models, metrics, feature-importance reports, and reusable momentum utilities |
| `Football Real-Time Momentum Prediction` (`FootBallRealPre`) | StatsBomb event ingestion, first-half event-state construction, momentum-curve summarisation, and controlled CatBoost experiments | Momentum feature CSVs and `pre_baseline` / `pre_momentum` / `ht_baseline` / `ht_momentum` comparison results |

The projects share the same football domain, league coverage, match-result labels (`H`, `D`, `A`), time-aware evaluation principle, and momentum-feature concept. They are separate runnable projects: this repository does not import the other repository at runtime. To reproduce the full research workflow, generate or validate event-derived momentum features here, then use the resulting feature tables in the broader modelling workflow from the companion repository when appropriate.

## What this project does

- Downloads or processes StatsBomb match and event data for the 2015?16 Big Five league experiment.
- Builds event-state tables and first-half momentum curves.
- Computes two curves for each match:
  - `M_eq(t)`: directional momentum, representing which team is controlling the match.
  - `M_int(t)`: intensity momentum, representing how active the match is.
- Resamples each curve on a 91-point first-half grid.
- Extracts 10 summary features per curve, producing 20 momentum features per match.
- Merges momentum features with official match tables using normalized team-name keys.
- Trains CatBoost classifiers with Optuna tuning.
- Compares four controlled conditions:
  - `pre_baseline`: pre-match features only.
  - `pre_momentum`: pre-match plus momentum features.
  - `ht_baseline`: pre-match plus half-time features.
  - `ht_momentum`: pre-match, half-time, and momentum features.
- Uses chronological 60/20/20 train, validation, and test splits.

## Repository layout

```text
FootBallRealPre/
??? README.md
??? requirements.txt
??? .gitignore
??? FootBallRealPre/
    ??? FootBallRealTimePrediction/
        ??? build_momentum_features.py
        ??? run_momentum_prediction.py
        ??? tune_england_pre_momentum.py
        ??? requirements.txt
        ??? 15-16_official_data/       # Official match tables
        ??? real_time_data/             # Event-state and momentum feature tables
        ??? sb_events/                  # StatsBomb event parquet files
        ??? src/
        ?   ??? Fecth_Match.py         # Fetch match metadata
        ?   ??? Fetch_Event.py         # Fetch event data
        ?   ??? form_table.py           # Build event/form tables
        ?   ??? patch.py               # Data repair/patch utilities
        ??? momentum_prediction_results/
```

The nested folder is retained to match the current project layout. When publishing, you may choose either the outer folder as the repository root or publish `FootBallRealPre/FootBallRealTimePrediction` as the repository root.

## Requirements

- Python 3.10?3.13 recommended.
- Internet access is required only for the StatsBomb fetch scripts.
- Sufficient disk space for parquet event files and generated tables.
- A compatible CatBoost installation for model training.

## Installation

From the directory containing `requirements.txt` and `run_momentum_prediction.py`:

```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you cloned the outer repository, enter the runnable project directory first:

```bash
cd FootBallRealPre/FootBallRealPre/FootBallRealTimePrediction
```

## Data workflow

The default scripts use paths relative to the directory containing the scripts:

```text
15-16_official_data/England.csv
15-16_official_data/France.csv
15-16_official_data/Germany.csv
15-16_official_data/Italy.csv
15-16_official_data/Spain.csv
real_time_data/event_state_table_big5_2015_16.csv
real_time_data/momentum_features_big5_2015_16.csv
```

The repository already contains historical experiment data and generated artifacts. To rebuild the event-derived features:

1. Fetch match metadata with `src/Fecth_Match.py` when a fresh StatsBomb source is required.
2. Fetch events with `src/Fetch_Event.py`.
3. Build form/event-state tables with `src/form_table.py` or `src/patch.py`, depending on the input table that needs repair.
4. Build first-half momentum features:

```bash
python build_momentum_features.py
```

The builder reads `real_time_data/event_state_table_big5_2015_16.csv` and writes `real_time_data/momentum_features_big5_2015_16.csv`.

StatsBomb data is subject to its own terms of use and attribution requirements. Verify the source license and redistribution conditions before publishing raw event files or derived data.

## Run the controlled prediction experiment

After the official league CSV files and momentum feature CSV are available:

```bash
python run_momentum_prediction.py
```

The script loads all five leagues, merges normalized match keys, tunes CatBoost with Optuna, evaluates the four experimental conditions, and writes:

```text
momentum_prediction_results/momentum_catboost_results.csv
```

Use the script's constants near the top of the file to change input/output locations, feature groups, or the experimental conditions. The default Optuna search can be computationally expensive.

## Focused England tuning

For the England pre-match momentum experiment:

```bash
python tune_england_pre_momentum.py
```

This script uses the England official table and the shared momentum feature table, performs a chronological split, tunes CatBoost with Optuna, and reports validation/test metrics.

## Evaluation and interpretation

The experiment reports accuracy, macro-F1, log loss, and multiclass ROC-AUC where the required class probabilities are available. Compare conditions on the same chronological test period. A higher score in one split does not establish general performance across seasons or leagues.

When extending the project:

- Keep feature construction causal: do not use post-match or second-half information in a first-half prediction.
- Keep train/validation/test chronology intact.
- Check unmatched team keys after merging official and momentum tables.
- Record the input-data version, Optuna seed, feature set, and package versions.
- Inspect the draw class separately; overall accuracy can hide class imbalance.

## Relationship to the companion repository in practice

A typical workflow is:

```text
StatsBomb events
    -> FootBallRealPre: event-state tables and momentum features
    -> Football Prediction on Big 5 Leagues: reusable modelling/evaluation workflow
    -> reports, comparisons, and downstream experiments
```

The companion repository contains a separate momentum module and broader pre-match/half-time training scripts. These implementations are conceptually related, but their input schemas and experiment scripts are not guaranteed to be drop-in compatible. Match the column names and target definitions before combining outputs.

## Development checks

Before opening a pull request:

```bash
python -m compileall src build_momentum_features.py run_momentum_prediction.py tune_england_pre_momentum.py
```

Keep generated outputs, caches, local environments, and credentials out of commits. Update this README when input paths, feature definitions, or experiment commands change.

## License and citation

No software license is currently declared. Add a license file before distributing the code or accepting external contributions. Add the appropriate StatsBomb attribution and dataset citation required by the data provider.

## Contact

For a reproducible issue, include the operating system, Python version, command, input-table version, and full traceback. Do not upload private data, API credentials, or restricted event files.
