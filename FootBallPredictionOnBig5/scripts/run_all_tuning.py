"""
Orchestrator: tune all (model, league, task) combinations.

Default: 6 models x 5 leagues x 2 tasks = 60 jobs.
Skips jobs whose best_params.json already exists, so the script is
resumable.

Usage:
    python scripts/run_all_tuning.py
    python scripts/run_all_tuning.py --models xgboost,random_forest
    python scripts/run_all_tuning.py --leagues England,Spain
    python scripts/run_all_tuning.py --search random --n-iter 60
"""

import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


ROOT = Path(__file__).resolve().parent.parent
TUNER = ROOT / 'scripts' / 'tune_hyperparameters.py'
OUT_DIR = ROOT / 'outputs' / 'reports' / 'hyperparameter_tuning'

ALL_MODELS = ['logistic_regression', 'xgboost', 'random_forest',
              'svm', 'naive_bayes', 'fnn']
ALL_LEAGUES = ['England', 'Spain', 'Germany', 'Italy', 'France']
ALL_TASKS = ['prematch', 'halftime']

# Per-model n_iter for RandomizedSearchCV.  Slower models get fewer
# samples because each fit costs more.  Naive Bayes always uses grid
# (there are only 5x6=30 candidates), so its value is irrelevant.
DEFAULT_N_ITER = {
    'logistic_regression': 40,
    'xgboost':              40,
    'random_forest':        40,
    'svm':                  60,
    'naive_bayes':         300,   # large enough to enumerate full grid
    'fnn':                  40,
}

DEFAULT_SEARCH = {
    'logistic_regression': 'grid',  # full grid is small (12 combos x 6 samp = 72)
    'naive_bayes': 'grid',           # also small
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default=','.join(ALL_MODELS),
                        help='Comma-separated subset of models.')
    parser.add_argument('--leagues', default=','.join(ALL_LEAGUES),
                        help='Comma-separated subset of leagues.')
    parser.add_argument('--tasks', default=','.join(ALL_TASKS),
                        help='Comma-separated subset of tasks.')
    parser.add_argument('--search', default='random',
                        choices=['grid', 'random'])
    parser.add_argument('--n-iter', type=int, default=60)
    parser.add_argument('--cv-splits', type=int, default=5)
    parser.add_argument('--no-skip', action='store_true',
                        help='Force re-run even if best_params.json exists.')
    parser.add_argument('--python', default=sys.executable)
    args = parser.parse_args()

    models = [s.strip() for s in args.models.split(',') if s.strip()]
    leagues = [s.strip() for s in args.leagues.split(',') if s.strip()]
    tasks = [s.strip() for s in args.tasks.split(',') if s.strip()]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    jobs = [(m, lg, t) for m in models for lg in leagues for t in tasks]
    n_total = len(jobs)
    print(f"\nLaunching {n_total} tuning jobs "
          f"({len(models)} models x {len(leagues)} leagues x {len(tasks)} tasks)")
    print(f"Output dir: {OUT_DIR}")
    print(f"Search    : {args.search} (n_iter={args.n_iter})")
    print(f"CV splits : {args.cv_splits}")

    completed = 0
    skipped = 0
    failed = []
    overall_t0 = datetime.now()

    for idx, (model, league, task) in enumerate(jobs, start=1):
        stem = f"{league.lower()}_{task}_{model}"
        best_path = OUT_DIR / f"{stem}__best_params.json"
        if best_path.exists() and not args.no_skip:
            print(f"\n[{idx}/{n_total}] SKIP {stem} (already done)")
            skipped += 1
            continue

        cmd = [
            args.python, str(TUNER),
            '--model', model,
            '--task', task,
            '--league', league,
            '--search', DEFAULT_SEARCH.get(model, args.search),
            '--n-iter', str(DEFAULT_N_ITER.get(model, args.n_iter)),
            '--cv-splits', str(args.cv_splits),
        ]

        print(f"\n[{idx}/{n_total}] >>> {stem}")
        print('  cmd:', ' '.join(cmd))
        t0 = datetime.now()
        try:
            subprocess.run(cmd, check=True, cwd=str(ROOT))
            dt = (datetime.now() - t0).total_seconds()
            completed += 1
            print(f"  done in {dt:.1f}s")
        except subprocess.CalledProcessError as exc:
            failed.append(stem)
            print(f"  FAILED: {exc}")

    overall_dt = (datetime.now() - overall_t0).total_seconds()
    print("\n" + "=" * 70)
    print(f"Total elapsed    : {overall_dt/60:.1f} min")
    print(f"Completed        : {completed}/{n_total}")
    print(f"Skipped (cached) : {skipped}/{n_total}")
    print(f"Failed           : {len(failed)}")
    for f in failed:
        print(f"   - {f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
