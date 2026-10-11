"""
Feature Importance Analysis Script

Supports:
- Halftime prediction and Pre-match prediction tasks
- Multiple leagues selection
- High-quality visualizations for publication
- Cross-league feature importance summary

Usage:
    python analyze_feature_importance.py --task halftime --league all
    python analyze_feature_importance.py --task prematch --league England
    python analyze_feature_importance.py --task halftime --league England,Spain,Germany
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import defaultdict
from src.data.loader import load_data, preprocess_data, clean_data
from src.data.splitter import time_series_split
from src.features.halftime_features import process_halftime_features
from src.features.prematch_features import process_prematch_features
from src.training.trainer import Trainer
from src.config.config import MODEL_PARAMS, MODEL_DISPLAY_NAMES


PUBLICATION_DPI = 300
FONT_FAMILY = 'Times New Roman'
FONT_SIZE_TITLE = 12
FONT_SIZE_LABEL = 10
FONT_SIZE_TICK = 8


LEAGUE_FILE_MAPPING = {
    'England': 'ENG-premier.csv',
    'Spain': 'span.csv',
    'Germany': 'Germany.csv',
    'Italy': 'Italy.csv',
    'France': 'France.csv',
}


def categorize_feature(feature_name: str) -> str:
    """
    Categorize feature into meaningful groups for analysis.
    
    Categories:
    - Odds/Betting: Bookmaker odds and implied probabilities
    - H2H: Head-to-head historical records
    - Team Form: Current streak, unbeaten/run records
    - Ranking: League standings and points
    - Venue History: Home/away specific performance
    - ELO/Rating: Team rating systems
    - Performance: Goals, xG, form points
    - Travel/Fatigue: Rest days, match density
    - Halftime Info: Half-time specific data
    - Pre-match Stats: Pre-match predictions
    - Other: Uncategorized features
    """
    f = feature_name.lower()
    
    if any(kw in f for kw in ['odds_', 'implied_prob', 'overround', 'odds_move']):
        return 'Odds/Betting'
    elif any(kw in f for kw in ['h2h_', 'head_to_head']):
        return 'H2H History'
    elif any(kw in f for kw in ['streak', 'unbeaten', 'winless']):
        return 'Team Form'
    elif any(kw in f for kw in ['rank', 'points_before', 'points_diff']):
        return 'Ranking'
    elif any(kw in f for kw in ['venue', 'home_win', 'away_win', 'home_goal', 'away_goal', 
                                 'home_loss', 'away_loss', 'home_draw', 'away_draw',
                                 'total_home', 'total_away', 'wins_in_the_window']):
        if 'rate' in f or 'per_match' in f or 'margin' in f:
            return 'Venue History'
        return 'Venue History'
    elif any(kw in f for kw in ['elo', 'rating']):
        return 'ELO/Rating'
    elif any(kw in f for kw in ['form_pts', 'xg_', 'goal_diff', 'mom_', 'opp_adj', 
                                 'ppg_diff', 'xg_power']):
        return 'Performance'
    elif any(kw in f for kw in ['travel', 'fatigue', 'match_density', 'rest_days',
                                 'home_recent', 'away_recent']):
        return 'Travel/Fatigue'
    elif any(kw in f for kw in ['half_time', 'ht_result', 'ht_', 'total_goals_at_half']):
        return 'Halftime Info'
    elif any(kw in f for kw in ['ppg', 'average_goals', 'over_', 'under_', 'btts',
                                 'pre_match', 'average_corners', 'average_cards']):
        return 'Pre-match Stats'
    else:
        return 'Other'


def get_feature_display_name(feature_name: str) -> str:
    """Convert feature name to publication-friendly display name."""
    name = feature_name.replace('_', ' ')
    name = name.replace('home ', 'Home ').replace('away ', 'Away ')
    name = name.replace(' h2h ', ' H2H ')
    name = name.replace('xg', 'xG')
    name = name.replace('ppg', 'PPG')
    name = name.replace('elo', 'ELO')
    return name


def setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': [FONT_FAMILY],
        'font.size': FONT_SIZE_TICK,
        'axes.titlesize': FONT_SIZE_TITLE,
        'axes.labelsize': FONT_SIZE_LABEL,
        'xtick.labelsize': FONT_SIZE_TICK,
        'ytick.labelsize': FONT_SIZE_TICK,
        'legend.fontsize': FONT_SIZE_TICK,
        'figure.dpi': PUBLICATION_DPI,
        'savefig.dpi': PUBLICATION_DPI,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.axisbelow': True,
    })


def load_league_data(league: str, data_dir: Path) -> pd.DataFrame:
    """Load data for a specific league."""
    if league not in LEAGUE_FILE_MAPPING:
        raise ValueError(f"Unknown league: {league}. Available: {list(LEAGUE_FILE_MAPPING.keys())}")
    
    file_path = data_dir / LEAGUE_FILE_MAPPING[league]
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path, low_memory=False)
    df['league'] = league
    return df


def analyze_single_league(
    df: pd.DataFrame,
    league: str,
    task: str,
    output_dir: Path
) -> dict:
    """
    Analyze feature importance for a single league.
    
    Returns:
        dict with feature importance scores
    """
    print(f"\n{'='*70}")
    print(f"Analyzing: {league} ({task})")
    print('='*70)
    
    df = preprocess_data(df)
    df = clean_data(df)
    
    train_df, val_df, test_df = time_series_split(df)
    
    if task == 'halftime':
        X_train, y_train = process_halftime_features(train_df)
        X_val, y_val = process_halftime_features(val_df)
        X_test, y_test = process_halftime_features(test_df)
    else:
        X_train, y_train = process_prematch_features(train_df)
        X_val, y_val = process_prematch_features(val_df)
        X_test, y_test = process_prematch_features(test_df)
    
    train_features = set(X_train.columns)
    common_features = train_features.intersection(set(X_val.columns))
    common_features = common_features.intersection(set(X_test.columns))
    common_features = sorted(list(common_features))
    
    X_train = X_train[common_features]
    X_val = X_val[common_features]
    X_test = X_test[common_features]
    
    print(f"Features: {X_train.shape[1]} (common across train/val/test)")
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    trainer = Trainer(
        model_type='random_forest',
        params=MODEL_PARAMS.get('random_forest', {}),
        task_name=f'feature_analysis_{league}_{task}'
    )
    
    trainer.train(X_train, y_train, X_val, y_val)
    
    importance = trainer.get_feature_importance()
    
    if importance:
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 10 Features for {league}:")
        print("-" * 60)
        for i, (feat, score) in enumerate(sorted_importance[:10], 1):
            category = categorize_feature(feat)
            print(f"  {i:2d}. [{category:<15}] {feat:<35} {score:.4f}")
        
        league_output = output_dir / f'{league.lower()}_{task}'
        league_output.mkdir(parents=True, exist_ok=True)
        
        importance_df = pd.DataFrame([
            {
                'rank': i,
                'feature': feat,
                'importance': score,
                'category': categorize_feature(feat)
            }
            for i, (feat, score) in enumerate(sorted_importance, 1)
        ])
        importance_df.to_csv(league_output / 'feature_importance.csv', index=False)
        
        generate_league_visualization(
            importance=sorted_importance,
            league=league,
            task=task,
            output_dir=league_output
        )
    
    return importance


def generate_league_visualization(
    importance: list,
    league: str,
    task: str,
    output_dir: Path
):
    """Generate publication-quality visualizations for a single league."""
    
    top_n = 20
    top_features = importance[:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    features = [get_feature_display_name(f[0]) for f in top_features]
    scores = [f[1] for f in top_features]
    categories = [categorize_feature(f[0]) for f in top_features]
    
    unique_cats = sorted(set(categories))
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_cats)))
    cat_colors = dict(zip(unique_cats, colors))
    
    bar_colors = [cat_colors[cat] for cat in categories]
    
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, scores, color=bar_colors, alpha=0.85, edgecolor='gray', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score (Gini)', fontweight='bold')
    ax.set_title(f'Top {top_n} Feature Importance - {league} ({task.capitalize()})', 
                 fontweight='bold', pad=15)
    
    legend_elements = [plt.Rectangle((0,0), 1, 1, facecolor=cat_colors[cat], alpha=0.85)
                       for cat in unique_cats]
    ax.legend(legend_elements, unique_cats, loc='lower right', 
              title='Category', framealpha=0.9)
    
    for i, (score, cat) in enumerate(zip(scores, categories)):
        ax.text(score + max(scores) * 0.01, i, f'{score:.3f}', 
                va='center', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'feature_importance_top{top_n}.png')
    plt.savefig(output_dir / f'feature_importance_top{top_n}.pdf')
    plt.close()
    
    cat_importance = defaultdict(list)
    for feat, score in importance:
        cat = categorize_feature(feat)
        cat_importance[cat].append((feat, score))
    
    cat_stats = []
    for cat, feats in cat_importance.items():
        scores = [f[1] for f in feats]
        cat_stats.append({
            'category': cat,
            'count': len(feats),
            'total_importance': sum(scores),
            'mean_importance': np.mean(scores),
            'max_importance': max(scores),
        })
    
    cat_stats.sort(key=lambda x: x['total_importance'], reverse=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cats = [c['category'] for c in cat_stats]
    totals = [c['total_importance'] for c in cat_stats]
    counts = [c['count'] for c in cat_stats]
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(cats)))
    
    bars = ax.bar(cats, totals, color=colors, alpha=0.85, edgecolor='gray', linewidth=0.5)
    
    ax.set_xlabel('Feature Category', fontweight='bold')
    ax.set_ylabel('Total Importance Score', fontweight='bold')
    ax.set_title(f'Feature Importance by Category - {league} ({task.capitalize()})',
                 fontweight='bold', pad=15)
    
    plt.xticks(rotation=45, ha='right')
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(totals)*0.01,
                f'n={count}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance_by_category.png')
    plt.savefig(output_dir / 'feature_importance_by_category.pdf')
    plt.close()


def analyze_cross_league(
    all_importance: dict,
    task: str,
    output_dir: Path
):
    """
    Generate cross-league feature importance analysis.
    
    Identifies features that are consistently important across all leagues.
    """
    print(f"\n{'='*70}")
    print("Cross-League Feature Importance Analysis")
    print('='*70)
    
    feature_league_scores = defaultdict(lambda: {'scores': [], 'leagues': [], 'ranks': []})
    
    for league, importance in all_importance.items():
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (feat, score) in enumerate(sorted_features, 1):
            feature_league_scores[feat]['scores'].append(score)
            feature_league_scores[feat]['leagues'].append(league)
            feature_league_scores[feat]['ranks'].append(rank)
    
    cross_league_stats = []
    n_leagues = len(all_importance)
    
    for feat, data in feature_league_scores.items():
        if len(data['scores']) == n_leagues:
            cross_league_stats.append({
                'feature': feat,
                'avg_importance': np.mean(data['scores']),
                'std_importance': np.std(data['scores']),
                'avg_rank': np.mean(data['ranks']),
                'min_rank': min(data['ranks']),
                'max_rank': max(data['ranks']),
                'leagues': ', '.join(data['leagues']),
                'category': categorize_feature(feat),
            })
    
    cross_league_stats.sort(key=lambda x: x['avg_importance'], reverse=True)
    
    top_30 = cross_league_stats[:30]
    
    print(f"\nTop 30 Features (Consistent Across All {n_leagues} Leagues):")
    print("-" * 90)
    print(f"{'Rank':<5} {'Feature':<40} {'Avg Imp':>10} {'Std':>8} {'Avg Rank':>10} {'Category':<15}")
    print("-" * 90)
    
    for i, stat in enumerate(top_30, 1):
        print(f"{i:<5} {stat['feature']:<40} {stat['avg_importance']:>10.4f} "
              f"{stat['std_importance']:>8.4f} {stat['avg_rank']:>10.1f} {stat['category']:<15}")
    
    summary_df = pd.DataFrame(cross_league_stats)
    summary_df.insert(0, 'overall_rank', range(1, len(summary_df) + 1))
    summary_df.to_csv(output_dir / f'cross_league_feature_importance_{task}.csv', index=False)
    
    top_30_df = pd.DataFrame(top_30)
    top_30_df.insert(0, 'rank', range(1, 31))
    top_30_df.to_csv(output_dir / f'cross_league_top30_features_{task}.csv', index=False)
    
    generate_cross_league_visualization(top_30, task, output_dir)
    
    generate_category_summary(cross_league_stats, task, output_dir)
    
    return cross_league_stats


def generate_cross_league_visualization(
    top_30: list,
    task: str,
    output_dir: Path
):
    """Generate publication-quality cross-league visualization."""
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    features = [get_feature_display_name(s['feature']) for s in top_30]
    avg_scores = [s['avg_importance'] for s in top_30]
    std_scores = [s['std_importance'] for s in top_30]
    categories = [s['category'] for s in top_30]
    
    unique_cats = sorted(set(categories))
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_cats)))
    cat_colors = dict(zip(unique_cats, colors))
    
    bar_colors = [cat_colors[cat] for cat in categories]
    
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, avg_scores, xerr=std_scores, 
                   color=bar_colors, alpha=0.85, 
                   edgecolor='gray', linewidth=0.5,
                   capsize=2, error_kw={'linewidth': 0.5})
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Average Importance Score (with std)', fontweight='bold')
    ax.set_title(f'Top 30 Features Across All 5 Leagues ({task.capitalize()})',
                 fontweight='bold', pad=15, fontsize=14)
    
    legend_elements = [plt.Rectangle((0,0), 1, 1, facecolor=cat_colors[cat], alpha=0.85)
                       for cat in unique_cats]
    ax.legend(legend_elements, unique_cats, loc='lower right',
              title='Category', framealpha=0.9, fontsize=8)
    
    for i, (avg, std) in enumerate(zip(avg_scores, std_scores)):
        ax.text(avg + std + max(avg_scores) * 0.02, i, f'{avg:.3f}',
                va='center', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'cross_league_top30_{task}.png')
    plt.savefig(output_dir / f'cross_league_top30_{task}.pdf')
    plt.close()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Feature Importance Comparison by League ({task.capitalize()})',
                 fontweight='bold', fontsize=14, y=1.02)
    
    leagues = sorted(set(', '.join(s['leagues'] for s in top_30).split(', ')))
    
    for idx, league in enumerate(leagues[:6]):
        ax = axes[idx // 3, idx % 3]
        
        league_features = []
        league_scores = []
        
        for stat in top_30:
            if league in stat['leagues']:
                league_features.append(get_feature_display_name(stat['feature'])[:25])
                league_scores.append(stat['avg_importance'])
        
        if league_features:
            y_pos = np.arange(len(league_features))
            ax.barh(y_pos, league_scores, color='steelblue', alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(league_features, fontsize=6)
            ax.invert_yaxis()
            ax.set_xlabel('Importance', fontsize=8)
            ax.set_title(league, fontweight='bold', fontsize=10)
            ax.grid(axis='x', alpha=0.3)
    
    for idx in range(len(leagues), 6):
        axes[idx // 3, idx % 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'cross_league_by_league_{task}.png')
    plt.savefig(output_dir / f'cross_league_by_league_{task}.pdf')
    plt.close()


def generate_category_summary(
    cross_league_stats: list,
    task: str,
    output_dir: Path
):
    """Generate category-level summary statistics."""
    
    cat_stats = defaultdict(lambda: {
        'count': 0,
        'total_importance': 0,
        'features': [],
        'avg_ranks': []
    })
    
    for stat in cross_league_stats:
        cat = stat['category']
        cat_stats[cat]['count'] += 1
        cat_stats[cat]['total_importance'] += stat['avg_importance']
        cat_stats[cat]['features'].append(stat['feature'])
        cat_stats[cat]['avg_ranks'].append(stat['avg_rank'])
    
    cat_summary = []
    for cat, data in cat_stats.items():
        cat_summary.append({
            'category': cat,
            'feature_count': data['count'],
            'total_importance': data['total_importance'],
            'avg_importance': data['total_importance'] / data['count'],
            'avg_rank': np.mean(data['avg_ranks']),
            'top_feature': data['features'][0] if data['features'] else '',
        })
    
    cat_summary.sort(key=lambda x: x['total_importance'], reverse=True)
    
    print(f"\nCategory Summary:")
    print("-" * 80)
    print(f"{'Category':<20} {'Count':>8} {'Total Imp':>12} {'Avg Imp':>10} {'Avg Rank':>10}")
    print("-" * 80)
    for c in cat_summary:
        print(f"{c['category']:<20} {c['feature_count']:>8} {c['total_importance']:>12.4f} "
              f"{c['avg_importance']:>10.4f} {c['avg_rank']:>10.1f}")
    
    cat_df = pd.DataFrame(cat_summary)
    cat_df.to_csv(output_dir / f'category_summary_{task}.csv', index=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    cats = [c['category'] for c in cat_summary]
    totals = [c['total_importance'] for c in cat_summary]
    counts = [c['feature_count'] for c in cat_summary]
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(cats)))
    
    bars = ax.bar(cats, totals, color=colors, alpha=0.85, edgecolor='gray', linewidth=0.5)
    
    ax.set_xlabel('Feature Category', fontweight='bold')
    ax.set_ylabel('Total Importance Score', fontweight='bold')
    ax.set_title(f'Feature Importance by Category - All Leagues ({task.capitalize()})',
                 fontweight='bold', pad=15)
    
    plt.xticks(rotation=45, ha='right')
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(totals)*0.01,
                f'n={count}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'category_summary_{task}.png')
    plt.savefig(output_dir / f'category_summary_{task}.pdf')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Feature Importance Analysis')
    parser.add_argument('--task', type=str, default='halftime',
                        choices=['halftime', 'prematch'],
                        help='Prediction task (default: halftime)')
    parser.add_argument('--league', type=str, default='all',
                        help='League to analyze. Use "all" for all leagues, '
                             'or comma-separated list (e.g., "England,Spain")')
    parser.add_argument('--data-dir', type=str, 
                        default='d:/program/SoccerPre/merged_data/feature_engineer5',
                        help='Data directory path')
    
    args = parser.parse_args()
    
    setup_publication_style()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(__file__).resolve().parent.parent / 'outputs' / 'reports' / 'feature_importance'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Feature Importance Analysis")
    print("=" * 70)
    print(f"Task: {args.task}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    if args.league.lower() == 'all':
        leagues = list(LEAGUE_FILE_MAPPING.keys())
    else:
        leagues = [l.strip() for l in args.league.split(',')]
    
    print(f"Leagues: {leagues}")
    
    all_importance = {}
    
    for league in leagues:
        try:
            df = load_league_data(league, data_dir)
            importance = analyze_single_league(df, league, args.task, output_dir)
            if importance:
                all_importance[league] = importance
        except Exception as e:
            print(f"\nError processing {league}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(all_importance) > 1:
        analyze_cross_league(all_importance, args.task, output_dir)
    
    print("\n" + "=" * 70)
    print("Feature Importance Analysis Complete")
    print("=" * 70)
    print(f"\nOutput files saved to: {output_dir}")


if __name__ == "__main__":
    main()
