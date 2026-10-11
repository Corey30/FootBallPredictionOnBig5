import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'sans-serif'

DATA_DIR = Path(r'd:\program\SoccerPre\merged_data\feature_engineer5')
OUTPUT_DIR = Path(r'd:\program\SoccerPre\outputs\league_analysis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LEAGUE_NAMES = {
    'England': 'Premier League',
    'France': 'Ligue 1',
    'Germany': 'Bundesliga',
    'Italy': 'Serie A',
    'Spain': 'La Liga',
}
LEAGUE_ORDER = ['England', 'France', 'Germany', 'Italy', 'Spain']
COLORS = {'H': '#e74c3c', 'D': '#95a5a6', 'A': '#3498db'}


def load_all_leagues():
    dfs = {}
    for league in LEAGUE_ORDER:
        path = DATA_DIR / f'{league}.csv'
        df = pd.read_csv(path)
        df['date_GMT'] = pd.to_datetime(df['date_GMT'], format='%b %d %Y - %I:%M%p', errors='coerce')
        df = df.sort_values('date_GMT').reset_index(drop=True)
        df['league'] = league
        dfs[league] = df
    return dfs


def plot_league_result_distribution(dfs):
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)
    fig.suptitle('Match Result Distribution Across Top 5 Leagues', fontsize=16, fontweight='bold')

    for i, league in enumerate(LEAGUE_ORDER):
        df = dfs[league]
        counts = df['target_match_result'].value_counts().reindex(['H', 'D', 'A'])
        total = counts.sum()
        pcts = counts / total * 100

        bars = axes[i].bar(['H (Home Win)', 'D (Draw)', 'A (Away Win)'], counts.values,
                           color=[COLORS['H'], COLORS['D'], COLORS['A']], edgecolor='white', linewidth=1.2)
        axes[i].set_title(f'{LEAGUE_NAMES[league]}\nTotal: {total} matches', fontsize=12)
        for bar, pct, cnt in zip(bars, pcts, counts):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                         f'{cnt}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)

    axes[0].set_ylabel('Number of Matches')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '1_league_result_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[1] Match result distribution -> 1_league_result_distribution.png')


def plot_league_result_comparison(dfs):
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(LEAGUE_ORDER))
    width = 0.25

    h_pcts, d_pcts, a_pcts = [], [], []
    for league in LEAGUE_ORDER:
        vc = dfs[league]['target_match_result'].value_counts(normalize=True)
        h_pcts.append(vc.get('H', 0) * 100)
        d_pcts.append(vc.get('D', 0) * 100)
        a_pcts.append(vc.get('A', 0) * 100)

    labels = [f'{LEAGUE_NAMES[l]}' for l in LEAGUE_ORDER]
    ax.bar(x - width, h_pcts, width, label='Home Win (H)', color=COLORS['H'], edgecolor='white')
    ax.bar(x, d_pcts, width, label='Draw (D)', color=COLORS['D'], edgecolor='white')
    ax.bar(x + width, a_pcts, width, label='Away Win (A)', color=COLORS['A'], edgecolor='white')

    for i in range(len(LEAGUE_ORDER)):
        ax.text(x[i] - width, h_pcts[i] + 0.5, f'{h_pcts[i]:.1f}%', ha='center', fontsize=9)
        ax.text(x[i], d_pcts[i] + 0.5, f'{d_pcts[i]:.1f}%', ha='center', fontsize=9)
        ax.text(x[i] + width, a_pcts[i] + 0.5, f'{a_pcts[i]:.1f}%', ha='center', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Home Win / Draw / Away Win Distribution Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(max(h_pcts), max(a_pcts)) + 8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '2_league_result_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[2] League result comparison -> 2_league_result_comparison.png')


def plot_team_home_away_draw(dfs):
    for league in LEAGUE_ORDER:
        df = dfs[league]
        teams = sorted(set(df['home_team_name'].unique()) | set(df['away_team_name'].unique()))

        records = []
        for team in teams:
            home = df[df['home_team_name'] == team]
            away = df[df['away_team_name'] == team]
            h_win = len(home[home['target_match_result'] == 'H'])
            h_draw = len(home[home['target_match_result'] == 'D'])
            h_loss = len(home[home['target_match_result'] == 'A'])
            a_win = len(away[away['target_match_result'] == 'A'])
            a_draw = len(away[away['target_match_result'] == 'D'])
            a_loss = len(away[away['target_match_result'] == 'H'])
            total = h_win + h_draw + h_loss + a_win + a_draw + a_loss
            if total >= 30:
                records.append({
                    'team': team,
                    'home_win': h_win, 'home_draw': h_draw, 'home_loss': h_loss,
                    'away_win': a_win, 'away_draw': a_draw, 'away_loss': a_loss,
                    'total': total,
                    'home_win_pct': h_win / (h_win + h_draw + h_loss) * 100 if (h_win + h_draw + h_loss) > 0 else 0,
                    'home_draw_pct': h_draw / (h_win + h_draw + h_loss) * 100 if (h_win + h_draw + h_loss) > 0 else 0,
                    'home_loss_pct': h_loss / (h_win + h_draw + h_loss) * 100 if (h_win + h_draw + h_loss) > 0 else 0,
                    'away_win_pct': a_win / (a_win + a_draw + a_loss) * 100 if (a_win + a_draw + a_loss) > 0 else 0,
                    'away_draw_pct': a_draw / (a_win + a_draw + a_loss) * 100 if (a_win + a_draw + a_loss) > 0 else 0,
                    'away_loss_pct': a_loss / (a_win + a_draw + a_loss) * 100 if (a_win + a_draw + a_loss) > 0 else 0,
                })

        if not records:
            continue

        team_df = pd.DataFrame(records).sort_values('home_win_pct', ascending=True)

        n_teams = len(team_df)
        fig_height = max(8, n_teams * 0.35)
        fig, axes = plt.subplots(1, 2, figsize=(18, fig_height), sharey=True)
        fig.suptitle(f'{LEAGUE_NAMES[league]} - Team Home/Away Result Distribution', fontsize=14, fontweight='bold')

        y = np.arange(n_teams)
        bar_h = 0.7

        ax = axes[0]
        ax.set_title('Home Matches', fontsize=12, fontweight='bold')
        left = np.zeros(n_teams)
        for label, col, color in [('Home Win', 'home_win_pct', COLORS['H']),
                                   ('Draw', 'home_draw_pct', COLORS['D']),
                                   ('Away Win', 'home_loss_pct', COLORS['A'])]:
            ax.barh(y, team_df[col].values, bar_h, left=left, label=label, color=color, edgecolor='white', linewidth=0.5)
            left += team_df[col].values
        ax.set_yticks(y)
        ax.set_yticklabels(team_df['team'].values, fontsize=8)
        ax.set_xlabel('Percentage (%)')
        ax.legend(loc='lower right', fontsize=9)
        ax.set_xlim(0, 100)

        ax = axes[1]
        ax.set_title('Away Matches', fontsize=12, fontweight='bold')
        left = np.zeros(n_teams)
        for label, col, color in [('Home Win', 'away_loss_pct', COLORS['H']),
                                   ('Draw', 'away_draw_pct', COLORS['D']),
                                   ('Away Win', 'away_win_pct', COLORS['A'])]:
            ax.barh(y, team_df[col].values, bar_h, left=left, label=label, color=color, edgecolor='white', linewidth=0.5)
            left += team_df[col].values
        ax.set_xlabel('Percentage (%)')
        ax.legend(loc='lower right', fontsize=9)
        ax.set_xlim(0, 100)

        plt.tight_layout()
        fname = f'3_{league}_team_home_away_draw.png'
        plt.savefig(OUTPUT_DIR / fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'[3] {LEAGUE_NAMES[league]} team distribution -> {fname}')


def plot_odds_distribution(dfs):
    odds_cols = {
        'odds_ft_home_team_win': 'Home Win Odds',
        'odds_ft_draw': 'Draw Odds',
        'odds_ft_away_team_win': 'Away Win Odds',
    }

    all_df = pd.concat([dfs[l] for l in LEAGUE_ORDER], ignore_index=True)
    all_df = all_df.dropna(subset=list(odds_cols.keys()))

    league_colors = {
        'England': '#e74c3c', 'France': '#3498db', 'Germany': '#2ecc71',
        'Italy': '#9b59b6', 'Spain': '#f39c12',
    }

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Odds Distribution by League', fontsize=16, fontweight='bold')

    for idx, (col, col_label) in enumerate(odds_cols.items()):
        ax = axes[0][idx]
        for league in LEAGUE_ORDER:
            df = dfs[league].dropna(subset=[col])
            data = df[col].values
            data = data[(data > 0) & (data < 20)]
            if len(data) == 0:
                continue
            ax.hist(data, bins=60, density=True, alpha=0.5,
                    label=LEAGUE_NAMES[league], color=league_colors[league], edgecolor='white')
        ax.set_title(f'{col_label} (By League)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Odds')
        ax.set_ylabel('Density')
        ax.legend(fontsize=9)
        ax.set_xlim(0, 15)

    for idx, (col, col_label) in enumerate(odds_cols.items()):
        ax = axes[1][idx]
        data = all_df[col].values
        data = data[(data > 0) & (data < 20)]
        if len(data) > 0:
            ax.hist(data, bins=80, density=True, alpha=0.7, color='#2c3e50', edgecolor='white')
            from scipy import stats
            kde_x = np.linspace(0, 15, 300)
            kde = stats.gaussian_kde(data)
            ax.plot(kde_x, kde(kde_x), color='#e74c3c', linewidth=2, label='KDE')
            ax.axvline(np.median(data), color='#f39c12', linestyle='--', linewidth=1.5,
                       label=f'Median={np.median(data):.2f}')
            ax.axvline(np.mean(data), color='#3498db', linestyle='--', linewidth=1.5,
                       label=f'Mean={np.mean(data):.2f}')
            ax.legend(fontsize=9)
        ax.set_title(f'{col_label} (Overall)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Odds')
        ax.set_ylabel('Density')
        ax.set_xlim(0, 15)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '4_odds_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[4] Odds distribution (combined) -> 4_odds_distribution.png')

    for league in LEAGUE_ORDER:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'{LEAGUE_NAMES[league]} - Odds Distribution', fontsize=16, fontweight='bold')

        for idx, (col, col_label) in enumerate(odds_cols.items()):
            ax = axes[idx]
            df = dfs[league].dropna(subset=[col])
            data = df[col].values
            data = data[(data > 0) & (data < 20)]
            if len(data) > 0:
                ax.hist(data, bins=60, density=True, alpha=0.7, color=league_colors[league], edgecolor='white')
                from scipy import stats
                kde_x = np.linspace(0, 15, 300)
                kde = stats.gaussian_kde(data)
                ax.plot(kde_x, kde(kde_x), color='#2c3e50', linewidth=2, label='KDE')
                ax.axvline(np.median(data), color='#f39c12', linestyle='--', linewidth=1.5,
                           label=f'Median={np.median(data):.2f}')
                ax.axvline(np.mean(data), color='#3498db', linestyle='--', linewidth=1.5,
                           label=f'Mean={np.mean(data):.2f}')
                ax.legend(fontsize=9)
            ax.set_title(col_label, fontsize=12, fontweight='bold')
            ax.set_xlabel('Odds')
            ax.set_ylabel('Density')
            ax.set_xlim(0, 15)

        plt.tight_layout()
        fname = f'4_{league}_odds_distribution.png'
        plt.savefig(OUTPUT_DIR / fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'[4] {LEAGUE_NAMES[league]} odds distribution -> {fname}')


def plot_odds_by_result(dfs):
    all_df = pd.concat([dfs[l] for l in LEAGUE_ORDER], ignore_index=True)
    all_df = all_df.dropna(subset=['odds_ft_home_team_win', 'odds_ft_draw', 'odds_ft_away_team_win', 'target_match_result'])

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Odds Distribution vs Actual Match Result', fontsize=16, fontweight='bold')

    odds_col_map = {
        'odds_ft_home_team_win': 'Home Win Odds',
        'odds_ft_draw': 'Draw Odds',
        'odds_ft_away_team_win': 'Away Win Odds',
    }

    for idx, (col, col_label) in enumerate(odds_col_map.items()):
        ax = axes[idx]
        for result, color, label in [('H', COLORS['H'], 'Actual Home Win'),
                                      ('D', COLORS['D'], 'Actual Draw'),
                                      ('A', COLORS['A'], 'Actual Away Win')]:
            data = all_df[all_df['target_match_result'] == result][col].values
            data = data[(data > 0) & (data < 20)]
            if len(data) > 0:
                ax.hist(data, bins=60, density=True, alpha=0.5, color=color, label=label, edgecolor='white')
        ax.set_title(col_label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Odds')
        ax.set_ylabel('Density')
        ax.legend(fontsize=9)
        ax.set_xlim(0, 15)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '5_odds_by_result.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[5] Odds distribution vs actual result -> 5_odds_by_result.png')


def print_summary(dfs):
    print('\n' + '=' * 80)
    print('Top 5 Leagues Data Summary')
    print('=' * 80)

    for league in LEAGUE_ORDER:
        df = dfs[league]
        vc = df['target_match_result'].value_counts()
        total = len(df)
        print(f'\n--- {LEAGUE_NAMES[league]} ({league}) ---')
        print(f'  Total matches: {total}')
        print(f'  Home Win (H): {vc.get("H",0)} ({vc.get("H",0)/total*100:.1f}%)')
        print(f'  Draw (D): {vc.get("D",0)} ({vc.get("D",0)/total*100:.1f}%)')
        print(f'  Away Win (A): {vc.get("A",0)} ({vc.get("A",0)/total*100:.1f}%)')
        odds_h = df['odds_ft_home_team_win'].dropna()
        odds_d = df['odds_ft_draw'].dropna()
        odds_a = df['odds_ft_away_team_win'].dropna()
        print(f'  Home Win Odds: Mean={odds_h.mean():.2f}, Median={odds_h.median():.2f}')
        print(f'  Draw Odds: Mean={odds_d.mean():.2f}, Median={odds_d.median():.2f}')
        print(f'  Away Win Odds: Mean={odds_a.mean():.2f}, Median={odds_a.median():.2f}')

    all_df = pd.concat([dfs[l] for l in LEAGUE_ORDER], ignore_index=True)
    total = len(all_df)
    vc = all_df['target_match_result'].value_counts()
    print(f'\n--- All Leagues Combined ---')
    print(f'  Total matches: {total}')
    print(f'  Home Win (H): {vc.get("H",0)} ({vc.get("H",0)/total*100:.1f}%)')
    print(f'  Draw (D): {vc.get("D",0)} ({vc.get("D",0)/total*100:.1f}%)')
    print(f'  Away Win (A): {vc.get("A",0)} ({vc.get("A",0)/total*100:.1f}%)')


if __name__ == '__main__':
    print('Loading data...')
    dfs = load_all_leagues()

    print_summary(dfs)

    print('\nGenerating visualizations...')
    plot_league_result_distribution(dfs)
    plot_league_result_comparison(dfs)
    plot_team_home_away_draw(dfs)
    plot_odds_distribution(dfs)
    plot_odds_by_result(dfs)

    print(f'\nAll figures saved to: {OUTPUT_DIR}')
