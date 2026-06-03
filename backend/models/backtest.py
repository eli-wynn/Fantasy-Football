"""
Walk-forward backtest of the projection model.

For each evaluation year (2022, 2023, 2024):
  1. Train on all seasons BEFORE that year
  2. Build player summaries using the 3 seasons before the eval year
  3. Project the eval year
  4. Compare projections to actual results (excluding injury-shortened seasons)
  5. Report MAE, rank accuracy, bust rate, sleeper rate

Run: python -m backend.models.backtest
"""
import numpy as np
import pandas as pd
from sqlalchemy import text
from backend.db import engine
from backend.models.train import engineer_features, FEATURES, TARGET
from backend.models.predict import build_weighted_summary
from xgboost import XGBRegressor

# ── Config ────────────────────────────────────────────────────────────────────

POSITIONS       = ["QB", "RB", "WR", "TE"]
EVAL_SEASONS    = [2022, 2023, 2024]   # seasons we test predictions against
MIN_GAMES       = 10                    # ignore players who played fewer games
                                        # (likely injured — not a fair test)
REPLACEMENT     = {"QB": 12, "RB": 24, "WR": 36, "TE": 12}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_stats() -> pd.DataFrame:
    """Pull every season from the DB into one big dataframe."""
    with engine.connect() as conn:
        df = pd.read_sql(
            text("SELECT * FROM weekly_stats ORDER BY player_id, season, week"),
            conn
        )
    print(f"Loaded {len(df):,} total rows ({df['season'].min()}–{df['season'].max()})")
    return df


def get_season_actuals(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    For a given season, calculate each player's actual full-season PPR total
    and games played. We use this to compare against our projections.
    Only keep players who played MIN_GAMES or more (injury filter).
    """
    season_df = df[df["season"] == season].copy()

    actuals = season_df.groupby(["player_id", "player_name", "position"]).agg(
        actual_games=("week", "count"),
        actual_total_ppr=("fantasy_points_ppr", "sum"),
        actual_ppg=("fantasy_points_ppr", "mean"),
    ).reset_index()

    # Injury filter — drop players who barely played
    actuals = actuals[actuals["actual_games"] >= MIN_GAMES]
    return actuals


# ── Training ──────────────────────────────────────────────────────────────────

def train_on_seasons(df: pd.DataFrame, train_seasons: list) -> dict:
    """
    Train one XGBoost model per position using only the specified seasons.
    Returns a dict of {position: trained_model}.

    This mirrors what train.py does but scoped to specific seasons so we
    never accidentally train on data from the future.
    """
    train_df = df[df["season"].isin(train_seasons)].copy()
    train_df = engineer_features(train_df)

    models = {}
    for pos in POSITIONS:
        pos_df = train_df[train_df["position"] == pos].copy()

        # Only keep features that exist and have data
        features = [
            f for f in FEATURES[pos]
            if f in pos_df.columns and pos_df[f].notna().any()
        ]

        # Drop rows missing the target or the core lag feature
        pos_df = pos_df.dropna(subset=["fp_lag1", TARGET])
        pos_df[features] = pos_df[features].fillna(0)

        if len(pos_df) < 100:
            print(f"    {pos}: not enough data ({len(pos_df)} rows), skipping")
            continue

        model = XGBRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            random_state=42, n_jobs=-1,
        )
        model.fit(pos_df[features], pos_df[TARGET], verbose=False)
        models[pos] = {"model": model, "features": features}
        print(f"    {pos}: trained on {len(pos_df):,} rows")

    return models


# ── Projecting ────────────────────────────────────────────────────────────────

def project_season(df: pd.DataFrame, models: dict, base_season: int) -> pd.DataFrame:
    """
    Use the trained models to project the season AFTER base_season.
    Builds player summaries from the 3 seasons up to and including base_season,
    then runs the model on them.

    This is the same logic as predict.py but scoped to historical data.
    """
    # Build per-player feature rows using weighted 3-season average
    season_data = {
        s: df[df["season"] == s]
        for s in range(base_season - 2, base_season + 1)
        if s in df["season"].values
    }

    if not season_data:
        return pd.DataFrame()

    summary = build_weighted_summary(season_data, base_season)

    all_rows = []
    for pos in POSITIONS:
        if pos not in models:
            continue

        model    = models[pos]["model"]
        features = models[pos]["features"]

        pos_df = summary[summary["position"] == pos].copy()
        for col in [f for f in features if f not in pos_df.columns]:
            pos_df[col] = 0.0

        X = pos_df[features].fillna(0)
        preds = model.predict(X)

        pos_df = pos_df.copy()
        pos_df["projected_ppg"] = np.maximum(preds, 0)
        all_rows.append(pos_df[["player_id", "player_name", "position", "projected_ppg"]])

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(projections: pd.DataFrame, actuals: pd.DataFrame, season: int) -> dict:
    """
    Compare projections to actuals for one season.

    Metrics explained:
    - MAE: average absolute error in PPG (lower = better)
    - Rank correlation: do our projected rankings match actual rankings?
      1.0 = perfect, 0.0 = random, negative = worse than random
    - Bust rate: % of players we projected in the top tier who flopped
    - Sleeper rate: % of players outside our top tier who outperformed it
    """
    # Join projections with actuals on player_id
    merged = projections.merge(actuals, on=["player_id", "position"], how="inner")

    if merged.empty:
        return {}

    results = {"season": season, "positions": {}}

    for pos in POSITIONS:
        pos_df = merged[merged["position"] == pos].copy()
        if len(pos_df) < 5:
            continue

        # ── MAE ───────────────────────────────────────────────────────────────
        # How many PPG off were we on average?
        mae = (pos_df["projected_ppg"] - pos_df["actual_ppg"]).abs().mean()

        # ── Rank correlation ──────────────────────────────────────────────────
        # Rank each player by our projection and by their actual finish.
        # Spearman correlation tells us how well the rankings agree.
        # 1.0 = we ranked them in exactly the right order.
        pos_df["proj_rank"]   = pos_df["projected_ppg"].rank(ascending=False)
        pos_df["actual_rank"] = pos_df["actual_ppg"].rank(ascending=False)
        rank_corr = pos_df["proj_rank"].corr(pos_df["actual_rank"], method="spearman")

        # ── Bust rate ─────────────────────────────────────────────────────────
        # Of the players we projected in the top tier (e.g. top 12 QBs),
        # what % actually finished outside 1.5x that tier (e.g. outside top 18)?
        # A bust is someone we drafted expecting starter value who gave backup value.
        top_n    = REPLACEMENT[pos]
        bust_cutoff = int(top_n * 1.5)
        projected_top = pos_df[pos_df["proj_rank"] <= top_n]
        busts    = projected_top[projected_top["actual_rank"] > bust_cutoff]
        bust_rate = len(busts) / max(len(projected_top), 1) * 100

        # ── Sleeper rate ──────────────────────────────────────────────────────
        # Of the players we had OUTSIDE the top tier,
        # what % actually finished INSIDE the top tier?
        # A sleeper is someone we undervalued who outperformed our expectation.
        projected_outside = pos_df[pos_df["proj_rank"] > top_n]
        sleepers  = projected_outside[projected_outside["actual_rank"] <= top_n]
        sleeper_rate = len(sleepers) / max(len(projected_outside), 1) * 100

        results["positions"][pos] = {
            "n":            len(pos_df),
            "mae":          round(float(mae), 2),
            "rank_corr":    round(float(rank_corr), 3),
            "bust_rate":    round(float(bust_rate), 1),
            "sleeper_rate": round(float(sleeper_rate), 1),
            "busts":        busts["player_name_x"].tolist() if "player_name_x" in busts.columns else [],
            "sleepers":     sleepers["player_name_x"].tolist() if "player_name_x" in sleepers.columns else [],
        }

    return results


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(all_results: list):
    """Print a clean summary table of all backtest results."""
    print("\n" + "=" * 70)
    print("BACKTEST REPORT")
    print("=" * 70)
    print(f"{'Season':<8} {'Pos':<5} {'N':<5} {'MAE':>6} {'RankCorr':>10} {'Bust%':>7} {'Sleeper%':>9}")
    print("-" * 70)

    for result in all_results:
        season = result["season"]
        for pos, metrics in result["positions"].items():
            print(
                f"{season:<8} {pos:<5} {metrics['n']:<5} "
                f"{metrics['mae']:>6.2f} {metrics['rank_corr']:>10.3f} "
                f"{metrics['bust_rate']:>6.1f}% {metrics['sleeper_rate']:>8.1f}%"
            )
        print()

    # Average across all seasons
    print("-" * 70)
    print("AVERAGES ACROSS ALL EVAL SEASONS:")
    for pos in POSITIONS:
        pos_results = [
            r["positions"][pos]
            for r in all_results
            if pos in r["positions"]
        ]
        if not pos_results:
            continue
        avg_mae       = np.mean([r["mae"] for r in pos_results])
        avg_rank_corr = np.mean([r["rank_corr"] for r in pos_results])
        avg_bust      = np.mean([r["bust_rate"] for r in pos_results])
        avg_sleeper   = np.mean([r["sleeper_rate"] for r in pos_results])
        print(
            f"{'AVG':<8} {pos:<5} {'':5} "
            f"{avg_mae:>6.2f} {avg_rank_corr:>10.3f} "
            f"{avg_bust:>6.1f}% {avg_sleeper:>8.1f}%"
        )

    print("\nMetric guide:")
    print("  MAE        — avg PPG error (lower is better)")
    print("  RankCorr   — ranking agreement, 1.0=perfect (higher is better)")
    print("  Bust%      — % of projected starters who flopped (lower is better)")
    print("  Sleeper%   — % of non-starters who outperformed (unavoidable, FYI only)")
    print("=" * 70)


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    df = load_all_stats()
    all_results = []

    for eval_season in EVAL_SEASONS:
        print(f"\n{'─' * 50}")
        print(f"Evaluating season: {eval_season}")

        # All seasons strictly before the eval season are fair to train on
        train_seasons = sorted([s for s in df["season"].unique() if s < eval_season])
        if len(train_seasons) < 2:
            print(f"  Not enough training seasons, skipping")
            continue

        print(f"  Training on: {train_seasons}")
        models = train_on_seasons(df, train_seasons)

        # Project the season before eval (base_season) to get eval_season predictions
        base_season = eval_season - 1
        print(f"  Projecting from {base_season} data → {eval_season} predictions")
        projections = project_season(df, models, base_season)

        # Get actual results for eval_season with injury filter
        actuals = get_season_actuals(df, eval_season)
        print(f"  Actual qualified players (>={MIN_GAMES} games): {len(actuals)}")

        result = evaluate(projections, actuals, eval_season)
        if result:
            all_results.append(result)

    print_report(all_results)


if __name__ == "__main__":
    run()
