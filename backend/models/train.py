"""
Train an XGBoost model to project fantasy points per position.
Run: python -m backend.models.train
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor
from sqlalchemy import text
from backend.db import engine

MODEL_DIR = Path(__file__).parent / "artifacts"
MODEL_DIR.mkdir(exist_ok=True)

POSITIONS = ["QB", "RB", "WR", "TE"]
TARGET = "fantasy_points_ppr"

# Share/opportunity stats that get their own lag + rolling features
# Only include ones actually present in the weekly data
SHARE_STATS = ["target_share", "air_yards_share", "wopr", "racr"]

# Per-position feature sets
FEATURES = {
    "QB": [
        "season", "week", "snap_pct",
        "completions_lag1", "attempts_lag1", "passing_yards_lag1",
        "passing_tds_lag1", "interceptions_lag1",
        "carries_lag1", "rushing_yards_lag1", "rushing_tds_lag1",
        "fp_lag1", "fp_lag2", "fp_lag3", "fp_roll4_mean", "fp_roll4_std",
        "season_fp_mean", "season_fp_games",
        "opp_pts_allowed_roll4", "opp_pts_allowed_season",
    ],
    "RB": [
        "season", "week", "snap_pct",
        "carries_lag1", "rushing_yards_lag1", "rushing_tds_lag1",
        "targets_lag1", "receptions_lag1", "receiving_yards_lag1", "receiving_tds_lag1",
        "target_share_lag1", "target_share_roll4_mean",
        "fp_lag1", "fp_lag2", "fp_lag3", "fp_roll4_mean", "fp_roll4_std",
        "season_fp_mean", "season_fp_games",
        "opp_pts_allowed_roll4", "opp_pts_allowed_season",
    ],
    "WR": [
        "season", "week", "snap_pct",
        "targets_lag1", "receptions_lag1", "receiving_yards_lag1", "receiving_tds_lag1",
        "target_share_lag1", "target_share_roll4_mean",
        "air_yards_share_lag1", "air_yards_share_roll4_mean",
        "wopr_lag1", "wopr_roll4_mean",
        "racr_lag1",
        "fp_lag1", "fp_lag2", "fp_lag3", "fp_roll4_mean", "fp_roll4_std",
        "season_fp_mean", "season_fp_games",
        "opp_pts_allowed_roll4", "opp_pts_allowed_season",
    ],
    "TE": [
        "season", "week", "snap_pct",
        "targets_lag1", "receptions_lag1", "receiving_yards_lag1", "receiving_tds_lag1",
        "target_share_lag1", "target_share_roll4_mean",
        "wopr_lag1", "wopr_roll4_mean",
        "racr_lag1",
        "fp_lag1", "fp_lag2", "fp_lag3", "fp_roll4_mean", "fp_roll4_std",
        "season_fp_mean", "season_fp_games",
        "opp_pts_allowed_roll4", "opp_pts_allowed_season",
    ],
}


def load_data() -> pd.DataFrame:
    query = text("SELECT * FROM weekly_stats ORDER BY player_id, season, week")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    print(f"Loaded {len(df):,} rows from weekly_stats")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "season", "week"]).copy()
    grp = df.groupby("player_id")

    # Fantasy point lags + rolling
    df["fp_lag1"] = grp["fantasy_points_ppr"].shift(1)
    df["fp_lag2"] = grp["fantasy_points_ppr"].shift(2)
    df["fp_lag3"] = grp["fantasy_points_ppr"].shift(3)
    df["fp_roll4_mean"] = (
        grp["fantasy_points_ppr"].shift(1).rolling(4).mean()
        .reset_index(level=0, drop=True)
    )
    df["fp_roll4_std"] = (
        grp["fantasy_points_ppr"].shift(1).rolling(4).std()
        .reset_index(level=0, drop=True)
    )

    # Counting stat lags
    for col in [
        "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
        "carries", "rushing_yards", "rushing_tds",
        "targets", "receptions", "receiving_yards", "receiving_tds",
    ]:
        if col in df.columns:
            df[f"{col}_lag1"] = grp[col].shift(1)

    # Share/opportunity stat lags + 4-week rolling mean
    for col in SHARE_STATS:
        if col in df.columns:
            shifted = grp[col].shift(1)
            df[f"{col}_lag1"] = shifted
            df[f"{col}_roll4_mean"] = (
                shifted.rolling(4).mean().reset_index(level=0, drop=True)
            )

    # Season-to-date (excluding current week to avoid leakage)
    season_grp = df.groupby(["player_id", "season"])
    df["season_fp_mean"] = season_grp["fantasy_points_ppr"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["season_fp_games"] = season_grp["fantasy_points_ppr"].transform(
        lambda x: x.shift(1).expanding().count()
    )

    # Fill missing snap% — use position-season median, then fall back to 0.6
    df["snap_pct"] = (
        df["snap_pct"]
        .fillna(df.groupby(["position", "season"])["snap_pct"].transform("median"))
        .fillna(0.6)
    )

    return df


def train_position(df: pd.DataFrame, pos: str) -> dict:
    pos_df = df[df["position"] == pos].copy()

    # Only keep features that exist AND have at least some non-null values
    features = [
        f for f in FEATURES[pos]
        if f in pos_df.columns and pos_df[f].notna().any()
    ]

    # Drop rows missing the target or any core lag features (fp_lag1 is essential)
    must_have = [f for f in features if f in ("fp_lag1", TARGET)]
    pos_df = pos_df.dropna(subset=must_have + [TARGET])

    # Fill remaining NaNs with 0 so optional features don't shrink the dataset
    pos_df[features] = pos_df[features].fillna(0)

    if len(pos_df) < 200:
        print(f"  {pos}: insufficient data ({len(pos_df)} rows), skipping")
        return {}

    X = pos_df[features]
    y = pos_df[TARGET]

    tscv = TimeSeriesSplit(n_splits=3)
    maes, rmses = [], []

    model = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1,
    )

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        maes.append(mean_absolute_error(y_val, preds))
        rmses.append(root_mean_squared_error(y_val, preds))

    # Final fit on all data
    model.fit(X, y, verbose=False)
    joblib.dump({"model": model, "features": features}, MODEL_DIR / f"{pos.lower()}_model.pkl")

    metrics = {
        "position": pos,
        "rows": len(pos_df),
        "mae": round(float(np.mean(maes)), 3),
        "rmse": round(float(np.mean(rmses)), 3),
        "features": features,
    }
    print(f"  {pos}: MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  n={metrics['rows']:,}")
    return metrics


def run():
    df = load_data()
    df = engineer_features(df)

    all_metrics = {}
    for pos in POSITIONS:
        metrics = train_position(df, pos)
        if metrics:
            all_metrics[pos] = metrics

    joblib.dump(all_metrics, MODEL_DIR / "metrics.pkl")
    print(f"\nModels saved to {MODEL_DIR}")
    return all_metrics


if __name__ == "__main__":
    run()
