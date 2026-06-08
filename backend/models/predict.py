"""
Load trained models and generate per-player projections for a target season.
Uses a 3-season weighted average to handle injuries and small samples.
Run: python -m backend.models.predict
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sqlalchemy import text
from backend.db import engine, projections, metadata

MODEL_DIR = Path(__file__).parent / "artifacts"
POSITIONS = ["QB", "RB", "WR", "TE"]
FULL_SEASON_GAMES = 17

# Replacement-level thresholds (last starter at position in a 12-team league)
REPLACEMENT_RANK = {"QB": 12, "RB": 24, "WR": 36, "TE": 12}

# Season weights — most recent season gets highest weight
SEASON_WEIGHTS = {0: 0.60, 1: 0.28, 2: 0.12}  # 0 = most recent


def load_multi_season_stats(base_season: int, n_seasons: int = 3) -> dict[int, pd.DataFrame]:
    """Load stats for the last n seasons."""
    seasons = {}
    with engine.connect() as conn:
        for i in range(n_seasons):
            season = base_season - i
            df = pd.read_sql(
                text("SELECT * FROM weekly_stats WHERE season = :s ORDER BY player_id, week"),
                conn, params={"s": season}
            )
            if not df.empty:
                seasons[season] = df
                print(f"  Loaded {len(df):,} rows for {season}")
    return seasons


def summarise_season(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Aggregate a single season into per-player weekly averages.
    Extrapolates to a full 17-game season based on per-game rate.
    """
    agg = df.groupby(["player_id", "player_name", "position", "team"]).agg(
        games_played=("week", "count"),
        fp_mean=("fantasy_points_ppr", "mean"),
        fp_std=("fantasy_points_ppr", "std"),
        completions_mean=("completions", "mean"),
        attempts_mean=("attempts", "mean"),
        passing_yards_mean=("passing_yards", "mean"),
        passing_tds_mean=("passing_tds", "mean"),
        interceptions_mean=("interceptions", "mean"),
        carries_mean=("carries", "mean"),
        rushing_yards_mean=("rushing_yards", "mean"),
        rushing_tds_mean=("rushing_tds", "mean"),
        targets_mean=("targets", "mean"),
        receptions_mean=("receptions", "mean"),
        receiving_yards_mean=("receiving_yards", "mean"),
        receiving_tds_mean=("receiving_tds", "mean"),
        snap_pct_mean=("snap_pct", "mean"),
        target_share_mean=("target_share", "mean"),
        air_yards_share_mean=("air_yards_share", "mean"),
        wopr_mean=("wopr", "mean"),
        racr_mean=("racr", "mean"),
        opp_pts_allowed_roll4_mean=("opp_pts_allowed_roll4", "mean"),
        opp_pts_allowed_season_mean=("opp_pts_allowed_season", "mean"),
    ).reset_index()

    agg["season"] = season
    agg["games_played"] = agg["games_played"].fillna(0)

    # Sample size confidence: how much to trust this season
    # Players with fewer games get pulled toward position mean later
    agg["sample_confidence"] = np.minimum(agg["games_played"] / FULL_SEASON_GAMES, 1.0)

    # Flag injury seasons — only penalise very small samples (< 5 games)
    # PPG is already per-game normalised so mid-season injuries don't distort the mean
    agg["injury_season"] = (agg["games_played"] < 5).astype(int)

    return agg


def build_weighted_summary(season_data: dict[int, pd.DataFrame], base_season: int) -> pd.DataFrame:
    """
    Build a single prediction row per player using weighted average across seasons.
    More recent seasons get more weight. Injury seasons get down-weighted.
    """
    sorted_seasons = sorted(season_data.keys(), reverse=True)
    summaries = []

    for i, season in enumerate(sorted_seasons):
        s = summarise_season(season_data[season], season)
        base_weight = SEASON_WEIGHTS.get(i, 0.05)

        # Down-weight only genuinely tiny samples (< 5 games)
        s["weight"] = base_weight * np.where(s["injury_season"] == 1, 0.8, 1.0)
        summaries.append(s)

    # Combine all seasons
    combined = pd.concat(summaries, ignore_index=True)

    # Weighted average per player
    stat_cols = [
        "fp_mean", "fp_std",
        "completions_mean", "attempts_mean", "passing_yards_mean",
        "passing_tds_mean", "interceptions_mean",
        "carries_mean", "rushing_yards_mean", "rushing_tds_mean",
        "targets_mean", "receptions_mean", "receiving_yards_mean", "receiving_tds_mean",
        "snap_pct_mean", "target_share_mean", "air_yards_share_mean",
        "wopr_mean", "racr_mean",
        "opp_pts_allowed_roll4_mean", "opp_pts_allowed_season_mean",
    ]

    rows = []
    for player_id, group in combined.groupby("player_id"):
        total_weight = group["weight"].sum()
        if total_weight == 0:
            continue

        row = {
            "player_id": player_id,
            "player_name": group["player_name"].iloc[0],
            "position": group["position"].iloc[0],
            "team": group.sort_values("season", ascending=False)["team"].iloc[0],
            "games_played": group.sort_values("season", ascending=False)["games_played"].iloc[0],
            "injury_season": group.sort_values("season", ascending=False)["injury_season"].iloc[0],
            "seasons_available": len(group),
        }

        for col in stat_cols:
            if col in group.columns:
                row[col] = np.average(group[col].fillna(0), weights=group["weight"])

        rows.append(row)

    result = pd.DataFrame(rows)

    # Build model features
    result["season"] = base_season + 1
    result["week"] = 9
    result["fp_lag1"] = result["fp_mean"]
    result["fp_lag2"] = result["fp_mean"]
    result["fp_lag3"] = result["fp_mean"]
    result["fp_roll4_mean"] = result["fp_mean"]
    result["fp_roll4_std"] = result["fp_std"].fillna(0)
    result["season_fp_mean"] = result["fp_mean"]
    result["season_fp_games"] = result["games_played"]
    result["snap_pct"] = result["snap_pct_mean"]
    result["target_share_lag1"] = result["target_share_mean"]
    result["target_share_roll4_mean"] = result["target_share_mean"]
    result["air_yards_share_lag1"] = result["air_yards_share_mean"]
    result["air_yards_share_roll4_mean"] = result["air_yards_share_mean"]
    result["wopr_lag1"] = result["wopr_mean"]
    result["wopr_roll4_mean"] = result["wopr_mean"]
    result["racr_lag1"] = result["racr_mean"]
    result["opp_pts_allowed_roll4"] = result["opp_pts_allowed_roll4_mean"]
    result["opp_pts_allowed_season"] = result["opp_pts_allowed_season_mean"]

    for stat in ["completions", "attempts", "passing_yards", "passing_tds", "interceptions",
                 "carries", "rushing_yards", "rushing_tds",
                 "targets", "receptions", "receiving_yards", "receiving_tds"]:
        result[f"{stat}_lag1"] = result[f"{stat}_mean"]

    print(f"  Built weighted summary for {len(result)} players across {len(sorted_seasons)} seasons")
    return result


def compute_vorp(df: pd.DataFrame, scoring: str = "ppr") -> pd.DataFrame:
    col = f"projected_pts_{scoring}"
    df = df.copy()
    baseline = {}
    for pos, rank in REPLACEMENT_RANK.items():
        pos_sorted = df[df["position"] == pos].nlargest(rank, col)
        baseline[pos] = pos_sorted[col].min() if len(pos_sorted) >= rank else 0
    df[f"vorp_{scoring}"] = df.apply(
        lambda r: r[col] - baseline.get(r["position"], 0), axis=1
    )
    return df


def run(source_season: int = 2024) -> pd.DataFrame:
    print(f"Building projections from {source_season} data (3-season weighted)...")
    season_data = load_multi_season_stats(source_season, n_seasons=3)
    summary = build_weighted_summary(season_data, source_season)

    all_rows = []
    metrics_cache = joblib.load(MODEL_DIR / "metrics.pkl")

    for pos in POSITIONS:
        model_path = MODEL_DIR / f"{pos.lower()}_model.pkl"
        if not model_path.exists():
            print(f"  {pos}: no model found, skipping")
            continue

        artifact = joblib.load(model_path)
        model = artifact["model"]
        features = artifact["features"]

        pos_df = summary[summary["position"] == pos].copy()
        for col in [f for f in features if f not in pos_df.columns]:
            pos_df[col] = 0.0

        X = pos_df[features].fillna(0)
        preds = model.predict(X)
        mae = metrics_cache.get(pos, {}).get("mae", 3.0)

        pos_df = pos_df.copy()
        pos_df["projected_pts_ppr"] = np.maximum(preds, 0)
        pos_df["projected_pts_std"] = np.maximum(
            preds - (0 if pos == "QB" else summary[summary["position"] == pos]["receptions_mean"].mean()),
            0
        )
        pos_df["projected_pts_half"] = (pos_df["projected_pts_ppr"] + pos_df["projected_pts_std"]) / 2
        pos_df["confidence_low"] = np.maximum(pos_df["projected_pts_ppr"] - mae * 1.5, 0)
        pos_df["confidence_high"] = pos_df["projected_pts_ppr"] + mae * 1.5

        all_rows.append(pos_df[[
            "player_id", "player_name", "position", "team",
            "projected_pts_ppr", "projected_pts_std", "projected_pts_half",
            "confidence_low", "confidence_high",
        ]])
        print(f"  {pos}: {len(pos_df)} players projected")

    result = pd.concat(all_rows, ignore_index=True)
    result = (
        result.sort_values("projected_pts_ppr", ascending=False)
        .drop_duplicates(subset="player_id", keep="first")
        .reset_index(drop=True)
    )

    result["season"] = source_season + 1
    result["adp"] = None
    result["adp_value_ppr"] = None

    result = compute_vorp(result, "ppr")
    result = compute_vorp(result, "std")

    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE projections RESTART IDENTITY"))
        conn.execute(projections.insert(), result.to_dict(orient="records"))

    print(f"Saved {len(result)} projections.")
    return result


if __name__ == "__main__":
    run()
