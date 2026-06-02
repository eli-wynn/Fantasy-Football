"""
Load trained models and generate per-player projections for a target season.
Writes results to the `projections` table and returns a DataFrame.
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sqlalchemy import text
from backend.db import engine, projections, metadata

MODEL_DIR = Path(__file__).parent / "artifacts"
POSITIONS = ["QB", "RB", "WR", "TE"]

# Replacement-level thresholds (last starter at position in a 12-team league)
REPLACEMENT_RANK = {"QB": 12, "RB": 24, "WR": 36, "TE": 12}


def load_recent_stats(season: int) -> pd.DataFrame:
    query = text("""
        SELECT * FROM weekly_stats
        WHERE season = :season
        ORDER BY player_id, week
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"season": season})


def build_player_summary(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Aggregate a season's weekly data into per-player feature rows for prediction."""
    agg = df.groupby(["player_id", "player_name", "position", "team"]).agg(
        games_played=("games_played", "sum"),
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
    ).reset_index()

    agg["season"] = season + 1  # projecting next season
    agg["week"] = 9             # mid-season as a neutral week anchor
    agg["fp_lag1"] = agg["fp_mean"]
    agg["fp_lag2"] = agg["fp_mean"]
    agg["fp_lag3"] = agg["fp_mean"]
    agg["fp_roll4_mean"] = agg["fp_mean"]
    agg["fp_roll4_std"] = agg["fp_std"].fillna(0)
    agg["season_fp_mean"] = agg["fp_mean"]
    agg["season_fp_games"] = agg["games_played"]
    agg["snap_pct"] = agg["snap_pct_mean"]

    for stat in ["completions", "attempts", "passing_yards", "passing_tds", "interceptions",
                 "carries", "rushing_yards", "rushing_tds",
                 "targets", "receptions", "receiving_yards", "receiving_tds"]:
        agg[f"{stat}_lag1"] = agg[f"{stat}_mean"]

    return agg


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
    print(f"Building projections from {source_season} data…")
    recent = load_recent_stats(source_season)
    summary = build_player_summary(recent, source_season)

    all_rows = []
    for pos in POSITIONS:
        model_path = MODEL_DIR / f"{pos.lower()}_model.pkl"
        if not model_path.exists():
            print(f"  {pos}: no model found, skipping")
            continue

        artifact = joblib.load(model_path)
        model = artifact["model"]
        features = artifact["features"]

        pos_df = summary[summary["position"] == pos].copy()
        missing = [f for f in features if f not in pos_df.columns]
        for col in missing:
            pos_df[col] = 0.0

        X = pos_df[features].fillna(0)
        preds = model.predict(X)

        # Rough confidence interval using ±1 residual std from training MAE
        import joblib as _jl
        metrics = _jl.load(MODEL_DIR / "metrics.pkl")
        mae = metrics.get(pos, {}).get("mae", 3.0)

        pos_df = pos_df.copy()
        pos_df["projected_pts_ppr"] = np.maximum(preds, 0)
        pos_df["projected_pts_std"] = np.maximum(preds - recent[
            recent["position"] == pos
        ]["receptions"].mean() * 1.0 if pos != "QB" else preds, 0)
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
