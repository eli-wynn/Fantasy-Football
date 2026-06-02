"""
Pull weekly NFL stats via nfl_data_py and load into PostgreSQL.
Run: python -m backend.data.ingest
"""
import pandas as pd
import nfl_data_py as nfl
from sqlalchemy import text
from backend.db import engine, weekly_stats, players, metadata

SEASONS = list(range(2019, 2025))  # 6 seasons of training data
SKILL_POSITIONS = {"QB", "RB", "WR", "TE"}

# Columns we WANT — fetched defensively (missing ones become NaN, not errors)
WANTED_COLS = [
    "player_id", "player_display_name", "position", "recent_team",
    "season", "week",
    "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
    "carries", "rushing_yards", "rushing_tds",
    "targets", "receptions", "receiving_yards", "receiving_tds",
    "target_share", "air_yards_share", "wopr", "racr",
    "sack_fumbles_lost", "rushing_fumbles_lost",
    "fantasy_points", "fantasy_points_ppr",
]

# Some columns have alternate names across seasons
ALTERNATES = {
    "player_display_name": ["player_name", "player_display_name"],
    "wopr": ["wopr", "wopr_x", "wopr_y"],
}

RENAME = {
    "recent_team": "team",
    "player_display_name": "player_name",
    "sack_fumbles_lost": "_sack_fum",
    "rushing_fumbles_lost": "_rush_fum",
    "fantasy_points": "fantasy_points_std",
}

FLOAT_COLS = [
    "age",
    "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
    "carries", "rushing_yards", "rushing_tds",
    "targets", "receptions", "receiving_yards", "receiving_tds",
    "target_share", "air_yards_share", "wopr", "racr", "ppr_sh", "dom",
    "fumbles_lost", "fantasy_points_ppr", "fantasy_points_std", "snap_pct",
]


def resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename alternate column names to our canonical names."""
    for canonical, alts in ALTERNATES.items():
        if canonical not in df.columns:
            for alt in alts:
                if alt in df.columns:
                    df = df.rename(columns={alt: canonical})
                    break
    return df


def fetch_snap_data(seasons: list) -> pd.DataFrame:
    try:
        snaps = nfl.import_snap_counts(seasons)
        # Find the offensive snap % column — name varies by version
        pct_col = next(
            (c for c in snaps.columns if "offense" in c.lower() and "pct" in c.lower()),
            next((c for c in snaps.columns if "snap" in c.lower() and "pct" in c.lower()), None)
        )
        id_col = next(
            (c for c in snaps.columns if "pfr" in c.lower() and "id" in c.lower()),
            next((c for c in snaps.columns if "player_id" in c.lower()), None)
        )
        if not pct_col or not id_col:
            print(f"  Snap count columns not found. Available: {snaps.columns.tolist()}")
            return pd.DataFrame()
        snaps = snaps[[id_col, "season", "week", pct_col]].copy()
        snaps.columns = ["pfr_player_id", "season", "week", "snap_pct"]
        snaps["snap_pct"] = snaps["snap_pct"] / 100.0
        return snaps
    except Exception as e:
        print(f"  Snap count fetch failed (non-fatal): {e}")
        return pd.DataFrame()


def build_weekly(seasons: list) -> pd.DataFrame:
    print(f"Fetching weekly stats for seasons {seasons[0]}-{seasons[-1]}...")

    # Fetch ALL columns — safer than requesting specific ones
    raw = nfl.import_weekly_data(seasons)
    print(f"  Raw columns available: {sorted(raw.columns.tolist())}")

    raw = resolve_columns(raw)

    # Keep only skill positions
    df = raw[raw["position"].isin(SKILL_POSITIONS)].copy()

    # Grab headshot_url while it's here (weekly data includes it)
    if "headshot_url" in df.columns:
        df["_headshot_url"] = df["headshot_url"]

    # Grab wanted columns that actually exist; fill missing ones with NaN
    present = [c for c in WANTED_COLS if c in df.columns]
    missing = [c for c in WANTED_COLS if c not in df.columns]
    if missing:
        print(f"  Columns not found (will be NULL): {missing}")
    df = df[present].copy()
    for col in missing:
        df[col] = None

    df = df.rename(columns=RENAME)

    # Combine fumble types
    df["fumbles_lost"] = df.get("_sack_fum", pd.Series(0, index=df.index)).fillna(0) + \
                         df.get("_rush_fum", pd.Series(0, index=df.index)).fillna(0)
    df = df.drop(columns=["_sack_fum", "_rush_fum"], errors="ignore")

    # ppr_sh and dom: compute from raw if not present
    if "ppr_sh" not in df.columns:
        df["ppr_sh"] = None
    if "dom" not in df.columns:
        df["dom"] = None

    # age: not in weekly data — will come from rosters join later
    if "age" not in df.columns:
        df["age"] = None

    # Snap counts (best effort)
    snaps = fetch_snap_data(seasons)
    if not snaps.empty:
        df = df.merge(snaps, left_on=["player_id", "season", "week"],
                      right_on=["pfr_player_id", "season", "week"], how="left")
        df = df.drop(columns=["pfr_player_id"], errors="ignore")
    else:
        df["snap_pct"] = None

    for col in FLOAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates(subset=["player_id", "season", "week"])
    print(f"  -> {len(df):,} rows across {df['player_id'].nunique():,} players")
    return df


def load_players(df_weekly: pd.DataFrame, seasons: list):
    """Populate the players table using data already present in weekly stats."""
    print("Building player profiles from weekly data...")
    cols = ["player_id", "player_name", "position", "team"]
    if "_headshot_url" in df_weekly.columns:
        cols.append("_headshot_url")

    roster_df = (
        df_weekly.sort_values("season", ascending=False)
        .drop_duplicates("player_id")[cols]
        .copy()
    )
    if "_headshot_url" in roster_df.columns:
        roster_df = roster_df.rename(columns={"_headshot_url": "headshot_url"})
    else:
        roster_df["headshot_url"] = None

    roster_df["age"] = None
    roster_df["years_exp"] = None
    roster_df["college"] = None

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE players"))
        conn.execute(players.insert(), roster_df.to_dict(orient="records"))
    print(f"  -> {len(roster_df)} player profiles saved")


def load_weekly(df: pd.DataFrame):
    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE weekly_stats RESTART IDENTITY"))
        conn.execute(weekly_stats.insert(), df.to_dict(orient="records"))
    print(f"  -> Loaded {len(df):,} rows into weekly_stats")


def run():
    df = build_weekly(SEASONS)
    load_weekly(df)
    load_players(df, SEASONS)
    print("Ingest complete.")


if __name__ == "__main__":
    run()
