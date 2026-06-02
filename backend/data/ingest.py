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
    "season", "week", "opponent_team", "headshot_url",
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


def compute_defensive_strength(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each player-week, compute how many PPR points that player's opponent
    has allowed to their position over the prior 4 weeks and season-to-date.
    Uses a lag so there is zero data leakage.
    """
    print("  Computing opponent defensive strength...")

    # Sum fantasy points scored against each team, by position and week
    allowed = (
        df.groupby(["opponent_team", "season", "week", "position"])["fantasy_points_ppr"]
        .sum()
        .reset_index()
        .rename(columns={"opponent_team": "def_team", "fantasy_points_ppr": "pts_allowed"})
    )

    allowed = allowed.sort_values(["def_team", "position", "season", "week"])
    grp = allowed.groupby(["def_team", "position"])

    # Lag by 1 week so current week is never included
    allowed["pts_allowed_lag1"] = grp["pts_allowed"].shift(1)

    allowed["opp_pts_allowed_roll4"] = (
        grp["pts_allowed_lag1"]
        .transform(lambda x: x.rolling(4, min_periods=1).mean())
    )
    allowed["opp_pts_allowed_season"] = (
        grp["pts_allowed_lag1"]
        .transform(lambda x: x.expanding().mean())
    )

    # Join back to player rows on their opponent
    df = df.merge(
        allowed[["def_team", "season", "week", "position",
                 "opp_pts_allowed_roll4", "opp_pts_allowed_season"]],
        left_on=["opponent_team", "season", "week", "position"],
        right_on=["def_team", "season", "week", "position"],
        how="left",
    )
    df = df.drop(columns=["def_team"], errors="ignore")
    return df


def fetch_snap_data(seasons: list) -> pd.DataFrame:
    try:
        snaps = nfl.import_snap_counts(seasons)
        # offense_pct is already 0-1 in some versions, 0-100 in others
        pct_col = next(
            (c for c in snaps.columns if "offense" in c.lower() and "pct" in c.lower()),
            next((c for c in snaps.columns if "snap" in c.lower() and "pct" in c.lower()), None)
        )
        if not pct_col:
            print(f"  Snap pct column not found. Available: {snaps.columns.tolist()}")
            return pd.DataFrame()

        snaps = snaps[["player", "team", "season", "week", pct_col]].copy()
        snaps.columns = ["player_name_snap", "team", "season", "week", "snap_pct"]

        # Normalise to 0-1 if values look like percentages
        if snaps["snap_pct"].max() > 1.5:
            snaps["snap_pct"] = snaps["snap_pct"] / 100.0

        # Aggregate in case a player has multiple rows per week
        snaps = snaps.groupby(["player_name_snap", "team", "season", "week"])["snap_pct"].mean().reset_index()
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

    # Snap counts — join on name + team + week
    snaps = fetch_snap_data(seasons)
    if not snaps.empty:
        df = df.merge(
            snaps,
            left_on=["player_name", "team", "season", "week"],
            right_on=["player_name_snap", "team", "season", "week"],
            how="left"
        )
        df = df.drop(columns=["player_name_snap"], errors="ignore")
        matched = df["snap_pct"].notna().sum()
        print(f"  Snap counts matched: {matched:,} rows")
    else:
        df["snap_pct"] = None

    for col in FLOAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = compute_defensive_strength(df)

    df = df.drop_duplicates(subset=["player_id", "season", "week"])
    print(f"  -> {len(df):,} rows across {df['player_id'].nunique():,} players")
    return df


def load_players(df_weekly: pd.DataFrame, seasons: list):
    """Populate the players table using data already present in weekly stats."""
    print("Building player profiles from weekly data...")
    cols = ["player_id", "player_name", "position", "team"]
    if "headshot_url" in df_weekly.columns:
        cols.append("headshot_url")

    roster_df = (
        df_weekly.sort_values("season", ascending=False)
        .drop_duplicates("player_id")[cols]
        .copy()
    )
    if "headshot_url" not in roster_df.columns:
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
