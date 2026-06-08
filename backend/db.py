import os
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, MetaData, Table, Text
)
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://draftscout:draftscout@localhost:5432/draftscout")

engine = create_engine(DATABASE_URL)
metadata = MetaData()

players = Table(
    "players",
    metadata,
    Column("player_id", String, primary_key=True),
    Column("player_name", String, nullable=False),
    Column("position", String),
    Column("team", String),
    Column("age", Float),
    Column("years_exp", Integer),
    Column("college", String),
    Column("headshot_url", Text),
)

weekly_stats = Table(
    "weekly_stats",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("player_id", String, nullable=False),
    Column("player_name", String, nullable=False),
    Column("position", String, nullable=False),
    Column("team", String),
    Column("age", Float),
    Column("season", Integer, nullable=False),
    Column("week", Integer, nullable=False),
    Column("games_played", Integer),
    # Passing
    Column("completions", Float),
    Column("attempts", Float),
    Column("passing_yards", Float),
    Column("passing_tds", Float),
    Column("interceptions", Float),
    # Rushing
    Column("carries", Float),
    Column("rushing_yards", Float),
    Column("rushing_tds", Float),
    # Receiving
    Column("targets", Float),
    Column("receptions", Float),
    Column("receiving_yards", Float),
    Column("receiving_tds", Float),
    # Opportunity / efficiency shares
    Column("target_share", Float),      # targets / team targets
    Column("air_yards_share", Float),   # share of team air yards
    Column("wopr", Float),              # weighted opportunity rating
    Column("racr", Float),              # receiver air conversion ratio
    Column("ppr_sh", Float),            # share of team PPR points
    Column("dom", Float),               # dominator rating (RB: rush + rec share)
    # Misc
    Column("fumbles_lost", Float),
    Column("fantasy_points_ppr", Float),
    Column("fantasy_points_std", Float),
    Column("snap_pct", Float),
    # Opponent defensive strength
    Column("opponent_team", String),
    Column("opp_pts_allowed_roll4", Float),   # 4-week rolling avg pts allowed to this position
    Column("opp_pts_allowed_season", Float),  # season-to-date avg pts allowed to this position
    # Vegas lines
    Column("implied_team_total", Float),      # how many points Vegas expects this team to score
    Column("game_total", Float),              # combined over/under for the game
    Column("temp", Float),                    # game temperature (fahrenheit)
    Column("wind", Float),                    # wind speed (mph)
)

projections = Table(
    "projections",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("player_id", String, nullable=False),
    Column("player_name", String, nullable=False),
    Column("position", String, nullable=False),
    Column("team", String),
    Column("season", Integer, nullable=False),
    Column("projected_pts_ppr", Float),
    Column("projected_pts_std", Float),
    Column("projected_pts_half", Float),
    Column("confidence_low", Float),
    Column("confidence_high", Float),
    Column("vorp_ppr", Float),
    Column("vorp_std", Float),
    Column("adp", Float),
    Column("adp_value_ppr", Float),
)


def init_db():
    metadata.create_all(engine)
    print("Tables created.")


if __name__ == "__main__":
    init_db()
