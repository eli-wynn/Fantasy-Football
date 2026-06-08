"""
Update projections with accurate team assignments and depth chart penalties.

Sources:
- nfl_data_py import_seasonal_rosters: end-of-season team assignments (accurate historical)
- Sleeper API: current depth chart order (who is actually starting in 2025)

Run: python backend/data/fetch_rosters.py
"""
import requests
import nfl_data_py as nfl
from sqlalchemy import text
from backend.db import engine

SKILL_POSITIONS = {"QB", "RB", "WR", "TE"}

DEPTH_PENALTY = {
    "QB": {1: 1.0, 2: 0.12, 3: 0.04},
    "RB": {1: 1.0, 2: 0.40, 3: 0.12},
    # WR and TE depth chart is unreliable for fantasy — slot WRs show as WR2
    # physically but are fantasy WR1s. Use no-ADP cap in blend_adp instead.
    "WR": {1: 1.0, 2: 1.0, 3: 1.0},
    "TE": {1: 1.0, 2: 1.0, 3: 1.0},
}


def normalise(name: str) -> str:
    return name.strip().lower().replace("'", "").replace(".", "")


def fetch_nfl_rosters(season: int = 2025) -> dict:
    """Get end-of-season team assignments from nfl_data_py.
    Falls back to previous season if the requested season is unavailable."""
    print(f"Fetching {season} seasonal rosters from nfl_data_py...")
    try:
        rosters = nfl.import_seasonal_rosters([season])
        if rosters.empty:
            raise ValueError("Empty roster data")
    except Exception as e:
        fallback = season - 1
        print(f"  {season} rosters unavailable ({e}), falling back to {fallback}...")
        rosters = nfl.import_seasonal_rosters([fallback])

    # Keep only skill positions, last week available per player
    rosters = rosters[rosters["position"].isin(SKILL_POSITIONS)].copy()
    rosters = rosters.sort_values("week", ascending=False).drop_duplicates("player_id")

    result = {}  # player_id -> {team, age}
    for _, row in rosters.iterrows():
        result[row["player_id"]] = {
            "team": row["team"],
            "age": row["age"],
        }

    print(f"  {len(result)} skill position players found")
    return result


def fetch_sleeper_depth() -> tuple[dict, dict]:
    """Get current depth chart order from Sleeper."""
    print("Fetching current depth charts from Sleeper...")
    data = requests.get("https://api.sleeper.app/v1/players/nfl").json()

    by_gsis = {}
    by_name = {}

    for player in data.values():
        if player.get("position") not in SKILL_POSITIONS:
            continue
        if player.get("status") not in ("Active", "Inactive"):
            continue

        info = {
            "depth_chart_order": player.get("depth_chart_order") or 99,
            "team": player.get("team"),
            "injury_status": player.get("injury_status"),
        }

        gsis_id = player.get("gsis_id")
        if gsis_id:
            by_gsis[gsis_id.strip()] = info

        full_name = player.get("full_name")
        if full_name:
            by_name[normalise(full_name)] = info

    print(f"  {len(by_gsis)} players with depth chart data")
    return by_gsis, by_name


def update_rosters(nfl_rosters: dict, sleeper_by_gsis: dict, sleeper_by_name: dict):
    updated_team = 0
    penalised = 0
    floored = 0
    age_updated = 0

    with engine.begin() as conn:
        proj_rows = conn.execute(text(
            "SELECT player_id, player_name, position FROM projections"
        )).fetchall()

        for player_id, player_name, position in proj_rows:
            norm_name = normalise(player_name)

            # --- Team assignment: use nfl_data_py only (season-accurate) ---
            nfl_info = nfl_rosters.get(player_id)
            sleeper_info = sleeper_by_gsis.get(player_id) or sleeper_by_name.get(norm_name)

            # Only use nfl_data_py for team — Sleeper reflects 2026 current rosters
            new_team = (nfl_info or {}).get("team")
            if new_team:
                conn.execute(text(
                    "UPDATE projections SET team = :team WHERE player_id = :pid"
                ), {"team": new_team, "pid": player_id})
                updated_team += 1

            # --- Depth chart penalty ---
            depth = (sleeper_info or {}).get("depth_chart_order", 1)
            penalties = DEPTH_PENALTY.get(position, {1: 1.0, 2: 0.3, 3: 0.1})
            multiplier = penalties.get(depth, penalties.get(max(penalties.keys()), 0.05))

            if multiplier < 1.0:
                conn.execute(text("""
                    UPDATE projections SET
                        projected_pts_ppr  = projected_pts_ppr  * :m,
                        projected_pts_std  = projected_pts_std  * :m,
                        projected_pts_half = projected_pts_half * :m,
                        confidence_low     = confidence_low     * :m,
                        confidence_high    = confidence_high    * :m
                    WHERE player_id = :pid
                """), {"m": multiplier, "pid": player_id})
                penalised += 1

        # --- Starter floor: depth 1 players shouldn't project below 70% of position avg ---
        for position in ("QB", "RB", "WR", "TE"):
            avg = conn.execute(text("""
                SELECT AVG(projected_pts_ppr) FROM projections
                WHERE position = :pos AND projected_pts_ppr > 0
            """), {"pos": position}).scalar() or 0

            floor = avg * 0.70

            below_floor = conn.execute(text("""
                SELECT player_id, player_name FROM projections
                WHERE position = :pos AND projected_pts_ppr < :floor
            """), {"pos": position, "floor": floor}).fetchall()

            for pid, pname in below_floor:
                info = sleeper_by_gsis.get(pid) or sleeper_by_name.get(normalise(pname))
                if info and info.get("depth_chart_order") == 1:
                    conn.execute(text("""
                        UPDATE projections SET
                            projected_pts_ppr  = :floor,
                            projected_pts_std  = :floor * 0.85,
                            projected_pts_half = :floor * 0.925
                        WHERE player_id = :pid
                    """), {"floor": floor, "pid": pid})
                    floored += 1

        # --- Recalculate VORP ---
        conn.execute(text("""
            UPDATE projections p1
            SET vorp_ppr = projected_pts_ppr - (
                SELECT COALESCE(MIN(sub.projected_pts_ppr), 0)
                FROM (
                    SELECT projected_pts_ppr FROM projections p2
                    WHERE p2.position = p1.position
                    ORDER BY projected_pts_ppr DESC
                    LIMIT CASE p1.position
                        WHEN 'QB' THEN 12
                        WHEN 'RB' THEN 24
                        WHEN 'WR' THEN 36
                        WHEN 'TE' THEN 12
                        ELSE 12
                    END
                ) sub
            )
        """))

        # --- Update age in players table ---
        player_rows = conn.execute(text(
            "SELECT player_id, player_name FROM players"
        )).fetchall()

        for player_id, player_name in player_rows:
            nfl_info = nfl_rosters.get(player_id)
            if nfl_info and nfl_info.get("age"):
                conn.execute(text(
                    "UPDATE players SET age = :age WHERE player_id = :pid"
                ), {"age": nfl_info["age"], "pid": player_id})
                age_updated += 1

    print(f"  Updated team for {updated_team} players")
    print(f"  Applied depth penalty to {penalised} backups")
    print(f"  Applied starter floor to {floored} players")
    print(f"  Updated age for {age_updated} players")

    try:
        import redis
        r = redis.from_url("redis://localhost:6379")
        r.flushall()
        print("  Cache cleared")
    except Exception:
        pass


if __name__ == "__main__":
    nfl_rosters = fetch_nfl_rosters(2025)
    sleeper_by_gsis, sleeper_by_name = fetch_sleeper_depth()
    update_rosters(nfl_rosters, sleeper_by_gsis, sleeper_by_name)
