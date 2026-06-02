import requests
from sqlalchemy import text
from backend.db import engine


def fetch_sleeper_adp():
    url = "https://api.sleeper.app/v1/players/nfl"
    response = requests.get(url)
    data = response.json()
    print(f"Total players returned: {len(data)}")

    adp_by_gsis = {}   # gsis_id -> rank
    adp_by_name = {}   # "firstname lastname" lowercase -> rank

    for player in data.values():
        search_rank = player.get("search_rank")
        if not search_rank or search_rank >= 9999999:
            continue

        # Primary match: gsis_id
        gsis_id = player.get("gsis_id")
        if gsis_id:
            adp_by_gsis[gsis_id.strip()] = search_rank

        # Fallback match: full name normalised (lowercase, no apostrophes/periods)
        full_name = player.get("full_name")
        if full_name:
            normalised = full_name.strip().lower().replace("'", "").replace(".", "")
            adp_by_name[normalised] = search_rank

    print(f"Players with valid ADP: {len(adp_by_gsis)} by ID, {len(adp_by_name)} by name")
    return adp_by_gsis, adp_by_name


def update_adp(adp_by_gsis, adp_by_name):
    updated_id = 0
    updated_name = 0

    with engine.begin() as conn:
        # Clear existing ADP data first
        conn.execute(text("UPDATE projections SET adp = NULL, adp_value_ppr = NULL"))

        # Pass 1: match by gsis_id
        for gsis_id, rank in adp_by_gsis.items():
            result = conn.execute(
                text("UPDATE projections SET adp = :adp WHERE player_id = :pid"),
                {"adp": rank, "pid": gsis_id}
            )
            updated_id += result.rowcount

        # Pass 2: name fallback for players still missing ADP
        missing = conn.execute(text(
            "SELECT player_id, player_name FROM projections WHERE adp IS NULL"
        )).fetchall()

        for player_id, player_name in missing:
            normalised = player_name.strip().lower().replace("'", "").replace(".", "")
            rank = adp_by_name.get(normalised)
            if rank:
                conn.execute(
                    text("UPDATE projections SET adp = :adp WHERE player_id = :pid"),
                    {"adp": rank, "pid": player_id}
                )
                updated_name += 1

        # Calculate ADP value: positive = model likes more than consensus
        conn.execute(text("""
            UPDATE projections p1
            SET adp_value_ppr = adp - (
                SELECT COUNT(*) + 1
                FROM projections p2
                WHERE p2.projected_pts_ppr > p1.projected_pts_ppr
            )
            WHERE adp IS NOT NULL
        """))

    try:
        import redis
        r = redis.from_url("redis://localhost:6379")
        r.flushall()
        print("Cache cleared")
    except Exception:
        pass

    print(f"Updated {updated_id} by player_id, {updated_name} by name fallback")
    print(f"Total with ADP: {updated_id + updated_name}")


if __name__ == "__main__":
    adp_by_gsis, adp_by_name = fetch_sleeper_adp()
    update_adp(adp_by_gsis, adp_by_name)
