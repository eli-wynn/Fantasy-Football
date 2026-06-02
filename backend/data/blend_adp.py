"""
Blend model projections with ADP consensus.
When ADP strongly disagrees with the model, adjust the projection toward consensus.
Run: python -m backend.data.blend_adp
"""
from sqlalchemy import text
from backend.db import engine

# How much weight to give ADP vs model (0.0 = pure model, 1.0 = pure ADP)
ADP_BLEND_WEIGHT = 0.35


def blend():
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT
                player_id,
                position,
                projected_pts_ppr,
                adp,
                ROW_NUMBER() OVER (ORDER BY projected_pts_ppr DESC) as model_rank
            FROM projections
            WHERE adp IS NOT NULL
            ORDER BY projected_pts_ppr DESC
        """)).fetchall()

        if not rows:
            print("No players with ADP found. Run fetch_adp.py first.")
            return

        adp_sorted = sorted(rows, key=lambda r: r[3])
        model_pts_by_rank = [r[2] for r in rows]

        blended = 0
        for i, row in enumerate(adp_sorted):
            player_id, position, model_pts, adp, model_rank = row
            adp_rank = i + 1

            adp_implied_pts = model_pts_by_rank[min(adp_rank - 1, len(model_pts_by_rank) - 1)]
            blended_pts = (
                (1 - ADP_BLEND_WEIGHT) * model_pts +
                ADP_BLEND_WEIGHT * adp_implied_pts
            )

            if abs(blended_pts - model_pts) / max(model_pts, 0.1) > 0.20:
                conn.execute(text("""
                    UPDATE projections SET projected_pts_ppr = :pts
                    WHERE player_id = :pid
                """), {"pts": round(blended_pts, 4), "pid": player_id})
                blended += 1

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

    try:
        import redis
        r = redis.from_url("redis://localhost:6379")
        r.flushall()
        print("Cache cleared")
    except Exception:
        pass

    print(f"Blended ADP into projections for {blended} players")


if __name__ == "__main__":
    blend()
