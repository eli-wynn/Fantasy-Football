"""
FastAPI backend for Draft Scout.
Run: uvicorn backend.api.main:app --reload
"""
import os
import json
import math
from typing import Optional
import redis
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from backend.db import engine, projections, weekly_stats

app = FastAPI(title="Draft Scout API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
try:
    cache = redis.from_url(REDIS_URL, decode_responses=True)
    cache.ping()
except Exception:
    cache = None  # run without cache if Redis unavailable

CACHE_TTL = 3600  # 1 hour


def cached(key: str):
    if cache:
        val = cache.get(key)
        if val:
            return json.loads(val)
    return None


def clean(row: dict) -> dict:
    """Replace float NaN/Inf with None so JSON serialization never fails."""
    return {
        k: (None if isinstance(v, float) and not math.isfinite(v) else v)
        for k, v in row.items()
    }


def set_cache(key: str, data):
    if cache:
        cache.setex(key, CACHE_TTL, json.dumps(data))


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "cache": "connected" if cache else "unavailable"}


# ---------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------

@app.get("/projections")
def get_projections(
    position: Optional[str] = Query(None, description="QB, RB, WR, or TE"),
    season: int = Query(2025),
    scoring: str = Query("ppr", description="ppr, std, or half"),
    limit: int = Query(200),
):
    cache_key = f"projections:{position}:{season}:{scoring}:{limit}"
    if hit := cached(cache_key):
        return hit

    sort_col = {
        "ppr": "projected_pts_ppr",
        "std": "projected_pts_std",
        "half": "projected_pts_half",
    }.get(scoring, "projected_pts_ppr")

    stmt = text(f"""
        SELECT player_id, player_name, position, team, season,
               projected_pts_ppr, projected_pts_std, projected_pts_half,
               confidence_low, confidence_high,
               vorp_ppr, vorp_std, adp, adp_value_ppr
        FROM projections
        WHERE season = :season
        {"AND position = :pos" if position else ""}
        ORDER BY {sort_col} DESC
        LIMIT :limit
    """)

    params = {"season": season, "limit": limit}
    if position:
        params["pos"] = position.upper()

    with engine.connect() as conn:
        rows = conn.execute(stmt, params).mappings().all()

    result = [clean(dict(r)) for r in rows]
    if not result:
        raise HTTPException(404, "No projections found. Run the pipeline first.")

    set_cache(cache_key, result)
    return result


@app.get("/projections/{player_id}")
def get_player_projection(player_id: str, season: int = Query(2025)):
    cache_key = f"proj:{player_id}:{season}"
    if hit := cached(cache_key):
        return hit

    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM projections WHERE player_id = :pid AND season = :s"),
            {"pid": player_id, "s": season},
        ).mappings().first()

    if not row:
        raise HTTPException(404, f"No projection for player_id={player_id}")

    result = clean(dict(row))
    set_cache(cache_key, result)
    return result


# ---------------------------------------------------------------------------
# Historical stats
# ---------------------------------------------------------------------------

@app.get("/stats/{player_id}")
def get_player_stats(player_id: str, season: Optional[int] = Query(None)):
    params: dict = {"pid": player_id}
    season_clause = "AND season = :season" if season else ""
    if season:
        params["season"] = season

    with engine.connect() as conn:
        rows = conn.execute(
            text(f"""
                SELECT * FROM weekly_stats
                WHERE player_id = :pid {season_clause}
                ORDER BY season, week
            """),
            params,
        ).mappings().all()

    if not rows:
        raise HTTPException(404, f"No stats found for player_id={player_id}")
    return [clean(dict(r)) for r in rows]


# ---------------------------------------------------------------------------
# Model accuracy
# ---------------------------------------------------------------------------

@app.get("/model/metrics")
def get_model_metrics():
    from pathlib import Path
    import joblib

    metrics_path = Path(__file__).parent.parent / "models" / "artifacts" / "metrics.pkl"
    if not metrics_path.exists():
        raise HTTPException(404, "Model not yet trained. Run backend/models/train.py first.")
    return joblib.load(metrics_path)


# ---------------------------------------------------------------------------
# Players (profile + headshots)
# ---------------------------------------------------------------------------

@app.get("/players")
def get_players(
    position: Optional[str] = Query(None),
    q: Optional[str] = Query(None, description="Name search"),
):
    """Return player profiles including headshot URLs and age."""
    clauses = []
    params: dict = {}
    if position:
        clauses.append("position = :pos")
        params["pos"] = position.upper()
    if q:
        clauses.append("LOWER(player_name) LIKE :q")
        params["q"] = f"%{q.lower()}%"

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with engine.connect() as conn:
        rows = conn.execute(
            text(f"SELECT * FROM players {where} ORDER BY player_name LIMIT 500"),
            params,
        ).mappings().all()
    return [clean(dict(r)) for r in rows]


@app.get("/players/{player_id}")
def get_player(player_id: str):
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM players WHERE player_id = :pid"),
            {"pid": player_id},
        ).mappings().first()
    if not row:
        raise HTTPException(404, f"Player {player_id} not found")
    return clean(dict(row))


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@app.get("/search")
def search_players(q: str = Query(..., min_length=2), season: int = Query(2025)):
    """Search projections joined with player profile (includes headshot_url)."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT p.player_id, p.player_name, p.position, p.team,
                       p.age, p.headshot_url,
                       pr.projected_pts_ppr, pr.vorp_ppr
                FROM players p
                LEFT JOIN projections pr
                    ON p.player_id = pr.player_id AND pr.season = :s
                WHERE LOWER(p.player_name) LIKE :q
                LIMIT 20
            """),
            {"s": season, "q": f"%{q.lower()}%"},
        ).mappings().all()
    return [clean(dict(r)) for r in rows]
