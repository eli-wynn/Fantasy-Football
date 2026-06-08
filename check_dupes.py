from backend.db import engine
from sqlalchemy import text

with engine.connect() as conn:
    rows = conn.execute(text("""
        SELECT adp, COUNT(*) as cnt
        FROM projections
        WHERE adp IS NOT NULL
        GROUP BY adp
        HAVING COUNT(*) > 1
        ORDER BY cnt DESC
        LIMIT 15
    """)).fetchall()
    print("ADP values with duplicates:")
    for r in rows:
        print(f"  ADP {r[0]}: {r[1]} players")
