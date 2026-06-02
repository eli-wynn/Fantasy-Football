# Draft Scout

AI-powered fantasy football draft assistant. Trains an XGBoost model on 6 seasons of NFL data to project fantasy points, calculate VORP, and surface value picks in real time during your draft.

## Architecture

```
nfl_data_py → PostgreSQL → XGBoost model → FastAPI → React UI
```

## Quick start

**Prerequisites:** Python 3.11+, Docker

```bash
# 1. Start Postgres + Redis
docker compose up -d

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Copy env vars
cp backend/.env.example backend/.env

# 4. Run the full pipeline
make pipeline   # ingest → train → predict

# 5. Start the API
make api        # http://localhost:8000/docs
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/projections` | All player projections (filter by position, scoring) |
| GET | `/projections/{player_id}` | Single player projection |
| GET | `/stats/{player_id}` | Historical weekly stats |
| GET | `/model/metrics` | MAE / RMSE per position |
| GET | `/search?q=` | Player name search |

## Project phases

- [x] Phase 1 — Data pipeline + XGBoost model + FastAPI
- [ ] Phase 2 — React frontend with sortable player table
- [ ] Phase 3 — Sleeper API integration + live draft board + VORP
- [ ] Phase 4 — Trade analyzer + deploy - use Fantasy Calc API for trade value considerations along with own predictions

## Data sources

- **[nfl_data_py](https://github.com/nflverse/nfl_data_py)** — weekly stats, snap counts (2019–2024)
- **Sleeper API** — live draft picks, league rosters
- **ESPN Fantasy API** — ADP data for value signals

## Model performance

Run `make api` then visit `/model/metrics` — or see the Jupyter notebook in `notebooks/`.
