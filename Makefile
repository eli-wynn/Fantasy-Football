PYTHON = .venv/Scripts/python
PIP    = .venv/Scripts/pip

.PHONY: up down install ingest train predict api pipeline

up:
	docker compose up -d

down:
	docker compose down

install:
	py -3.9 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r backend/requirements.txt

ingest:
	$(PYTHON) -m backend.data.ingest

train:
	$(PYTHON) -m backend.models.train

predict:
	$(PYTHON) -m backend.models.predict

api:
	.venv/Scripts/uvicorn backend.api.main:app --reload --port 8000

# Run the full pipeline from scratch
pipeline: ingest train predict
