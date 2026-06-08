# HCM Elite AI Service

Python microservice for real estate price prediction, similar property search, and market analysis.

## Stack

- **FastAPI** — REST API
- **scikit-learn** — RandomForest price prediction model
- **sentence-transformers** — Property embeddings (384-dim, all-MiniLM-L6-v2)
- **pgvector** — Vector similarity search in Supabase Postgres

## Features

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/train` | POST | Train price model + generate all embeddings |
| `/predict` | POST | Predict property price per m² |
| `/similar` | POST | Find similar properties by source_id or text |
| `/market/district` | GET | Market insights by district |
| `/market/analyze` | POST | Generate district analysis |
| `/market/undervalued` | GET | Properties below market price |
| `/health` | GET | Health check |

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set database URL
cp .env.example .env
# Edit .env with your Supabase DB connection string

# Train the model (loads data from crawled_properties table)
curl -X POST http://localhost:8000/train

# Start the service
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Deploy to Railway

1. Push this repo to GitHub
2. On Railway: **New Project → Deploy from GitHub repo**
3. Set env var `SUPABASE_DB_URL` in Railway dashboard
4. After deploy, call `POST /train` to initialize the model

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SUPABASE_DB_URL` | Yes | Postgres connection string (port 6543 for Supabase) |
| `MODEL_CACHE_DIR` | No | Cache directory (default: `./models`) |
