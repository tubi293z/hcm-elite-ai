"""
FastAPI microservice for AI-powered real estate analysis.

Endpoints:
  POST /train              — Train price prediction model + generate embeddings
  POST /predict            — Predict price for a property
  POST /similar            — Find similar properties by source_id or text query
  GET  /market/district    — Market insights by district
  GET  /market/undervalued — Properties priced below market
  GET  /health             — Health check
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import price_model
import embeddings
import market as market_analysis

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ai-service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("AI service starting...")
    price_model.load()
    yield
    log.info("AI service shutting down.")


app = FastAPI(
    title="RE-Poster Elite AI Service",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ──

class PredictRequest(BaseModel):
    area: float = Field(gt=0, description="Diện tích (m²)")
    front: float = Field(default=0, ge=0, description="Chiều ngang (m)")
    depth: float = Field(default=0, ge=0, description="Chiều dài (m)")
    floor: int = Field(default=0, ge=0, description="Số tầng")
    is_no_hau: bool = False
    district_name: str = Field(default="Unknown")
    street_type: str = Field(default="hem_thuong")


class SimilarBySourceRequest(BaseModel):
    source_id: str
    limit: int = Field(default=10, ge=1, le=50)


class SimilarByTextRequest(BaseModel):
    title: str = Field(default="")
    description: str = Field(default="")
    attribute: str | list[str] | None = None
    area: float | None = None
    front: float | None = None
    depth: float | None = None
    floor: int | None = None
    limit: int = Field(default=10, ge=1, le=50)


# ── Routes ──

@app.post("/train")
def train():
    result_p = price_model.train()
    result_e = embeddings.generate_all()
    return {
        "price_model": result_p,
        "embeddings": result_e,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        result = price_model.predict(
            area=req.area, front=req.front, depth=req.depth,
            floor=req.floor, is_no_hau=req.is_no_hau,
            district_name=req.district_name, street_type=req.street_type,
        )
    except Exception as e:
        log.exception("Predict failed")
        raise HTTPException(status_code=500, detail=str(e))

    if result is None:
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Call POST /train first.",
        )
    return result


@app.post("/similar")
def similar(body: SimilarBySourceRequest | SimilarByTextRequest):
    try:
        if isinstance(body, SimilarBySourceRequest):
            results = embeddings.find_similar(body.source_id, body.limit)
        else:
            results = embeddings.find_similar_by_text(
                body.model_dump(), body.limit
            )
    except Exception as e:
        log.exception("Similar search failed")
        raise HTTPException(status_code=500, detail=str(e))

    return {"results": results, "count": len(results)}


@app.get("/market/district")
def market_district(district: str | None = None):
    try:
        if district:
            return {"insights": market_analysis.get_latest_insights(district)}
        return {"insights": market_analysis.get_latest_insights()}
    except Exception as e:
        log.exception("Market district analysis failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/market/analyze")
def market_analyze():
    try:
        return market_analysis.generate_district_insights()
    except Exception as e:
        log.exception("Market analysis failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/market/undervalued")
def market_undervalued():
    try:
        return {
            "results": market_analysis.find_undervalued_properties(),
        }
    except Exception as e:
        log.exception("Undervalued search failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": price_model.load(),
    }
