"""
Property embeddings for similarity search.

Uses sentence-transformers/all-MiniLM-L6-v2 to encode property
(title + description + features) into 384-dim vectors, stored
in Supabase pgvector for cosine similarity search.
"""

import os
import pickle
import warnings
from typing import Any
import numpy as np

from config import MODEL_CACHE_DIR
from db import fetch_all, execute, fetch_one

warnings.filterwarnings("ignore")

MODEL_NAME = "all-MiniLM-L6-v2"
ENCODER_CACHE = os.path.join(MODEL_CACHE_DIR, "embedding_model.pkl")

_model = None


def _get_model():
    global _model
    if _model is not None:
        return _model
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer(MODEL_NAME)
    return _model


def _property_to_text(p: dict[str, Any]) -> str:
    parts = []
    if p.get("title"):
        parts.append(p["title"])
    if p.get("description"):
        parts.append(p["description"][:500])
    attrs = p.get("attribute")
    if attrs and isinstance(attrs, (list, str)):
        if isinstance(attrs, list):
            parts.extend(attrs)
        else:
            parts.append(str(attrs))
    meta = []
    for k in ["area", "front", "depth", "floor"]:
        v = p.get(k)
        if v is not None:
            meta.append(f"{k}={v}")
    if meta:
        parts.append(" | ".join(meta))
    text = ". ".join(parts) or "no description"
    return text


def generate_all() -> dict:
    rows = fetch_all("""
        SELECT source_id, title, description, attribute,
               area, front, depth, floor
        FROM crawled_properties
        WHERE source_id IS NOT NULL
    """)
    if not rows:
        return {"status": "error", "message": "No properties found"}

    model = _get_model()
    batch_size = 64
    total = 0

    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        texts = [_property_to_text(p) for p in batch]
        embeddings = model.encode(texts, show_progress_bar=False)

        values = []
        for p, emb in zip(batch, embeddings):
            source_id = p["source_id"]
            vec = ",".join(str(round(v, 6)) for v in emb)
            values.append(f"('{source_id}', '[{vec}]'::vector(384))")

        if values:
            sql = f"""
                INSERT INTO property_embeddings (source_id, embedding)
                VALUES {','.join(values)}
                ON CONFLICT (source_id)
                DO UPDATE SET embedding = EXCLUDED.embedding, updated_at = now()
            """
            execute(sql)
        total += len(batch)

    return {"status": "ok", "embedded": total}


def find_similar(
    source_id: str, limit: int = 10
) -> list[dict]:
    row = fetch_one(
        "SELECT embedding FROM property_embeddings WHERE source_id = %s",
        (source_id,),
    )
    if not row:
        similar = find_similar_by_text({"title": source_id, "description": ""})
        return similar

    vec_str = "[" + ",".join(str(x) for x in row["embedding"]) + "]"
    sql = f"""
        SELECT
            cp.source_id, cp.title, cp.price_absolute, cp.area,
            cp.price_per_m2, cp.district_name, cp.street_type_label,
            cp.front, cp.depth, cp.floor,
            1 - (pe.embedding <=> '{vec_str}'::vector) AS similarity
        FROM property_embeddings pe
        JOIN crawled_properties cp ON cp.source_id = pe.source_id
        WHERE pe.source_id != %s
        ORDER BY pe.embedding <=> '{vec_str}'::vector
        LIMIT %s
    """
    return fetch_all(sql, (source_id, limit))


def find_similar_by_text(
    query: dict, limit: int = 10
) -> list[dict]:
    model = _get_model()
    text = _property_to_text(query)
    vec = model.encode([text])[0]
    vec_str = "[" + ",".join(str(round(v, 6)) for v in vec) + "]"

    sql = f"""
        SELECT
            cp.source_id, cp.title, cp.price_absolute, cp.area,
            cp.price_per_m2, cp.district_name, cp.street_type_label,
            cp.front, cp.depth, cp.floor,
            1 - (pe.embedding <=> '{vec_str}'::vector) AS similarity
        FROM property_embeddings pe
        JOIN crawled_properties cp ON cp.source_id = pe.source_id
        ORDER BY pe.embedding <=> '{vec_str}'::vector
        LIMIT %s
    """
    return fetch_all(sql, (limit,))
