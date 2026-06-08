"""
Price prediction model using RandomForestRegressor.

Features: area, front (width), depth, floor, is_no_hau, district (one-hot), street_type
Target: price_per_m2
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from config import MODEL_CACHE_DIR
from db import fetch_all

warnings.filterwarnings("ignore")

MODEL_PATH = os.path.join(MODEL_CACHE_DIR, "price_model.pkl")
ENCODERS_PATH = os.path.join(MODEL_CACHE_DIR, "encoders.pkl")
COLS_PATH = os.path.join(MODEL_CACHE_DIR, "feature_cols.pkl")

_FEATURE_COLS: list[str] = []
_ENCODERS: dict[str, LabelEncoder] = {}
_MODEL: RandomForestRegressor | None = None


def _load_training_data() -> pd.DataFrame:
    rows = fetch_all("""
        SELECT
            source_id, price_per_m2, area, front, depth, floor,
            is_no_hau, district_name, street_type
        FROM crawled_properties
        WHERE price_per_m2 > 0 AND price_per_m2 < 500
          AND area > 10 AND area < 2000
          AND front > 1
          AND district_name IS NOT NULL
    """)
    return pd.DataFrame(rows)


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["front"] = pd.to_numeric(d["front"], errors="coerce").fillna(0)
    d["depth"] = pd.to_numeric(d["depth"], errors="coerce").fillna(0)
    d["floor"] = pd.to_numeric(d["floor"], errors="coerce").fillna(0).clip(0, 50)
    d["area"] = pd.to_numeric(d["area"], errors="coerce").fillna(0)
    d["price_per_m2"] = pd.to_numeric(d["price_per_m2"], errors="coerce")

    d["is_no_hau"] = d["is_no_hau"].fillna(False).astype(int)
    d["street_type"] = d.get("street_type") or "hem_thuong"
    d["district_name"] = d["district_name"].fillna("Unknown")

    d["frontage_ratio"] = d.apply(
        lambda r: r["front"] / r["area"] if r["area"] > 0 else 0, axis=1
    )
    d["floor_area_ratio"] = d.apply(
        lambda r: (r["floor"] * r["front"]) / r["area"] if r["area"] > 0 else 0, axis=1
    )
    return d


def train() -> dict:
    df = _load_training_data()
    if len(df) < 100:
        return {"status": "error", "message": f"Not enough data ({len(df)} rows)"}

    df = _engineer_features(df)
    df = df.dropna(subset=["price_per_m2"])
    if len(df) < 100:
        return {"status": "error", "message": f"Not enough clean data ({len(df)} rows)"}

    global _ENCODERS, _FEATURE_COLS, _MODEL

    district_enc = LabelEncoder()
    street_enc = LabelEncoder()
    df["district_encoded"] = district_enc.fit_transform(df["district_name"])
    df["street_encoded"] = street_enc.fit_transform(df["street_type"])
    _ENCODERS = {"district": district_enc, "street": street_enc}

    feature_cols = [
        "area", "front", "depth", "floor", "is_no_hau",
        "frontage_ratio", "floor_area_ratio",
        "district_encoded", "street_encoded",
    ]
    _FEATURE_COLS = feature_cols

    X = df[feature_cols].values
    y = df["price_per_m2"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300, max_depth=20, min_samples_leaf=5,
        n_jobs=-1, random_state=42, verbose=0
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(_ENCODERS, f)
    with open(COLS_PATH, "wb") as f:
        pickle.dump(feature_cols, f)

    _MODEL = model

    return {
        "status": "ok",
        "samples": len(df),
        "mae": round(mae, 2),
        "r2_score": round(r2, 4),
        "feature_cols": feature_cols,
    }


def load():
    global _MODEL, _ENCODERS, _FEATURE_COLS
    if _MODEL is not None:
        return True
    if not os.path.exists(MODEL_PATH):
        return False
    with open(MODEL_PATH, "rb") as f:
        _MODEL = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        _ENCODERS = pickle.load(f)
    with open(COLS_PATH, "rb") as f:
        _FEATURE_COLS = pickle.load(f)
    return True


def predict(
    area: float, front: float, depth: float, floor: int,
    is_no_hau: bool, district_name: str, street_type: str,
) -> dict | None:
    if not load():
        return None

    feat = {
        "area": area, "front": front, "depth": depth, "floor": floor,
        "is_no_hau": 1 if is_no_hau else 0,
        "frontage_ratio": front / area if area > 0 else 0,
        "floor_area_ratio": (floor * front) / area if area > 0 else 0,
    }

    try:
        district_enc = _ENCODERS["district"]
        street_enc = _ENCODERS["street"]
    except KeyError:
        return None

    seen_districts = list(district_enc.classes_)
    seen_streets = list(street_enc.classes_)

    d_enc = (
        district_enc.transform([district_name])[0]
        if district_name in seen_districts
        else -1
    )
    s_enc = (
        street_enc.transform([street_type])[0]
        if street_type in seen_streets
        else -1
    )

    feat["district_encoded"] = d_enc
    feat["street_encoded"] = s_enc

    X = np.array([[feat[c] for c in _FEATURE_COLS]], dtype=float)

    pred = _MODEL.predict(X)[0]

    # Simple confidence interval using percentile estimation
    trees = np.array([t.predict(X)[0] for t in _MODEL.estimators_])
    lower = float(np.percentile(trees, 10))
    upper = float(np.percentile(trees, 90))

    return {
        "predicted_price_per_m2": round(float(pred), 2),
        "confidence_range": [
            round(max(0, lower), 2),
            round(upper, 2),
        ],
        "district_seen": district_name in seen_districts,
    }
