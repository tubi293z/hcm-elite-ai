"""
Market analysis — district-level statistics, trends, and insights.
"""

import numpy as np
from db import fetch_all, execute


def analyze_district(district_name: str | None = None) -> list[dict]:
    where = ""
    params: tuple = ()
    if district_name:
        where = "WHERE district_name = %s"
        params = (district_name,)

    rows = fetch_all(f"""
        SELECT
            district_name,
            COUNT(*)                                             AS sample_size,
            ROUND(AVG(price_per_m2)::numeric, 2)                AS avg_price_per_m2,
            ROUND(MIN(price_per_m2)::numeric, 2)                AS min_price_per_m2,
            ROUND(MAX(price_per_m2)::numeric, 2)                AS max_price_per_m2,
            ROUND(AVG(area)::numeric, 2)                        AS avg_area,
            ROUND(AVG(price_absolute)::numeric, 2)              AS avg_price_absolute
        FROM crawled_properties
        {where}
        GROUP BY district_name
        ORDER BY sample_size DESC
    """, params)
    return rows


def _compute_trend(district: str, street_type: str | None) -> str:
    """Simple trend heuristic based on price per m2 variability."""
    where = "district_name = %s"
    params: list = [district]
    if street_type:
        where += " AND street_type = %s"
        params.append(street_type)

    rows = fetch_all(f"""
        SELECT price_per_m2
        FROM crawled_properties
        WHERE {where}
          AND price_per_m2 > 0 AND price_per_m2 < 500
        ORDER BY created DESC NULLS LAST
        LIMIT 500
    """, tuple(params))

    prices = [r["price_per_m2"] for r in rows if r.get("price_per_m2")]
    if len(prices) < 20:
        return "stable", 0.0

    # Split in half to compare recent vs older
    mid = len(prices) // 2
    older = np.mean(prices[mid:])
    recent = np.mean(prices[:mid])

    if older == 0:
        return "stable", 0.0

    change = (recent - older) / older
    if change > 0.05:
        return "up", round(min(abs(change), 1.0), 3)
    elif change < -0.05:
        return "down", round(min(abs(change), 1.0), 3)
    return "stable", round(abs(change), 3)


def generate_district_insights() -> dict:
    districts = analyze_district()
    count = 0
    for d in districts:
        trend, strength = _compute_trend(d["district_name"], None)
        d["price_trend"] = trend
        d["trend_strength"] = strength

        insight = _build_insight(d)
        d["insight"] = insight

        execute("""
            INSERT INTO market_insights
                (district_name, street_type, sample_size,
                 avg_price_per_m2, median_price_per_m2,
                 min_price_per_m2, max_price_per_m2,
                 avg_area, price_trend, trend_strength, insight)
            VALUES (%s, NULL, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            d["district_name"], d["sample_size"],
            d["avg_price_per_m2"], d["avg_price_per_m2"],
            d["min_price_per_m2"], d["max_price_per_m2"],
            d["avg_area"], trend, strength, insight,
        ))
        count += 1

    return {"status": "ok", "districts_analyzed": count, "insights": districts}


def _build_insight(d: dict) -> str:
    district = d["district_name"]
    avg = d.get("avg_price_per_m2", 0)
    trend = d.get("price_trend", "stable")
    strength = d.get("trend_strength", 0)

    parts = [f"📊 {district}: {d['sample_size']} mẫu, giá TB {avg} tr/m²."]

    if avg < 30:
        parts.append("Giá thấp, phù hợp đầu tư sinh lời.")
    elif avg < 60:
        parts.append("Giá trung bình, thị trường ổn định.")
    else:
        parts.append("Giá cao, khu vực trung tâm hoặc đã phát triển.")

    if trend == "up" and strength > 0.1:
        parts.append(f"📈 Xu hướng tăng (độ mạnh {strength}).")
    elif trend == "down" and strength > 0.1:
        parts.append(f"📉 Xu hướng giảm (độ mạnh {strength}).")
    else:
        parts.append("➡️ Thị trường ổn định.")

    if d.get("min_price_per_m2") and avg:
        discount = round((1 - d["min_price_per_m2"] / avg) * 100)
        if discount > 20:
            parts.append(f"💡 Có BĐS rẻ hơn TB tới {discount}% — cơ hội tốt!")

    return " ".join(parts)


def get_latest_insights(district: str | None = None) -> list[dict]:
    where = ""
    params: tuple = ()
    if district:
        where = "WHERE district_name = %s"
        params = (district,)

    return fetch_all(f"""
        SELECT DISTINCT ON (district_name)
            district_name, sample_size,
            avg_price_per_m2, min_price_per_m2, max_price_per_m2,
            avg_area, price_trend, trend_strength, insight,
            generated_at
        FROM market_insights
        {where}
        ORDER BY district_name, generated_at DESC
    """, params)


def find_undervalued_properties() -> list[dict]:
    """Find properties priced significantly below district average."""
    return fetch_all("""
        WITH district_avg AS (
            SELECT district_name, AVG(price_per_m2) AS avg_ppm
            FROM crawled_properties
            WHERE price_per_m2 > 0 AND price_per_m2 < 500
            GROUP BY district_name
        )
        SELECT
            cp.source_id, cp.title, cp.price_absolute, cp.area,
            cp.price_per_m2, cp.district_name, cp.street_type_label,
            ROUND(
                ((cp.price_per_m2 - da.avg_ppm) / da.avg_ppm * 100)::numeric, 1
            ) AS discount_pct,
            cp.front, cp.depth, cp.floor
        FROM crawled_properties cp
        JOIN district_avg da ON da.district_name = cp.district_name
        WHERE cp.price_per_m2 > 0
          AND cp.price_per_m2 < da.avg_ppm * 0.85
          AND cp.area > 20
        ORDER BY discount_pct ASC
        LIMIT 50
    """)
