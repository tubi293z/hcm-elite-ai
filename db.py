import psycopg2
import psycopg2.extras
from contextlib import contextmanager
from config import DB_URL

@contextmanager
def get_db():
    conn = psycopg2.connect(DB_URL, sslmode="require")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def fetch_all(sql: str, params: tuple = ()) -> list[dict]:
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

def fetch_one(sql: str, params: tuple = ()) -> dict | None:
    rows = fetch_all(sql, params)
    return rows[0] if rows else None

def execute(sql: str, params: tuple = ()) -> None:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
