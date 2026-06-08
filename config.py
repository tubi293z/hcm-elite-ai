import os
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("SUPABASE_DB_URL", "")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./models")

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
