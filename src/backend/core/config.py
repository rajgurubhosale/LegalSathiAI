import os
from pathlib import Path
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[3] / ".env")

missing = [name for name in ("SECRET_KEY", "DATABASE_URL") if not os.getenv(name, "").strip()]
if missing:
    raise RuntimeError(f"Missing required settings in .env: {', '.join(missing)}")

SECRET_KEY = os.environ["SECRET_KEY"].strip()
if len(SECRET_KEY) < 32:
    raise RuntimeError("SECRET_KEY must be at least 32 characters")
    
DATABASE_URL = os.environ["DATABASE_URL"].strip()
ALGORITHM = "HS256"
ACCESS_TOKEN_MINUTES = 30
