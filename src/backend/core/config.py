import os
from pathlib import Path
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[3] / ".env")

missing = [name for name in ("SECRET_KEY", "DATABASE_URL") if not os.getenv(name, "").strip()]
if missing:
    raise RuntimeError(f"Missing required settings in .env: {', '.join(missing)}")

SECRET_KEY = os.environ["SECRET_KEY"].strip()
DATABASE_URL = os.environ["DATABASE_URL"].strip()
ALGORITHM = "HS256"
ACCESS_TOKEN_MINUTES = 30
